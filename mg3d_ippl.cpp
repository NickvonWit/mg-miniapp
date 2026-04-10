// Mini-app: Matrix-Free Multigrid-Preconditioned CG for 3D Poisson (independent nx,ny,nz)
// -----------------------------------------------------------------------------
// Serial C++17, single-file, educational. No deps beyond the STL.
// Domain: [0,1]^3 with homogeneous Dirichlet BCs (u=0 on boundary).
// Discretization: 7-point FD Laplacian, allowing hx!=hy!=hz.
// Hierarchy: geometric 2:1 coarsening per-dimension until coarsest <= 3^3.
// Smoother: weighted Jacobi implemented as u <- u + ω D^{-1} (f - A u).
// Transfers: full-weighting restriction (3D), trilinear prolongation.
// Preconditioner: one V-cycle used in right-preconditioned CG (PCG).
// Matrix-free: no sparse matrices are assembled; A*v is applied on-the-fly.
// Manufactured test: u=sin(pi x) sin(pi y) sin(pi z), f=3 pi^2 u.
//
// Build:
//   g++ -O3 -std=c++17 miniapp_multigrid_preconditioner_poisson3d.cpp -o mg3d_pcg
// Run:
//   ./mg3d_pcg -nx 65 -ny 97 -nz 129 -nu1 2 -nu2 2 -w 0.8 -tol 1e-8 -maxit 200
// -----------------------------------------------------------------------------

/** \file miniapp_multigrid_preconditioner_poisson3d.cpp
 *  \brief Matrix-free MG-preconditioned CG for the 3D Poisson equation on \f$[0,1]^3\f$.
 *
 *  \details
 *  \par PDE and BCs
 *  Solve \f$-\Delta u = f\f$ in \f$\Omega=[0,1]^3\f$ with homogeneous Dirichlet boundary
 *  conditions \f$u|_{\partial\Omega}=0\f$.
 *
 *  \par Discretization (matrix-free)
 *  Tensor grid sizes \f$(n_x,n_y,n_z)\f$ imply spacings \f$h_x=1/(n_x-1)\f$, etc.
 *  The anisotropic 7-point Laplacian on interior nodes is
 *  \f[\begin{aligned}
 *    (Au)_{i,j,k} &= \left(\tfrac{2}{h_x^2}+\tfrac{2}{h_y^2}+\tfrac{2}{h_z^2}\right)u_{i,j,k}
 *    - \tfrac{u_{i-1,j,k}+u_{i+1,j,k}}{h_x^2}
 *    - \tfrac{u_{i,j-1,k}+u_{i,j+1,k}}{h_y^2}
 *    - \tfrac{u_{i,j,k-1}+u_{i,j,k+1}}{h_z^2}.
 *  \end{aligned}\f]
 *  No sparse matrix is assembled; see ::MultiGrid3D::apply_A.
 *
 *  \par Multigrid V-cycle (right preconditioner)
 *  Weighted Jacobi smoother, 3D full-weighting restriction, trilinear prolongation,
 *  geometric 2:1 coarsening per dimension, approximate coarse solve by Jacobi sweeps.
 *
 *  \par References
 *  - Briggs, Henson, McCormick, \emph{A Multigrid Tutorial}, SIAM (2000).
 *  - Trottenberg, Oosterlee, Schüller, \emph{Multigrid}, Academic (2000).
 *  - Saad, \emph{Iterative Methods for Sparse Linear Systems}, SIAM (2003).
 *  - Hackbusch, \emph{Multi-Grid Methods and Applications}, Springer (1985).
 */

#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <IpplCore.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "Types/Vector.h"

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "Field/Field.h"
#include "Field/BConds.h"

#include "Index/NDIndex.h"
#include "Kokkos_Macros.hpp"
#include "Meshes/UniformCartesian.h"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"

/** \brief Row-major 3D index mapping.
 *  \param i x-index (0..nx-1), \param j y-index (0..ny-1), \param k z-index (0..nz-1)
 *  \param nx leading dimension (x), \param ny size in y
 *  \return linear index \f$\mathrm{id} = i + n_x\,(j + n_y\,k)\f$.
 */
static inline int id3(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}
constexpr unsigned Dim = 3;
using Mesh_t           = ippl::UniformCartesian<double, Dim>;
using Centering_t      = Mesh_t::DefaultCentering;
using Field_t          = ippl::Field<double, Dim, Mesh_t, Centering_t>;
using BC_t             = ippl::BConds<Field_t, Dim>;

/** \brief Geometry-only description of one multigrid level.
 *  \details Level \f$\ell\f$ stores only sizes and spacings
 *  \f$(n_x^{(\ell)},n_y^{(\ell)},n_z^{(\ell)})\f$ and
 *  \f$(h_x^{(\ell)},h_y^{(\ell)},h_z^{(\ell)})\f$.
 */
struct Level3D {
    int nx = 0, ny = 0, nz = 0;     //!< grid points per dimension (including boundaries)
    double hx = 1, hy = 1, hz = 1;  //!< spacing per dimension
};

struct LevelField3D {
    ippl::Vector<int, 3> nx        = {0, 0, 0};
    ippl::Vector<double, 3> hx     = {1, 1, 1};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    ippl::NDIndex<Dim> global_domain;
    ippl::FieldLayout<Dim> layout;
    Mesh_t mesh;

    Field_t u, f, r, t;
    BC_t bc;

    LevelField3D(int nx_, int ny_, int nz_)
        : nx({nx_, ny_, nz_})
        , hx({1.0 / double(nx_ - 1), 1.0 / double(ny_ - 1), 1.0 / double(nz_ - 1)})
        , global_domain(ippl::Index(nx_), ippl::Index(ny_), ippl::Index(nz_))
        , layout(MPI_COMM_WORLD, global_domain, {true, true, true})
        , mesh(global_domain, hx, origin)
        , u(mesh, layout)
        , f(mesh, layout)
        , r(mesh, layout)
        , t(mesh, layout) {
        for (unsigned face = 0; face < 2 * Dim; ++face) {
            bc[face] = std::make_shared<ippl::ZeroFace<Field_t>>(face);
        }

        u.setFieldBC(bc);
        f.setFieldBC(bc);
        r.setFieldBC(bc);
        t.setFieldBC(bc);

        u = 0.0;
        f = 0.0;
        r = 0.0;
        t = 0.0;
    }
};
/** \brief Matrix-free geometric multigrid hierarchy with V-cycle preconditioner.
 *  \details
 *  \par Operator
 *  Anisotropic FD Laplacian applied by ::apply_A with no matrix assembly.
 *
 *  \par Workspaces
 *  Per-level vectors (uL,fL,rL,tL) are scratch buffers used during preconditioning.
 *
 *  \par Parameters
 *  \c nu1 / \c nu2 are pre/post smoothing steps; \c omega is the Jacobi weight.
 */
struct MultiGrid3D {
    std::vector<LevelField3D> L;  //!< L[0] finest
    int nu1 = 2, nu2 = 2;         //!< pre/post smoothing steps
    double omega = 0.8;           //!< weighted-Jacobi parameter
    Inform dbg   = Inform("dbg", std::cout);

    /** \brief Construct hierarchy and allocate workspaces.
     *  \param nx_f finest grid points in x
     *  \param ny_f finest grid points in y
     *  \param nz_f finest grid points in z
     *  \param nu1_ pre-smoothing sweeps
     *  \param nu2_ post-smoothing sweeps
     *  \param omega_ Jacobi weight \f$\omega\f$
     */
    MultiGrid3D(int nx_f, int ny_f, int nz_f, int nu1_ = 2, int nu2_ = 2, double omega_ = 0.8)
        : nu1(nu1_)
        , nu2(nu2_)
        , omega(omega_) {
        build_hierarchy(nx_f, ny_f, nz_f);
        dbg.setMessageLevel(1);
        // allocate_workspace();
    }

    /** \brief Build 2:1 coarsening hierarchy per dimension until any dim \f$\le 3\f$.
     *  \details Each coarsening step maps \f$n\mapsto (n-1)/2 + 1\f$ (keeping boundaries).
     */
    void build_hierarchy(const int nx_f, const int ny_f, const int nz_f) {
        Inform out("hierarchy", std::cout);
        out.setMessageLevel(2);
        out << "Building hierarchy." << endl;

        if (nx_f < 3 || ny_f < 3 || nz_f < 3) {
            std::cerr << "Grid too small. Use dims >= 3.\n";
            std::exit(1);
        }

        auto half = [](int n) {
            return (n - 1) / 2 + 1;
        };

        // First count levels so vector never reallocates.
        int nlevels = 0;
        {
            int nx = nx_f, ny = ny_f, nz = nz_f;
            while (true) {
                ++nlevels;
                if (nx <= 3 || ny <= 3 || nz <= 3) {
                    break;
                }
                nx = std::max(3, half(nx));
                ny = std::max(3, half(ny));
                nz = std::max(3, half(nz));
            }
        }

        L.clear();
        L.reserve(nlevels);

        int nx = nx_f, ny = ny_f, nz = nz_f;
        while (true) {
            L.emplace_back(nx, ny, nz);

            if (nx <= 3 || ny <= 3 || nz <= 3) {
                break;
            }

            nx = std::max(3, half(nx));
            ny = std::max(3, half(ny));
            nz = std::max(3, half(nz));
        }

        out << "Built hierarchy." << endl;
    }
    // /** \brief Allocate level-wise work vectors (u,f,r,t). */
    // void allocate_workspace() {
    //     size_t Ls = L.size();
    //     uL.resize(Ls);
    //     fL.resize(Ls);
    //     rL.resize(Ls);
    //     tL.resize(Ls);
    //     for (size_t ell = 0; ell < Ls; ++ell) {
    //         const int N = L[ell].nx * L[ell].ny * L[ell].nz;
    //         uL[ell].assign(N, 0.0);
    //         fL[ell].assign(N, 0.0);
    //         rL[ell].assign(N, 0.0);
    //         tL[ell].assign(N, 0.0);
    //     }
    // }

    /** \brief Apply anisotropic 3D FD Laplacian \f$A=-\Delta_h\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u input vector of nodal values
     *  \param out output vector storing \f$Au\f$
     *  \note Dirichlet boundaries are fixed; updates are applied only to interior nodes.
     */
    static void apply_A(const LevelField3D& lev, Field_t& u, Field_t& out) {
        Inform mes("Apply A", std::cout);
        mes.setMessageLevel(2);
        mes << "Applying A." << endl;

        const int nx = lev.nx[0];
        const int ny = lev.nx[1];
        const int nz = lev.nx[2];

        const double ihx2 = 1.0 / (lev.hx[0] * lev.hx[0]);
        const double ihy2 = 1.0 / (lev.hx[1] * lev.hx[1]);
        const double ihz2 = 1.0 / (lev.hx[2] * lev.hx[2]);
        const double diag = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;

        const auto lDom  = lev.layout.getLocalNDIndex();
        const int nghost = u.getNghost();

        u.fillHalo();
        out = 0.0;

        const auto uview = u.getView();
        auto outview     = out.getView();

        Kokkos::parallel_for(
            "apply_A", out.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                // Outside physical domain: ignore.
                if (ig < 0 || ig >= nx || jg < 0 || jg >= ny || kg < 0 || kg >= nz) {
                    return;
                }

                // Dirichlet boundary nodes are fixed to zero.
                if (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1) {
                    outview(i, j, k) = 0.0;
                    return;
                }

                // Strictly interior point: 7-point stencil is safe.
                outview(i, j, k) = diag * uview(i, j, k)
                                   - ihx2 * (uview(i - 1, j, k) + uview(i + 1, j, k))
                                   - ihy2 * (uview(i, j - 1, k) + uview(i, j + 1, k))
                                   - ihz2 * (uview(i, j, k - 1) + uview(i, j, k + 1));
            });

        ippl::fence();
    }
    /** \brief Compute residual \f$r = f - Au\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u current iterate
     *  \param f right-hand side
     *  \param r output residual
     *  \param Au_scratch temporary storage for \f$Au\f$
     */
    static void residual(const LevelField3D& lev, Field_t& u, Field_t& f, Field_t& r,
                         Field_t& Au_scratch) {
        Inform mes("Residual", std::cout);
        mes.setMessageLevel(2);
        mes << "Calculating residual." << endl;

        apply_A(lev, u, Au_scratch);

        const auto fview  = f.getView();
        const auto Auview = Au_scratch.getView();
        auto rview        = r.getView();

        Kokkos::parallel_for(
            "residual", r.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                rview(i, j, k) = fview(i, j, k) - Auview(i, j, k);
            });

        ippl::fence();
    }
    /** \brief Weighted-Jacobi smoother: \f$u \leftarrow u + \omega D^{-1}(f-Au)\f$.
     *  \param lev level geometry
     *  \param u in/out iterate
     *  \param f right-hand side
     *  \param r_scratch residual workspace
     *  \param Au_scratch workspace for \f$Au\f$
     *  \param iters number of sweeps
     *  \param omega damping \f$\omega\in(0,1)\f$
     *  \details Uses the constant diagonal \f$D=2/h_x^2+2/h_y^2+2/h_z^2\f$ on the level.
     */
    void smooth_jacobi(const LevelField3D& lev, Field_t& u, Field_t& f, Field_t& r_scratch,
                       Field_t& Au_scratch, int iters, double omega) {
        Inform mes("Jacobi", std::cout);
        mes.setMessageLevel(2);
        mes << "Applying jacobi." << endl;

        const int nx = lev.nx[0];
        const int ny = lev.nx[1];
        const int nz = lev.nx[2];

        const double ihx2 = 1.0 / (lev.hx[0] * lev.hx[0]);
        const double ihy2 = 1.0 / (lev.hx[1] * lev.hx[1]);
        const double ihz2 = 1.0 / (lev.hx[2] * lev.hx[2]);
        const double diag = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;

        const auto lDom  = lev.layout.getLocalNDIndex();
        const int nghost = u.getNghost();

        auto uview       = u.getView();
        const auto rview = r_scratch.getView();

        for (int it = 0; it < iters; ++it) {
            residual(lev, u, f, r_scratch, Au_scratch);

            Kokkos::parallel_for(
                "jacobi_update", u.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;

                    if (ig <= 0 || ig >= nx - 1 || jg <= 0 || jg >= ny - 1 || kg <= 0
                        || kg >= nz - 1) {
                        return;
                    }

                    uview(i, j, k) += omega * (rview(i, j, k) / diag);
                });

            ippl::fence();
        }
    }
    /** \brief 3D full-weighting restriction: coarse RHS = \f$R\,r_f\f$.
     *  \param fine fine level geometry
     *  \param r_f fine residual
     *  \param coarse coarse level geometry
     *  \param f_c output coarse RHS
     *  \details Weights relative to center (2I,2J,2K): center 8; faces 4; edges 2; corners 1; then
     * divide by 64.
     */
    static void restrict_fullweight(const LevelField3D& fine, Field_t& r_f,
                                    const LevelField3D& coarse, Field_t& f_c) {
        Inform mes("Restrict", std::cout);
        mes.setMessageLevel(2);
        mes << "Restricting." << endl;

        const int nxc = coarse.nx[0];
        const int nyc = coarse.nx[1];
        const int nzc = coarse.nx[2];

        r_f.fillHalo();
        f_c = 0.0;

        const auto lDomF = fine.layout.getLocalNDIndex();
        const auto lDomC = coarse.layout.getLocalNDIndex();

        const int nghF = r_f.getNghost();
        const int nghC = f_c.getNghost();

        const auto rf = r_f.getView();
        auto fc       = f_c.getView();

        Kokkos::parallel_for(
            "restrict_fullweight", f_c.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int I, const int J, const int K) {
                const int Ig = I + lDomC[0].first() - nghC;
                const int Jg = J + lDomC[1].first() - nghC;
                const int Kg = K + lDomC[2].first() - nghC;

                // Outside physical coarse domain: ignore / keep zero.
                if (Ig < 0 || Ig >= nxc || Jg < 0 || Jg >= nyc || Kg < 0 || Kg >= nzc) {
                    return;
                }

                // Dirichlet boundary on coarse grid.
                if (Ig == 0 || Ig == nxc - 1 || Jg == 0 || Jg == nyc - 1 || Kg == 0
                    || Kg == nzc - 1) {
                    fc(I, J, K) = 0.0;
                    return;
                }

                // Corresponding fine-grid center in global coordinates.
                const int ig = 2 * Ig;
                const int jg = 2 * Jg;
                const int kg = 2 * Kg;

                // Convert fine global coordinates to fine local storage coordinates.
                const int i = ig - lDomF[0].first() + nghF;
                const int j = jg - lDomF[1].first() + nghF;
                const int k = kg - lDomF[2].first() + nghF;

                double sum = 0.0;

                for (int dk = -1; dk <= 1; ++dk) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int di = -1; di <= 1; ++di) {
                            const int adi = (di < 0) ? -di : di;
                            const int adj = (dj < 0) ? -dj : dj;
                            const int adk = (dk < 0) ? -dk : dk;

                            const int wclass = adi + adj + adk;
                            const double w   = (wclass == 0)   ? 8.0
                                               : (wclass == 1) ? 4.0
                                               : (wclass == 2) ? 2.0
                                                               : 1.0;

                            sum += w * rf(i + di, j + dj, k + dk);
                        }
                    }
                }

                fc(I, J, K) = sum / 64.0;
            });

        ippl::fence();
    }
    /** \brief Trilinear prolongation: fine correction += \f$P\,u_c\f$.
     *  \param coarse coarse level geometry
     *  \param u_c coarse correction
     *  \param fine fine level geometry
     *  \param u_f in/out fine correction (accumulated)
     */
    static void prolong_add(const LevelField3D& coarse, Field_t& u_c, const LevelField3D& fine,
                            Field_t& u_f) {
        Inform mes("Prolong", std::cout);
        mes.setMessageLevel(2);
        mes << "Prolonging." << endl;

        const int nxc = coarse.nx[0];
        const int nyc = coarse.nx[1];
        const int nzc = coarse.nx[2];

        const int nxf = fine.nx[0];
        const int nyf = fine.nx[1];
        const int nzf = fine.nx[2];

        u_c.fillHalo();

        const auto lDomC = coarse.layout.getLocalNDIndex();
        const auto lDomF = fine.layout.getLocalNDIndex();

        const int nghC = u_c.getNghost();
        const int nghF = u_f.getNghost();

        const auto uc = u_c.getView();
        auto uf       = u_f.getView();

        Kokkos::parallel_for(
            "prolong_add", u_f.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + lDomF[0].first() - nghF;
                const int jg = j + lDomF[1].first() - nghF;
                const int kg = k + lDomF[2].first() - nghF;

                // Outside physical fine domain.
                if (ig < 0 || ig >= nxf || jg < 0 || jg >= nyf || kg < 0 || kg >= nzf) {
                    return;
                }

                // Do not touch physical Dirichlet boundaries.
                if (ig == 0 || ig == nxf - 1 || jg == 0 || jg == nyf - 1 || kg == 0
                    || kg == nzf - 1) {
                    return;
                }

                // For each dimension:
                // even fine index  -> coincides with a coarse node
                // odd  fine index  -> midpoint between two coarse nodes
                const int I0g = ig / 2;
                const int J0g = jg / 2;
                const int K0g = kg / 2;

                const bool xOdd = (ig & 1);
                const bool yOdd = (jg & 1);
                const bool zOdd = (kg & 1);

                const int I1g = xOdd ? (I0g + 1) : I0g;
                const int J1g = yOdd ? (J0g + 1) : J0g;
                const int K1g = zOdd ? (K0g + 1) : K0g;

                const double wx0 = xOdd ? 0.5 : 1.0;
                const double wx1 = xOdd ? 0.5 : 0.0;

                const double wy0 = yOdd ? 0.5 : 1.0;
                const double wy1 = yOdd ? 0.5 : 0.0;

                const double wz0 = zOdd ? 0.5 : 1.0;
                const double wz1 = zOdd ? 0.5 : 0.0;

                const int I0 = I0g - lDomC[0].first() + nghC;
                const int I1 = I1g - lDomC[0].first() + nghC;
                const int J0 = J0g - lDomC[1].first() + nghC;
                const int J1 = J1g - lDomC[1].first() + nghC;
                const int K0 = K0g - lDomC[2].first() + nghC;
                const int K1 = K1g - lDomC[2].first() + nghC;

                const double corr =
                    wx0 * wy0 * wz0 * uc(I0, J0, K0) + wx1 * wy0 * wz0 * uc(I1, J0, K0)
                    + wx0 * wy1 * wz0 * uc(I0, J1, K0) + wx1 * wy1 * wz0 * uc(I1, J1, K0)
                    + wx0 * wy0 * wz1 * uc(I0, J0, K1) + wx1 * wy0 * wz1 * uc(I1, J0, K1)
                    + wx0 * wy1 * wz1 * uc(I0, J1, K1) + wx1 * wy1 * wz1 * uc(I1, J1, K1);

                uf(i, j, k) += corr;
            });

        ippl::fence();
    }
    /** \brief One matrix-free V-cycle on level \f$\ell\f$ using workspace vectors.
     *  \details Steps: \f$\nu_1\f$ pre-smooth → residual → restrict → coarse solve → prolongate →
     * \f$\nu_2\f$ post-smooth.
     */
    void vcycle(int ell) {
        dbg << "Starting vcycle at level " << ell << "." << endl;
        LevelField3D& lev = L[ell];
        auto &u = lev.u, &f = lev.f, &r = lev.r, &t = lev.t;
        if (ell == (int)L.size() - 1) {
            smooth_jacobi(lev, u, f, r, t, 50, omega);  // coarsest pseudo-solve
            dbg << "Pseudo-solved level " << ell << "." << endl;
            return;
        }
        dbg << "    Starting jacobi smoothing." << endl;
        smooth_jacobi(lev, u, f, r, t, nu1, omega);
        dbg << "    Calculating residual." << endl;
        residual(lev, u, f, r, t);
        dbg << "    Restricting." << endl;
        restrict_fullweight(L[ell], r, L[ell + 1], L[ell + 1].f);
        L[ell + 1].u = 0.0;
        vcycle(ell + 1);
        dbg << "Prolong adding error at level " << ell << "." << endl;
        prolong_add(L[ell + 1], L[ell + 1].u, L[ell], u);
        dbg << "    Starting jacobi smoothing." << endl;
        smooth_jacobi(lev, u, f, r, t, nu2, omega);
        dbg << "Finished vcycle at level " << ell << "." << endl;
    }

    /** \brief Apply preconditioner: \f$z = M^{-1} r\f$ using one V-cycle (matrix-free).
     *  \param r_finest input residual on finest grid
     *  \param z_finest output preconditioned vector
     */
    void apply_precond(const Field_t& r_finest, Field_t& z_finest) {
        L[0].f = r_finest;
        for (size_t ell = 0; ell < L.size(); ++ell)
            L[ell].u = 0.0;
        vcycle(0);
        z_finest = L[0].u;
    }
};

/** \brief Preconditioned Conjugate Gradient (PCG) for SPD systems using MG V-cycles as right
 * preconditioner.
 *  \details Algorithm (Saad, Ch. 9):
 *  \f[
 *   r_0=f-Au_0,\ z_0=M^{-1}r_0,\ p_0=z_0,\
 *   \alpha_k=\frac{r_k^T z_k}{p_k^T A p_k},\
 *   u_{k+1}=u_k+\alpha_k p_k,\
 *   r_{k+1}=r_k-\alpha_k A p_k,\
 *   z_{k+1}=M^{-1}r_{k+1},\
 *   \beta_{k+1}=\frac{r_{k+1}^T z_{k+1}}{r_k^T z_k},\
 *   p_{k+1}=z_{k+1}+\beta_{k+1}p_k.
 *  \f]
 */
struct PCG {
    int maxit  = 200;
    double tol = 1e-8;

    static double dot(const Field_t& a, const Field_t& b) {
        const auto av = a.getView();
        const auto bv = b.getView();

        double local = 0.0;
        Kokkos::parallel_reduce(
            "pcg_dot", a.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                sum += av(i, j, k) * bv(i, j, k);
            },
            local);

        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global;
    }

    static double nrm2(const Field_t& x) { return std::sqrt(dot(x, x)); }

    static void axpy(Field_t& y, double a, const Field_t& x) {
        auto yv       = y.getView();
        const auto xv = x.getView();

        Kokkos::parallel_for(
            "pcg_axpy", y.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                yv(i, j, k) += a * xv(i, j, k);
            });
        ippl::fence();
    }

    static void assign_diff(Field_t& y, const Field_t& a, const Field_t& b) {
        auto yv       = y.getView();
        const auto av = a.getView();
        const auto bv = b.getView();

        Kokkos::parallel_for(
            "pcg_assign_diff", y.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                yv(i, j, k) = av(i, j, k) - bv(i, j, k);
            });
        ippl::fence();
    }

    static void update_search_dir(Field_t& p, const Field_t& z, double beta) {
        auto pv       = p.getView();
        const auto zv = z.getView();

        Kokkos::parallel_for(
            "pcg_update_p", p.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                pv(i, j, k) = zv(i, j, k) + beta * pv(i, j, k);
            });
        ippl::fence();
    }

    int solve(LevelField3D& levA, Field_t& u, const Field_t& f, MultiGrid3D& mg,
              double& final_relres) {
        Field_t r(levA.mesh, levA.layout);
        Field_t z(levA.mesh, levA.layout);
        Field_t p(levA.mesh, levA.layout);
        Field_t Ap(levA.mesh, levA.layout);

        r.setFieldBC(levA.bc);
        z.setFieldBC(levA.bc);
        p.setFieldBC(levA.bc);
        Ap.setFieldBC(levA.bc);

        r  = 0.0;
        z  = 0.0;
        p  = 0.0;
        Ap = 0.0;

        MultiGrid3D::apply_A(levA, u, Ap);
        assign_diff(r, f, Ap);

        double normf = nrm2(f);
        if (normf == 0.0)
            normf = 1.0;

        double rel = nrm2(r) / normf;
        if (rel < tol) {
            final_relres = rel;
            return 0;
        }

        mg.apply_precond(r, z);
        p = z;

        double rz_old = dot(r, z);

        int it = 0;
        for (; it < maxit; ++it) {
            MultiGrid3D::apply_A(levA, p, Ap);

            const double pAp = dot(p, Ap);
            if (std::abs(pAp) < 1e-300) {
                break;
            }

            const double alpha = rz_old / pAp;

            axpy(u, alpha, p);
            axpy(r, -alpha, Ap);

            rel = nrm2(r) / normf;
            if (rel < tol) {
                ++it;
                break;
            }

            mg.apply_precond(r, z);

            const double rz_new = dot(r, z);
            const double beta   = rz_new / rz_old;

            update_search_dir(p, z, beta);
            rz_old = rz_new;
        }

        final_relres = rel;
        return it;
    }
};
/** \brief Manufactured 3D test: \f$u=\sin(\pi x)\sin(\pi y)\sin(\pi z)\f$, \f$f=3\pi^2 u\f$
 * (Dirichlet consistent). */
void fill_manufactured(LevelField3D& lev, Field_t& f, Field_t& u_exact) {
    Inform dbg("manufactured", std::cout);
    dbg.setMessageLevel(1);
    dbg << "Filling exact field." << endl;

    const double pi = 3.14159265358979323846;

    const int nx = lev.nx[0];
    const int ny = lev.nx[1];
    const int nz = lev.nx[2];

    const double hx = lev.hx[0];
    const double hy = lev.hx[1];
    const double hz = lev.hx[2];

    const auto lDom  = lev.layout.getLocalNDIndex();
    const int nghost = f.getNghost();

    auto fview   = f.getView();
    auto uexview = u_exact.getView();

    Kokkos::parallel_for(
        "fill_manufactured", f.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int ig = i + lDom[0].first() - nghost;
            const int jg = j + lDom[1].first() - nghost;
            const int kg = k + lDom[2].first() - nghost;

            const double x = ig * hx;
            const double y = jg * hy;
            const double z = kg * hz;

            const bool boundary =
                (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1);

            const double uex = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);

            uexview(i, j, k) = uex;
            fview(i, j, k)   = boundary ? 0.0 : 3.0 * pi * pi * uex;
        });
    dbg << "Filled exact field." << endl;
    ippl::fence();
}

/** \brief Manufactured 3D test on \f$[0,1]^3\f$ with homogeneous Dirichlet BCs.
 *
 *  \details Uses the exact solution
 *  \f[
 *    u(x,y,z)=
 *    \sin(\pi x)\sin(\pi y)\sin(\pi z)
 *    + 0.25\,\sin(4\pi x)\sin(3\pi y)\sin(5\pi z)
 *    + 0.10\,\sin(7\pi x)\sin(6\pi y)\sin(2\pi z),
 *  \f]
 *  which satisfies \f$u=0\f$ on \f$\partial[0,1]^3\f$.
 *  The corresponding right-hand side for \f$-\Delta u = f\f$ is
 *  \f[
 *    f(x,y,z)=
 *    3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)
 *    + 0.25\cdot 50\pi^2 \sin(4\pi x)\sin(3\pi y)\sin(5\pi z)
 *    + 0.10\cdot 89\pi^2 \sin(7\pi x)\sin(6\pi y)\sin(2\pi z).
 *  \f]
 *  This is harder than the single-mode manufactured solution because it excites
 *  multiple spatial frequencies while remaining smooth and exactly Dirichlet consistent.
 */
void fill_manufactured_hard(LevelField3D& lev, Field_t& f, Field_t& u_exact) {
    Inform dbg("manufactured_hard", std::cout);
    dbg.setMessageLevel(1);
    dbg << "Filling exact field." << endl;

    const double pi = 3.14159265358979323846;

    const int nx = lev.nx[0];
    const int ny = lev.nx[1];
    const int nz = lev.nx[2];

    const double hx = lev.hx[0];
    const double hy = lev.hx[1];
    const double hz = lev.hx[2];

    const auto lDom  = lev.layout.getLocalNDIndex();
    const int nghost = f.getNghost();

    auto fview   = f.getView();
    auto uexview = u_exact.getView();

    Kokkos::parallel_for(
        "fill_manufactured_hard", f.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int ig = i + lDom[0].first() - nghost;
            const int jg = j + lDom[1].first() - nghost;
            const int kg = k + lDom[2].first() - nghost;

            if (ig < 0 || ig >= nx || jg < 0 || jg >= ny || kg < 0 || kg >= nz) {
                uexview(i, j, k) = 0.0;
                fview(i, j, k)   = 0.0;
                return;
            }

            const bool boundary =
                (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1);

            const double x = ig * hx;
            const double y = jg * hy;
            const double z = kg * hz;

            const double s111 = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);

            const double s435 =
                std::sin(4.0 * pi * x) * std::sin(3.0 * pi * y) * std::sin(5.0 * pi * z);

            const double s762 =
                std::sin(7.0 * pi * x) * std::sin(6.0 * pi * y) * std::sin(2.0 * pi * z);

            const double uex = 1.0 * s111 + 0.25 * s435 + 0.10 * s762;

            const double rhs = pi * pi * (3.0 * s111 + 0.25 * 50.0 * s435 + 0.10 * 89.0 * s762);

            uexview(i, j, k) = boundary ? 0.0 : uex;
            fview(i, j, k)   = boundary ? 0.0 : rhs;
        });

    ippl::fence();
    dbg << "Filled exact field." << endl;
}

/** \\brief Non-separable polynomial manufactured 3D test on \\f$[0,1]^3\\f$ with
 *  homogeneous Dirichlet BCs.
 *
 *  \\details Uses the exact solution
 *  \\f[
 *    u(x,y,z)=x(1-x)\\,y(1-y)\\,z(1-z)\\,\\bigl(1+x+2y+3z+xy+yz+zx\\bigr),
 *  \\f]
 *  which satisfies \\f$u=0\\f$ on \\f$\\partial[0,1]^3\\f$.
 *  The right-hand side is \\f$f=-\\Delta u\\f$, evaluated analytically in factored form.
 *  This test is harder than a single-mode sine manufactured solution because it is
 *  non-separable and contains mixed polynomial terms while remaining smooth and
 *  exactly Dirichlet consistent.
 */
void fill_manufactured_poly(LevelField3D& lev, Field_t& f, Field_t& u_exact) {
    Inform dbg("manufactured_poly", std::cout);
    dbg.setMessageLevel(1);
    dbg << "Filling exact field." << endl;

    const int nx = lev.nx[0];
    const int ny = lev.nx[1];
    const int nz = lev.nx[2];

    const double hx = lev.hx[0];
    const double hy = lev.hx[1];
    const double hz = lev.hx[2];

    const auto lDom  = lev.layout.getLocalNDIndex();
    const int nghost = f.getNghost();

    auto fview   = f.getView();
    auto uexview = u_exact.getView();

    Kokkos::parallel_for(
        "fill_manufactured_poly", f.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const int ig = i + lDom[0].first() - nghost;
            const int jg = j + lDom[1].first() - nghost;
            const int kg = k + lDom[2].first() - nghost;

            if (ig < 0 || ig >= nx || jg < 0 || jg >= ny || kg < 0 || kg >= nz) {
                uexview(i, j, k) = 0.0;
                fview(i, j, k)   = 0.0;
                return;
            }

            const double x = ig * hx;
            const double y = jg * hy;
            const double z = kg * hz;

            const bool boundary =
                (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1);

            const double a = x * (1.0 - x);
            const double b = y * (1.0 - y);
            const double c = z * (1.0 - z);

            const double ap = 1.0 - 2.0 * x;
            const double bp = 1.0 - 2.0 * y;
            const double cp = 1.0 - 2.0 * z;

            const double p  = 1.0 + x + 2.0 * y + 3.0 * z + x * y + y * z + z * x;
            const double px = 1.0 + y + z;
            const double py = 2.0 + x + z;
            const double pz = 3.0 + x + y;

            const double uex = a * b * c * p;

            // Since a'' = b'' = c'' = -2 and p_xx = p_yy = p_zz = 0:
            // u_xx = b c (-2 p + 2 a' p_x)
            // u_yy = a c (-2 p + 2 b' p_y)
            // u_zz = a b (-2 p + 2 c' p_z)
            const double rhs = b * c * (2.0 * p - 2.0 * ap * px) + a * c * (2.0 * p - 2.0 * bp * py)
                               + a * b * (2.0 * p - 2.0 * cp * pz);

            uexview(i, j, k) = boundary ? 0.0 : uex;
            fview(i, j, k)   = boundary ? 0.0 : rhs;
        });

    ippl::fence();
    dbg << "Filled exact field." << endl;
}
/** \brief Command-line arguments. */
struct Args {
    int nx = 65, ny = 65, nz = 65;
    int nu1 = 2, nu2 = 2;
    double w   = 0.8;
    double tol = 1e-8;
    int maxit  = 200;
};

/** \brief Parse command line: -nx -ny -nz -nu1 -nu2 -w -tol -maxit. */
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-nx") && i + 1 < argc)
            a.nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-ny") && i + 1 < argc)
            a.ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-nz") && i + 1 < argc)
            a.nz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-nu1") && i + 1 < argc)
            a.nu1 = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-nu2") && i + 1 < argc)
            a.nu2 = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-w") && i + 1 < argc)
            a.w = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-tol") && i + 1 < argc)
            a.tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-maxit") && i + 1 < argc)
            a.maxit = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::cout << "\nUsage: ./mg3d_pcg -nx <int> -ny <int> -nz <int> -nu1 <int> -nu2 <int> "
                         "-w <omega> -tol <tol> -maxit <int>\n";
            std::exit(0);
        }
    }
    return a;
}

/** \brief Driver: assemble manufactured problem and solve with matrix-free MG-preconditioned CG. */
int main(int argc, char** argv) {
    // Init ippl
    ippl::initialize(argc, argv);

    ippl::Info->setOutputLevel(0);
    ippl::Warn->setOutputLevel(1);
    ippl::Error->setOutputLevel(1);

    Args args = parse_args(argc, argv);
    if (args.nx < 3 || args.ny < 3 || args.nz < 3) {
        std::cerr << "All dims must be >=3\n";
        ippl::finalize();
        return 1;
    }
    {
        Inform msg("init");
        msg.setPrintNode(INFORM_ALL_NODES);
        msg << "Hello from IPPL, rank=" << ippl::Comm->rank() << ", size=" << ippl::Comm->size()
            << endl;

        // Build geometry + allocate matrix-free MG workspace
        MultiGrid3D mg(args.nx, args.ny, args.nz, args.nu1, args.nu2, args.w);

        LevelField3D& Alev = mg.L[0];

        Field_t u(Alev.mesh, Alev.layout);
        Field_t f(Alev.mesh, Alev.layout);
        Field_t uex(Alev.mesh, Alev.layout);

        u.setFieldBC(Alev.bc);
        f.setFieldBC(Alev.bc);
        uex.setFieldBC(Alev.bc);

        u   = 0.0;
        f   = 0.0;
        uex = 0.0;

        fill_manufactured_poly(Alev, f, uex);

        if (ippl::Comm->rank() == 0) {
            std::cout << "3D Matrix-Free MG-preconditioned CG for -Δu=f on [0,1]^3\n";
            std::cout << "Grid: nx=" << Alev.nx[0] << ", ny=" << Alev.nx[1] << ", nz=" << Alev.nx[2]
                      << ", hx=" << Alev.hx[0] << ", hy=" << Alev.hx[1] << ", hz=" << Alev.hx[2]
                      << "\n";
            std::cout << "Smoother: weighted Jacobi (ω=" << args.w << ", nu1=" << args.nu1
                      << ", nu2=" << args.nu2 << ")\n";
        }

        static IpplTimings::TimerRef pcg_timer = IpplTimings::getTimer("pcg_timer");

        PCG pcg;
        pcg.maxit = args.maxit;
        pcg.tol   = args.tol;

        double rel = 1.0;

        IpplTimings::startTimer(pcg_timer);
        auto t0 = std::chrono::high_resolution_clock::now();
        int it  = pcg.solve(Alev, u, f, mg, rel);
        auto t1 = std::chrono::high_resolution_clock::now();
        IpplTimings::stopTimer(pcg_timer);

        double sec = std::chrono::duration<double>(t1 - t0).count();

        const int nx = Alev.nx[0];
        const int ny = Alev.nx[1];
        const int nz = Alev.nx[2];

        const auto lDom  = Alev.layout.getLocalNDIndex();
        const int nghost = u.getNghost();

        const auto uview   = u.getView();
        const auto uexview = uex.getView();

        double e2_local = 0.0;
        double u2_local = 0.0;

        Kokkos::parallel_reduce(
            "relL2_error", u.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                const bool boundary =
                    (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1);

                if (!boundary) {
                    const double e = uview(i, j, k) - uexview(i, j, k);
                    sum += e * e;
                }
            },
            e2_local);

        Kokkos::parallel_reduce(
            "relL2_exact_norm", uex.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                const bool boundary =
                    (ig == 0 || ig == nx - 1 || jg == 0 || jg == ny - 1 || kg == 0 || kg == nz - 1);

                if (!boundary) {
                    const double v = uexview(i, j, k);
                    sum += v * v;
                }
            },
            u2_local);

        double e2 = 0.0;
        double u2 = 0.0;
        MPI_Allreduce(&e2_local, &e2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&u2_local, &u2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        const double relL2 = std::sqrt(e2) / std::max(1e-300, std::sqrt(u2));

        if (ippl::Comm->rank() == 0) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "PCG iters: " << it << ", final rel. residual: " << rel
                      << ", rel. L2 error: " << relL2 << "\n";
            std::cout << "Wall time: " << sec << " s\n";
        }
        IpplTimings::print();
    }
    // Finalize ippl
    ippl::finalize();
    return 0;
}
