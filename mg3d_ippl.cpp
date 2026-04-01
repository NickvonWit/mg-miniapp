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
#include <vector>

#include "Utility/IpplTimings.h"

/** \brief Row-major 3D index mapping.
 *  \param i x-index (0..nx-1), \param j y-index (0..ny-1), \param k z-index (0..nz-1)
 *  \param nx leading dimension (x), \param ny size in y
 *  \return linear index \f$\mathrm{id} = i + n_x\,(j + n_y\,k)\f$.
 */
static inline int id3(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

/** \brief Geometry-only description of one multigrid level.
 *  \details Level \f$\ell\f$ stores only sizes and spacings
 *  \f$(n_x^{(\ell)},n_y^{(\ell)},n_z^{(\ell)})\f$ and
 *  \f$(h_x^{(\ell)},h_y^{(\ell)},h_z^{(\ell)})\f$.
 */
struct Level3D {
    int nx = 0, ny = 0, nz = 0;     //!< grid points per dimension (including boundaries)
    double hx = 1, hy = 1, hz = 1;  //!< spacing per dimension
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
    std::vector<Level3D> L;  //!< L[0] finest
    int nu1 = 2, nu2 = 2;    //!< pre/post smoothing steps
    double omega = 0.8;      //!< weighted-Jacobi parameter

    // Workspace per level (matrix-free; no A stored)
    std::vector<std::vector<double> > uL;  //!< level-wise correction buffers
    std::vector<std::vector<double> > fL;  //!< level-wise RHS buffers
    std::vector<std::vector<double> > rL;  //!< level-wise residual scratch
    std::vector<std::vector<double> > tL;  //!< level-wise Au scratch

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
        allocate_workspace();
    }

    /** \brief Build 2:1 coarsening hierarchy per dimension until any dim \f$\le 3\f$.
     *  \details Each coarsening step maps \f$n\mapsto (n-1)/2 + 1\f$ (keeping boundaries).
     */
    void build_hierarchy(int nx_f, int ny_f, int nz_f) {
        int nx = nx_f, ny = ny_f, nz = nz_f;
        if (nx < 3 || ny < 3 || nz < 3) {
            std::cerr << "Grid too small. Use dims >= 3.\n";
            std::exit(1);
        }
        while (true) {
            Level3D lev;
            lev.nx = nx;
            lev.ny = ny;
            lev.nz = nz;
            lev.hx = 1.0 / (nx - 1);
            lev.hy = 1.0 / (ny - 1);
            lev.hz = 1.0 / (nz - 1);
            L.push_back(lev);
            if (nx <= 3 || ny <= 3 || nz <= 3)
                break;
            auto half = [](int n) {
                return (n - 1) / 2 + 1;
            };
            nx = std::max(3, half(nx));
            ny = std::max(3, half(ny));
            nz = std::max(3, half(nz));
        }
    }

    /** \brief Allocate level-wise work vectors (u,f,r,t). */
    void allocate_workspace() {
        size_t Ls = L.size();
        uL.resize(Ls);
        fL.resize(Ls);
        rL.resize(Ls);
        tL.resize(Ls);
        for (size_t ell = 0; ell < Ls; ++ell) {
            const int N = L[ell].nx * L[ell].ny * L[ell].nz;
            uL[ell].assign(N, 0.0);
            fL[ell].assign(N, 0.0);
            rL[ell].assign(N, 0.0);
            tL[ell].assign(N, 0.0);
        }
    }

    /** \brief Apply anisotropic 3D FD Laplacian \f$A=-\Delta_h\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u input vector of nodal values
     *  \param out output vector storing \f$Au\f$
     *  \note Dirichlet boundaries are fixed; updates are applied only to interior nodes.
     */
    static void apply_A(const Level3D& lev, const std::vector<double>& u,
                        std::vector<double>& out) {
        const int nx = lev.nx, ny = lev.ny, nz = lev.nz;
        const double ihx2 = 1.0 / (lev.hx * lev.hx);
        const double ihy2 = 1.0 / (lev.hy * lev.hy);
        const double ihz2 = 1.0 / (lev.hz * lev.hz);
        const double diag = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;
        std::fill(out.begin(), out.end(), 0.0);
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i) {
                    const int id = id3(i, j, k, nx, ny);
                    out[id] = diag * u[id]
                              - ihx2 * (u[id3(i - 1, j, k, nx, ny)] + u[id3(i + 1, j, k, nx, ny)])
                              - ihy2 * (u[id3(i, j - 1, k, nx, ny)] + u[id3(i, j + 1, k, nx, ny)])
                              - ihz2 * (u[id3(i, j, k - 1, nx, ny)] + u[id3(i, j, k + 1, nx, ny)]);
                }
    }

    /** \brief Compute residual \f$r = f - Au\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u current iterate
     *  \param f right-hand side
     *  \param r output residual
     *  \param Au_scratch temporary storage for \f$Au\f$
     */
    static void residual(const Level3D& lev, const std::vector<double>& u,
                         const std::vector<double>& f, std::vector<double>& r,
                         std::vector<double>& Au_scratch) {
        apply_A(lev, u, Au_scratch);
        const int N = lev.nx * lev.ny * lev.nz;
        for (int id = 0; id < N; ++id)
            r[id] = f[id] - Au_scratch[id];
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
    static void smooth_jacobi(const Level3D& lev, std::vector<double>& u,
                              const std::vector<double>& f, std::vector<double>& r_scratch,
                              std::vector<double>& Au_scratch, int iters, double omega) {
        const int nx = lev.nx, ny = lev.ny, nz = lev.nz;
        const double ihx2 = 1.0 / (lev.hx * lev.hx);
        const double ihy2 = 1.0 / (lev.hy * lev.hy);
        const double ihz2 = 1.0 / (lev.hz * lev.hz);
        const double D    = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;  // diag(A)
        for (int it = 0; it < iters; ++it) {
            residual(lev, u, f, r_scratch, Au_scratch);
            for (int k = 1; k < nz - 1; ++k)
                for (int j = 1; j < ny - 1; ++j)
                    for (int i = 1; i < nx - 1; ++i) {
                        const int id = id3(i, j, k, nx, ny);
                        u[id] += omega * (r_scratch[id] / D);
                    }
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
    static void restrict_fullweight(const Level3D& fine, const std::vector<double>& r_f,
                                    const Level3D& coarse, std::vector<double>& f_c) {
        const int nxf = fine.nx, nyf = fine.ny, nzf = fine.nz;
        const int nxc = coarse.nx, nyc = coarse.ny, nzc = coarse.nz;
        std::fill(f_c.begin(), f_c.end(), 0.0);
        for (int K = 1; K < nzc - 1; ++K)
            for (int J = 1; J < nyc - 1; ++J)
                for (int I = 1; I < nxc - 1; ++I) {
                    const int i = 2 * I, j = 2 * J, k = 2 * K;
                    double sum = 0.0;
                    for (int dk = -1; dk <= 1; ++dk)
                        for (int dj = -1; dj <= 1; ++dj)
                            for (int di = -1; di <= 1; ++di) {
                                int wclass = std::abs(di) + std::abs(dj) + std::abs(dk);
                                double w   = (wclass == 0)
                                                 ? 8.0
                                                 : (wclass == 1 ? 4.0 : (wclass == 2 ? 2.0 : 1.0));
                                sum += w * r_f[id3(i + di, j + dj, k + dk, nxf, nyf)];
                            }
                    f_c[id3(I, J, K, nxc, nyc)] = sum / 64.0;
                }
    }

    /** \brief Trilinear prolongation: fine correction += \f$P\,u_c\f$.
     *  \param coarse coarse level geometry
     *  \param u_c coarse correction
     *  \param fine fine level geometry
     *  \param u_f in/out fine correction (accumulated)
     */
    static void prolong_add(const Level3D& coarse, const std::vector<double>& u_c,
                            const Level3D& fine, std::vector<double>& u_f) {
        const int nxc = coarse.nx, nyc = coarse.ny, nzc = coarse.nz;
        const int nxf = fine.nx, nyf = fine.ny, nzf = fine.nz;
        for (int K = 1; K < nzc - 1; ++K)
            for (int J = 1; J < nyc - 1; ++J)
                for (int I = 1; I < nxc - 1; ++I) {
                    const double c000 = u_c[id3(I, J, K, nxc, nyc)];
                    const double c100 = u_c[id3(I + 1, J, K, nxc, nyc)];
                    const double c010 = u_c[id3(I, J + 1, K, nxc, nyc)];
                    const double c110 = u_c[id3(I + 1, J + 1, K, nxc, nyc)];
                    const double c001 = u_c[id3(I, J, K + 1, nxc, nyc)];
                    const double c101 = u_c[id3(I + 1, J, K + 1, nxc, nyc)];
                    const double c011 = u_c[id3(I, J + 1, K + 1, nxc, nyc)];
                    const double c111 = u_c[id3(I + 1, J + 1, K + 1, nxc, nyc)];

                    const int i = 2 * I, j = 2 * J, k = 2 * K;

                    // layer k
                    u_f[id3(i, j, k, nxf, nyf)] += c000;
                    u_f[id3(i + 1, j, k, nxf, nyf)] += 0.5 * (c000 + c100);
                    u_f[id3(i, j + 1, k, nxf, nyf)] += 0.5 * (c000 + c010);
                    u_f[id3(i + 1, j + 1, k, nxf, nyf)] += 0.25 * (c000 + c100 + c010 + c110);

                    // layer k+1
                    u_f[id3(i, j, k + 1, nxf, nyf)] += 0.5 * (c000 + c001);
                    u_f[id3(i + 1, j, k + 1, nxf, nyf)] += 0.25 * (c000 + c100 + c001 + c101);
                    u_f[id3(i, j + 1, k + 1, nxf, nyf)] += 0.25 * (c000 + c010 + c001 + c011);
                    u_f[id3(i + 1, j + 1, k + 1, nxf, nyf)] +=
                        0.125 * (c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111);
                }
    }

    /** \brief One matrix-free V-cycle on level \f$\ell\f$ using workspace vectors.
     *  \details Steps: \f$\nu_1\f$ pre-smooth → residual → restrict → coarse solve → prolongate →
     * \f$\nu_2\f$ post-smooth.
     */
    void vcycle(int ell) {
        Level3D& lev = L[ell];
        auto &u = uL[ell], &f = fL[ell], &r = rL[ell], &t = tL[ell];
        if (ell == (int)L.size() - 1) {
            smooth_jacobi(lev, u, f, r, t, 50, omega);  // coarsest pseudo-solve
            return;
        }
        smooth_jacobi(lev, u, f, r, t, nu1, omega);
        residual(lev, u, f, r, t);
        restrict_fullweight(L[ell], r, L[ell + 1], fL[ell + 1]);
        std::fill(uL[ell + 1].begin(), uL[ell + 1].end(), 0.0);
        vcycle(ell + 1);
        prolong_add(L[ell + 1], uL[ell + 1], L[ell], uL[ell]);
        smooth_jacobi(lev, u, f, r, t, nu2, omega);
    }

    /** \brief Apply preconditioner: \f$z = M^{-1} r\f$ using one V-cycle (matrix-free).
     *  \param r_finest input residual on finest grid
     *  \param z_finest output preconditioned vector
     */
    void apply_precond(const std::vector<double>& r_finest, std::vector<double>& z_finest) {
        fL[0] = r_finest;
        for (size_t ell = 0; ell < L.size(); ++ell)
            std::fill(uL[ell].begin(), uL[ell].end(), 0.0);
        vcycle(0);
        z_finest = uL[0];
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

    /** \brief Euclidean dot product \f$(a,b)\f$. */
    static inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i)
            s += a[i] * b[i];
        return s;
    }
    /** \brief y ← y + a x. */
    static inline void axpy(std::vector<double>& y, double a, const std::vector<double>& x) {
        for (size_t i = 0; i < y.size(); ++i)
            y[i] += a * x[i];
    }
    /** \brief 2-norm \f$\|x\|_2\f$. */
    static inline double nrm2(const std::vector<double>& x) {
        double s = 0;
        for (double v : x)
            s += v * v;
        return std::sqrt(s);
    }

    /** \brief Solve \f$Au=f\f$ using PCG with MG preconditioning.
     *  \param levA operator geometry (spacings, sizes)
     *  \param u in/out solution
     *  \param f right-hand side
     *  \param mg multigrid preconditioner
     *  \param final_relres returns final \f$\|r\|/\|f\|\f$
     *  \return iterations performed
     */
    int solve(const Level3D& levA, std::vector<double>& u, const std::vector<double>& f,
              MultiGrid3D& mg, double& final_relres) {
        const int N = (int)u.size();
        std::vector<double> r(N, 0.0), z(N, 0.0), p(N, 0.0), Ap(N, 0.0);

        MultiGrid3D::apply_A(levA, u, Ap);
        for (int i = 0; i < N; ++i)
            r[i] = f[i] - Ap[i];

        double normf = nrm2(f);
        if (normf == 0.0)
            normf = 1.0;
        double rel = nrm2(r) / normf;
        if (rel < tol) {
            final_relres = rel;
            return 0;
        }

        mg.apply_precond(r, z);
        p             = z;
        double rz_old = dot(r, z);

        int it = 0;
        for (; it < maxit; ++it) {
            MultiGrid3D::apply_A(levA, p, Ap);
            double alpha = rz_old / dot(p, Ap);
            axpy(u, alpha, p);
            axpy(r, -alpha, Ap);
            rel = nrm2(r) / normf;
            if (rel < tol) {
                ++it;
                break;
            }
            mg.apply_precond(r, z);
            double rz_new = dot(r, z);
            double beta   = rz_new / rz_old;
            for (int i = 0; i < N; ++i)
                p[i] = z[i] + beta * p[i];
            rz_old = rz_new;
        }
        final_relres = rel;
        return it;
    }
};

/** \brief Manufactured 3D test: \f$u=\sin(\pi x)\sin(\pi y)\sin(\pi z)\f$, \f$f=3\pi^2 u\f$
 * (Dirichlet consistent). */
void fill_manufactured(const Level3D& lev, std::vector<double>& f, std::vector<double>& u_exact) {
    const double pi = 3.14159265358979323846;
    const int nx = lev.nx, ny = lev.ny, nz = lev.nz;
    f.assign(nx * ny * nz, 0.0);
    u_exact.assign(nx * ny * nz, 0.0);
    for (int k = 0; k < nz; ++k) {
        double z = k * lev.hz;
        for (int j = 0; j < ny; ++j) {
            double y = j * lev.hy;
            for (int i = 0; i < nx; ++i) {
                double x     = i * lev.hx;
                const int id = id3(i, j, k, nx, ny);
                double uex   = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
                u_exact[id]  = uex;
                f[id]        = 3.0 * pi * pi * uex;  // -Δ u = (π²+π²+π²) u
            }
        }
    }
    // enforce zero RHS on Dirichlet boundary nodes
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            f[id3(0, j, k, nx, ny)]      = 0;
            f[id3(nx - 1, j, k, nx, ny)] = 0;
        }
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            f[id3(i, 0, k, nx, ny)]      = 0;
            f[id3(i, ny - 1, k, nx, ny)] = 0;
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            f[id3(i, j, 0, nx, ny)]      = 0;
            f[id3(i, j, nz - 1, nx, ny)] = 0;
        }
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

    ippl::Info->setOutputLevel(1);
    ippl::Warn->setOutputLevel(1);
    ippl::Error->setOutputLevel(1);

    Inform msg("mg");
    msg << "hello from IPPL, rank=" << ippl::Comm->rank() << ", size=" << ippl::Comm->size()
        << endl;

    Args args = parse_args(argc, argv);
    if (args.nx < 3 || args.ny < 3 || args.nz < 3) {
        std::cerr << "All dims must be >=3\n";
        ippl::finalize();
        return 1;
    }

    // Build geometry + allocate matrix-free MG workspace
    MultiGrid3D mg(args.nx, args.ny, args.nz, args.nu1, args.nu2, args.w);

    const Level3D& Alev = mg.L[0];
    std::vector<double> u(Alev.nx * Alev.ny * Alev.nz, 0.0);
    std::vector<double> f, uex;
    fill_manufactured(Alev, f, uex);

    std::cout << "3D Matrix-Free MG-preconditioned CG for -Δu=f on [0,1]^3\n";
    std::cout << "Grid: nx=" << Alev.nx << ", ny=" << Alev.ny << ", nz=" << Alev.nz
              << ", hx=" << Alev.hx << ", hy=" << Alev.hy << ", hz=" << Alev.hz << "\n";
    std::cout << "Smoother: weighted Jacobi (ω=" << args.w << ", nu1=" << args.nu1
              << ", nu2=" << args.nu2 << ")\n";

    static IpplTimings::TimerRef pcg_timer = IpplTimings::getTimer("pcg_timer");

    IpplTimings::startTimer(pcg_timer);
    PCG pcg;
    pcg.maxit  = args.maxit;
    pcg.tol    = args.tol;
    double rel = 1.0;
    auto t0    = std::chrono::high_resolution_clock::now();
    int it     = pcg.solve(Alev, u, f, mg, rel);
    auto t1    = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    IpplTimings::stopTimer(pcg_timer);

    // Relative L2 error vs exact (interior only)
    double e2 = 0.0, u2 = 0.0;
    const int nx = Alev.nx, ny = Alev.ny, nz = Alev.nz;
    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i) {
                int id   = id3(i, j, k, nx, ny);
                double e = u[id] - uex[id];
                e2 += e * e;
                u2 += uex[id] * uex[id];
            }
    double relL2 = std::sqrt(e2) / std::max(1e-300, std::sqrt(u2));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "PCG iters: " << it << ", final rel. residual: " << rel
              << ", rel. L2 error: " << relL2 << "\n";
    std::cout << "Wall time: " << sec << " s\n";

    IpplTimings::print();
    // Finalize ippl
    ippl::finalize();
    return 0;
}
