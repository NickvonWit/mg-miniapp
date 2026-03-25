// Mini-app: Matrix-Free Multigrid-Preconditioned CG for 3D Poisson (independent nx,ny,nz)
// -----------------------------------------------------------------------------
// Kokkos C++17, single-file, educational. No deps beyond Kokkos.
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
//   g++ -O3 -std=c++17 -I<kokkos_include> -L<kokkos_lib> mg3d_kokkos.cpp -lkokkoscore -o mg3d_kokkos
// Run:
//   ./mg3d_kokkos -nx 65 -ny 97 -nz 129 -nu1 2 -nu2 2 -w 0.8 -tol 1e-8 -maxit 200
// -----------------------------------------------------------------------------

/** \file mg3d_kokkos.cpp
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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

using DeviceView = Kokkos::View<double*>;

/** \brief Row-major 3D index mapping.
 *  \param i x-index (0..nx-1), \param j y-index (0..ny-1), \param k z-index (0..nz-1)
 *  \param nx leading dimension (x), \param ny size in y
 *  \return linear index \f$\mathrm{id} = i + n_x\,(j + n_y\,k)\f$.
 */
KOKKOS_INLINE_FUNCTION
int id3(int i, int j, int k, int nx, int ny) { return i + nx * (j + ny * k); }

/** \brief Geometry-only description of one multigrid level.
 *  \details Level \f$\ell\f$ stores only sizes and spacings
 *  \f$(n_x^{(\ell)},n_y^{(\ell)},n_z^{(\ell)})\f$ and
 *  \f$(h_x^{(\ell)},h_y^{(\ell)},h_z^{(\ell)})\f$.
 */
struct Level3D {
    int    nx = 0, ny = 0, nz = 0;  //!< grid points per dimension (including boundaries)
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
    std::vector<Level3D> L;                 //!< L[0] finest
    int                  nu1 = 2, nu2 = 2;  //!< pre/post smoothing steps
    double               omega = 0.8;       //!< weighted-Jacobi parameter

    // Workspace per level (matrix-free; no A stored)
    std::vector<DeviceView> uL;  //!< level-wise correction buffers
    std::vector<DeviceView> fL;  //!< level-wise RHS buffers
    std::vector<DeviceView> rL;  //!< level-wise residual scratch
    std::vector<DeviceView> tL;  //!< level-wise Au scratch

    /** \brief Construct hierarchy and allocate workspaces.
     *  \param nx_f finest grid points in x
     *  \param ny_f finest grid points in y
     *  \param nz_f finest grid points in z
     *  \param nu1_ pre-smoothing sweeps
     *  \param nu2_ post-smoothing sweeps
     *  \param omega_ Jacobi weight \f$\omega\f$
     */
    MultiGrid3D(int nx_f, int ny_f, int nz_f, int nu1_ = 2, int nu2_ = 2, double omega_ = 0.8)
        : nu1(nu1_), nu2(nu2_), omega(omega_) {
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
            if (nx <= 3 || ny <= 3 || nz <= 3) break;
            auto half = [](int n) { return (n - 1) / 2 + 1; };
            nx        = std::max(3, half(nx));
            ny        = std::max(3, half(ny));
            nz        = std::max(3, half(nz));
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
            uL[ell]     = DeviceView("uL_" + std::to_string(ell), N);
            fL[ell]     = DeviceView("fL_" + std::to_string(ell), N);
            rL[ell]     = DeviceView("rL_" + std::to_string(ell), N);
            tL[ell]     = DeviceView("tL_" + std::to_string(ell), N);
        }
    }

    /** \brief Apply anisotropic 3D FD Laplacian \f$A=-\Delta_h\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u input vector of nodal values
     *  \param out output vector storing \f$Au\f$
     *  \note Dirichlet boundaries are fixed; updates are applied only to interior nodes.
     */
    static void apply_A(const Level3D& lev, const DeviceView& u, const DeviceView& out) {
        const int    nx = lev.nx, ny = lev.ny, nz = lev.nz;
        const double ihx2 = 1.0 / (lev.hx * lev.hx);
        const double ihy2 = 1.0 / (lev.hy * lev.hy);
        const double ihz2 = 1.0 / (lev.hz * lev.hz);
        const double diag = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;
        Kokkos::deep_copy(out, 0.0);
        Kokkos::parallel_for(
            "apply_A", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {nx - 1, ny - 1, nz - 1}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                const int id = id3(i, j, k, nx, ny);
                out(id)      = diag * u(id) -
                          ihx2 * (u(id3(i - 1, j, k, nx, ny)) + u(id3(i + 1, j, k, nx, ny))) -
                          ihy2 * (u(id3(i, j - 1, k, nx, ny)) + u(id3(i, j + 1, k, nx, ny))) -
                          ihz2 * (u(id3(i, j, k - 1, nx, ny)) + u(id3(i, j, k + 1, nx, ny)));
            });
    }

    /** \brief Compute residual \f$r = f - Au\f$ (matrix-free).
     *  \param lev level geometry
     *  \param u current iterate
     *  \param f right-hand side
     *  \param r output residual
     *  \param Au_scratch temporary storage for \f$Au\f$
     */
    static void residual(const Level3D& lev, const DeviceView& u, const DeviceView& f,
                         DeviceView& r, DeviceView& Au_scratch) {
        apply_A(lev, u, Au_scratch);
        const int N = lev.nx * lev.ny * lev.nz;
        Kokkos::parallel_for(
            "residual", N, KOKKOS_LAMBDA(int id) { r(id) = f(id) - Au_scratch(id); });
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
    static void smooth_jacobi(const Level3D& lev, DeviceView& u, const DeviceView& f,
                              DeviceView& r_scratch, DeviceView& Au_scratch, int iters,
                              double omega) {
        const int    nx = lev.nx, ny = lev.ny, nz = lev.nz;
        const double ihx2 = 1.0 / (lev.hx * lev.hx);
        const double ihy2 = 1.0 / (lev.hy * lev.hy);
        const double ihz2 = 1.0 / (lev.hz * lev.hz);
        const double D    = 2.0 * ihx2 + 2.0 * ihy2 + 2.0 * ihz2;  // diag(A)
        for (int it = 0; it < iters; ++it) {
            residual(lev, u, f, r_scratch, Au_scratch);
            Kokkos::parallel_for(
                "jacobi_update",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {nx - 1, ny - 1, nz - 1}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    const int id = id3(i, j, k, nx, ny);
                    u(id) += omega * (r_scratch(id) / D);
                });
        }
    }

    /** \brief 3D full-weighting restriction: coarse RHS = \f$R\,r_f\f$.
     *  \param fine fine level geometry
     *  \param r_f fine residual
     *  \param coarse coarse level geometry
     *  \param f_c output coarse RHS
     *  \details Weights relative to center (2I,2J,2K): center 8; faces 4; edges 2; corners 1; then divide by 64.
     */
    static void restrict_fullweight(const Level3D& fine, const DeviceView& r_f,
                                    const Level3D& coarse, DeviceView& f_c) {
        const int nxf = fine.nx, nyf = fine.ny;
        const int nxc = coarse.nx, nyc = coarse.ny, nzc = coarse.nz;
        Kokkos::deep_copy(f_c, 0.0);
        Kokkos::parallel_for(
            "restrict",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {nxc - 1, nyc - 1, nzc - 1}),
            KOKKOS_LAMBDA(int I, int J, int K) {
                const int i = 2 * I, j = 2 * J, k = 2 * K;
                double    sum = 0.0;
                for (int dk = -1; dk <= 1; ++dk)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            int wclass =
                                (di < 0 ? -di : di) + (dj < 0 ? -dj : dj) + (dk < 0 ? -dk : dk);
                            double w = (wclass == 0)
                                           ? 8.0
                                           : (wclass == 1 ? 4.0 : (wclass == 2 ? 2.0 : 1.0));
                            sum += w * r_f(id3(i + di, j + dj, k + dk, nxf, nyf));
                        }
                f_c(id3(I, J, K, nxc, nyc)) = sum / 64.0;
            });
    }

    /** \brief Trilinear prolongation: fine correction += \f$P\,u_c\f$.
     *  \param coarse coarse level geometry
     *  \param u_c coarse correction
     *  \param fine fine level geometry
     *  \param u_f in/out fine correction (accumulated)
     */
    static void prolong_add(const Level3D& coarse, const DeviceView& u_c, const Level3D& fine,
                            DeviceView& u_f) {
        const int nxc = coarse.nx, nyc = coarse.ny, nzc = coarse.nz;
        const int nxf = fine.nx, nyf = fine.ny;
        Kokkos::parallel_for(
            "prolong",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {nxc - 1, nyc - 1, nzc - 1}),
            KOKKOS_LAMBDA(int I, int J, int K) {
                const double c000 = u_c(id3(I, J, K, nxc, nyc));
                const double c100 = u_c(id3(I + 1, J, K, nxc, nyc));
                const double c010 = u_c(id3(I, J + 1, K, nxc, nyc));
                const double c110 = u_c(id3(I + 1, J + 1, K, nxc, nyc));
                const double c001 = u_c(id3(I, J, K + 1, nxc, nyc));
                const double c101 = u_c(id3(I + 1, J, K + 1, nxc, nyc));
                const double c011 = u_c(id3(I, J + 1, K + 1, nxc, nyc));
                const double c111 = u_c(id3(I + 1, J + 1, K + 1, nxc, nyc));

                const int i = 2 * I, j = 2 * J, k = 2 * K;

                // layer k
                Kokkos::atomic_add(&u_f(id3(i, j, k, nxf, nyf)), c000);
                Kokkos::atomic_add(&u_f(id3(i + 1, j, k, nxf, nyf)), 0.5 * (c000 + c100));
                Kokkos::atomic_add(&u_f(id3(i, j + 1, k, nxf, nyf)), 0.5 * (c000 + c010));
                Kokkos::atomic_add(&u_f(id3(i + 1, j + 1, k, nxf, nyf)),
                                   0.25 * (c000 + c100 + c010 + c110));

                // layer k+1
                Kokkos::atomic_add(&u_f(id3(i, j, k + 1, nxf, nyf)), 0.5 * (c000 + c001));
                Kokkos::atomic_add(&u_f(id3(i + 1, j, k + 1, nxf, nyf)),
                                   0.25 * (c000 + c100 + c001 + c101));
                Kokkos::atomic_add(&u_f(id3(i, j + 1, k + 1, nxf, nyf)),
                                   0.25 * (c000 + c010 + c001 + c011));
                Kokkos::atomic_add(&u_f(id3(i + 1, j + 1, k + 1, nxf, nyf)),
                                   0.125 * (c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111));
            });
    }

    /** \brief One matrix-free V-cycle on level \f$\ell\f$ using workspace vectors.
     *  \details Steps: \f$\nu_1\f$ pre-smooth → residual → restrict → coarse solve → prolongate → \f$\nu_2\f$ post-smooth.
     */
    void vcycle(int ell) {
        Level3D& lev = L[ell];
        auto &   u = uL[ell], &f = fL[ell], &r = rL[ell], &t = tL[ell];
        if (ell == (int) L.size() - 1) {
            smooth_jacobi(lev, u, f, r, t, 50, omega);  // coarsest pseudo-solve
            return;
        }
        smooth_jacobi(lev, u, f, r, t, nu1, omega);
        residual(lev, u, f, r, t);
        restrict_fullweight(L[ell], r, L[ell + 1], fL[ell + 1]);
        Kokkos::deep_copy(uL[ell + 1], 0.0);
        vcycle(ell + 1);
        prolong_add(L[ell + 1], uL[ell + 1], L[ell], uL[ell]);
        smooth_jacobi(lev, u, f, r, t, nu2, omega);
    }

    /** \brief Apply preconditioner: \f$z = M^{-1} r\f$ using one V-cycle (matrix-free).
     *  \param r_finest input residual on finest grid
     *  \param z_finest output preconditioned vector
     */
    void apply_precond(const DeviceView& r_finest, DeviceView& z_finest) {
        Kokkos::deep_copy(fL[0], r_finest);
        for (size_t ell = 0; ell < L.size(); ++ell) Kokkos::deep_copy(uL[ell], 0.0);
        vcycle(0);
        Kokkos::deep_copy(z_finest, uL[0]);
    }
};

/** \brief Preconditioned Conjugate Gradient (PCG) for SPD systems using MG V-cycles as right preconditioner.
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
    int    maxit = 200;
    double tol   = 1e-8;

    /** \brief Euclidean dot product \f$(a,b)\f$. */
    static inline double dot(const DeviceView& a, const DeviceView& b) {
        double    s = 0;
        const int N = a.extent(0);
        Kokkos::parallel_reduce(
            "dot", N, KOKKOS_LAMBDA(int i, double& lsum) { lsum += a(i) * b(i); }, s);
        return s;
    }
    /** \brief y ← y + a x. */
    static inline void axpy(const DeviceView& y, double a, const DeviceView& x) {
        const int N = y.extent(0);
        Kokkos::parallel_for("axpy", N, KOKKOS_LAMBDA(int i) { y(i) += a * x(i); });
    }
    /** \brief 2-norm \f$\|x\|_2\f$. */
    static inline double nrm2(const DeviceView& x) { return std::sqrt(dot(x, x)); }

    /** \brief Solve \f$Au=f\f$ using PCG with MG preconditioning.
     *  \param levA operator geometry (spacings, sizes)
     *  \param u in/out solution
     *  \param f right-hand side
     *  \param mg multigrid preconditioner
     *  \param final_relres returns final \f$\|r\|/\|f\|\f$
     *  \return iterations performed
     */
    int solve(const Level3D& levA, DeviceView& u, const DeviceView& f, MultiGrid3D& mg,
              double& final_relres) {
        const int  N = (int) u.extent(0);
        DeviceView r("r", N), z("z", N), p("p", N), Ap("Ap", N);

        MultiGrid3D::apply_A(levA, u, Ap);
        Kokkos::parallel_for("init_r", N, KOKKOS_LAMBDA(int i) { r(i) = f(i) - Ap(i); });

        double normf = nrm2(f);
        if (normf == 0.0) normf = 1.0;
        double rel = nrm2(r) / normf;
        if (rel < tol) {
            final_relres = rel;
            return 0;
        }

        mg.apply_precond(r, z);
        Kokkos::deep_copy(p, z);
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
            Kokkos::parallel_for(
                "update_p", N, KOKKOS_LAMBDA(int i) { p(i) = z(i) + beta * p(i); });
            rz_old = rz_new;
        }
        final_relres = rel;
        return it;
    }
};

/** \brief Manufactured 3D test: \f$u=\sin(\pi x)\sin(\pi y)\sin(\pi z)\f$, \f$f=3\pi^2 u\f$ (Dirichlet consistent). */
void fill_manufactured(const Level3D& lev, DeviceView& f, DeviceView& u_exact) {
    const double pi = 3.14159265358979323846;
    const int    nx = lev.nx, ny = lev.ny, nz = lev.nz;
    const double hx = lev.hx, hy = lev.hy, hz = lev.hz;

    Kokkos::parallel_for(
        "fill_manufactured", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int i, int j, int k) {
            double    x = i * hx, y = j * hy, z = k * hz;
            const int id  = id3(i, j, k, nx, ny);
            double    uex = Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * Kokkos::sin(pi * z);
            u_exact(id)   = uex;
            // enforce zero RHS on Dirichlet boundary nodes
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                f(id) = 0.0;
            else
                f(id) = 3.0 * pi * pi * uex;  // -Δ u = (π²+π²+π²) u
        });
}

/** \brief Command-line arguments. */
struct Args {
    int    nx = 65, ny = 65, nz = 65;
    int    nu1 = 2, nu2 = 2;
    double w     = 0.8;
    double tol   = 1e-8;
    int    maxit = 200;
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
            std::cout << "\nUsage: ./mg3d_kokkos -nx <int> -ny <int> -nz <int> -nu1 <int> -nu2 "
                         "<int> -w <omega> -tol <tol> -maxit <int>\n";
            std::exit(0);
        }
    }
    return a;
}

/** \brief Driver: assemble manufactured problem and solve with matrix-free MG-preconditioned CG. */
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        Args args = parse_args(argc, argv);
        if (args.nx < 3 || args.ny < 3 || args.nz < 3) {
            std::cerr << "All dims must be >=3\n";
            Kokkos::finalize();
            return 1;
        }

        // Build geometry + allocate matrix-free MG workspace
        MultiGrid3D mg(args.nx, args.ny, args.nz, args.nu1, args.nu2, args.w);

        const Level3D& Alev = mg.L[0];
        const int      N    = Alev.nx * Alev.ny * Alev.nz;
        DeviceView     u("u", N);
        DeviceView     f("f", N), uex("uex", N);
        fill_manufactured(Alev, f, uex);

        std::cout << "3D Matrix-Free MG-preconditioned CG for -Δu=f on [0,1]^3\n";
        std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
        std::cout << "Grid: nx=" << Alev.nx << ", ny=" << Alev.ny << ", nz=" << Alev.nz
                  << ", hx=" << Alev.hx << ", hy=" << Alev.hy << ", hz=" << Alev.hz << "\n";
        std::cout << "Smoother: weighted Jacobi (ω=" << args.w << ", nu1=" << args.nu1
                  << ", nu2=" << args.nu2 << ")\n";

        PCG pcg;
        pcg.maxit         = args.maxit;
        pcg.tol           = args.tol;
        double        rel = 1.0;
        Kokkos::Timer timer;
        int           it  = pcg.solve(Alev, u, f, mg, rel);
        double        sec = timer.seconds();

        // Relative L2 error vs exact (interior only)
        double    e2 = 0.0, u2 = 0.0;
        const int nx = Alev.nx, ny = Alev.ny, nz = Alev.nz;
        Kokkos::parallel_reduce(
            "L2_error", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {nx - 1, ny - 1, nz - 1}),
            KOKKOS_LAMBDA(int i, int j, int k, double& le2, double& lu2) {
                int    id = id3(i, j, k, nx, ny);
                double e  = u(id) - uex(id);
                le2 += e * e;
                lu2 += uex(id) * uex(id);
            },
            e2, u2);
        double relL2 = std::sqrt(e2) / std::max(1e-300, std::sqrt(u2));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "PCG iters: " << it << ", final rel. residual: " << rel
                  << ", rel. L2 error: " << relL2 << "\n";
        std::cout << "Wall time: " << sec << " s\n";
    }
    Kokkos::finalize();
    return 0;
}
