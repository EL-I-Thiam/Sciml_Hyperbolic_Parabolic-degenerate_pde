import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from src.solvers import solve_imex_ssp2_step

os.makedirs('figures', exist_ok=True)
os.makedirs('/tmp/figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.0,
    'font.family': 'serif',
    'mathtext.fontset': 'cm'
})

def solve_viscous_burgers_solver(nu, N=120, Nt=100, Tf=1.0):
    """Solves u_t + u u_x = nu u_xx on [-1, 1] x [0, Tf] via unified solve_imex_ssp2_step."""
    L = 1.0; dx = 2.0 * L / N
    x = -L + (np.arange(N) + 0.5) * dx
    dt = Tf / Nt
    u = -np.sin(np.pi * x)
    
    trajectory = np.zeros((Nt + 1, N))
    trajectory[0, :] = u.copy()
    
    f_burgers = lambda s: 0.5 * s**2
    df_burgers = lambda s: s
    D_burgers = lambda s: nu * np.ones_like(s)
    
    for n in range(1, Nt + 1):
        u = solve_imex_ssp2_step(u, dt, dx, f_func=f_burgers, df_func=df_burgers, D_func=D_burgers, limiter='superbee')
        trajectory[n, :] = u.copy()
        
    return x, np.linspace(0, Tf, Nt + 1), trajectory

def run_benchmark():
    print("=" * 85)
    print("=== BENCHMARK 4: VISCOUS BURGERS SCIML INVERSE ASSIMILATION (Raissi et al. 2019) ===")
    print("=" * 85)
    
    nu_true = 0.01 / np.pi # ~ 0.0031831
    nu_init = 0.010000
    
    # 1. Generate ground truth trajectory using solver
    x, t_grid, traj_true = solve_viscous_burgers_solver(nu_true, N=120, Nt=100)
    
    # 2. Sample 200 sparse space-time observation sensors with 2% noise
    np.random.seed(99)
    N_obs = 200
    idx_t = np.random.randint(5, len(t_grid), size=N_obs)
    idx_x = np.random.randint(5, len(x), size=N_obs)
    obs_u = traj_true[idx_t, idx_x] + 0.02 * np.random.randn(N_obs) * np.max(np.abs(traj_true))

    # 3. Live optimization of viscosity nu via L-BFGS-B calling the forward solver
    t_start = time.time()
    def burgers_loss(nu_param):
        nu_val = nu_param[0]
        _, _, traj_sim = solve_viscous_burgers_solver(nu_val, N=120, Nt=100)
        pred_u = traj_sim[idx_t, idx_x]
        return np.mean((pred_u - obs_u)**2)

    print(f"Ground Truth Viscosity (nu_true) : {nu_true:.6f}")
    print(f"Initial Guess Viscosity (nu_init): {nu_init:.6f} (+214.2% error)")
    print("Executing live parameter calibration via L-BFGS-B through the Burgers forward solver...")
    
    res = minimize(burgers_loss, [nu_init], method='L-BFGS-B', bounds=[(0.0005, 0.05)], options={'eps': 1e-4})
    cpu_time = time.time() - t_start
    nu_calibrated = float(res.x[0])
    rel_error = abs(nu_calibrated - nu_true) / nu_true * 100.0

    # 4. Evaluate full-field trajectory with calibrated parameter
    _, _, traj_reconstructed = solve_viscous_burgers_solver(nu_calibrated, N=120, Nt=100)
    field_l2_error = np.linalg.norm(traj_reconstructed - traj_true) / np.linalg.norm(traj_true) * 100.0

    print(f"L-BFGS-B Solver Convergence   : {res.nfev} forward solver evaluations completed.")
    print(f"Calibrated Viscosity (nu*)       : {nu_calibrated:.6f} (Error: {rel_error:.2f}%)")
    print(f"Global Space-Time L2 Field Error : {field_l2_error:.2f}%")
    print(f"Total SciML Optimization CPU Time: {cpu_time:.2f} seconds")

    # Figure 11 plotted 100% directly from simulated fields
    fig11, axes11 = plt.subplots(1, 2, figsize=(14.0, 5.2))
    
    # (a) Left: Space-time contour field with 200 sensors
    ax = axes11[0]
    T_mesh, X_mesh = np.meshgrid(t_grid, x)
    c1 = ax.contourf(T_mesh, X_mesh, traj_reconstructed.T, levels=50, cmap='rainbow')
    cb = fig11.colorbar(c1, ax=ax)
    cb.set_label(r'Velocity $u(x, t)$')
    t_sensors = t_grid[idx_t]
    x_sensors = x[idx_x]
    ax.scatter(t_sensors, x_sensors, color='black', marker='x', s=20, label=r'Sensors (200 pts)')
    ax.set_xlabel(r'Time $t$'); ax.set_ylabel(r'Spatial coordinate $x$')
    ax.set_xlim(0.0, 1.0); ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='lower left', framealpha=0.85, fontsize=8.5)
    ax.text(0.04, 0.92, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (b) Right: Velocity profile cross-sections from solver
    ax = axes11[1]
    t_cuts = [0.25, 0.50, 0.75, 1.00]
    colors_cuts = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    ax.plot(x, traj_true[0, :], 'k:', lw=1.2, label='Exact $t=0$')
    for idx, tc in enumerate(t_cuts):
        idx_step = int(round(tc * 100))
        ax.plot(x, traj_reconstructed[idx_step, :], color=colors_cuts[idx], lw=1.8, label=f'Model $t={tc:.2f}$')
        ax.plot(x[::8], traj_true[idx_step, ::8], 'o', color=colors_cuts[idx], ms=3.5, alpha=0.7)
    ax.set_xlabel(r'Spatial coordinate $x$'); ax.set_ylabel(r'Velocity $u(x, t)$')
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    fig11.tight_layout()
    fig11.savefig('figures/benchmark_4_burgers_assimilation.png', dpi=300)
    fig11.savefig('/tmp/figures/benchmark_4_burgers_assimilation.png', dpi=300)
    plt.close(fig11)
    print(">> Saved: figures/benchmark_4_burgers_assimilation.png (100% computed from solver!)")

if __name__ == "__main__":
    run_benchmark()
