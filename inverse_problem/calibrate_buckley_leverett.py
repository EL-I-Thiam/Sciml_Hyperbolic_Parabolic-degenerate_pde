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

def run_calibration():
    print("=" * 85)
    print("=== INVERSE SCIML CALIBRATION: BUCKLEY-LEVERETT CAPILLARY DIFFUSIVITY ===")
    print("=" * 85)
    
    L = 1.0; Tf = 0.30; N = 80; dx = L / N; x = (np.arange(N) + 0.5) * dx
    dt = 0.002; Nt = int(round(Tf / dt))
    
    eps_true = 0.0100; p_true = 2.00
    eps_init = 0.0300; p_init = 1.20

    f_bl = lambda s: (np.clip(s, 0.0, 1.0)**2) / (np.clip(s, 0.0, 1.0)**2 + 0.5*((1.0 - np.clip(s, 0.0, 1.0))**2) + 1e-12)
    df_bl = lambda s: (np.clip(s, 0.0, 1.0)*(1.0 - np.clip(s, 0.0, 1.0))) / ((np.clip(s, 0.0, 1.0)**2 + 0.5*((1.0 - np.clip(s, 0.0, 1.0))**2))**2 + 1e-12)

    def simulate_bl(eps_cap, p_exp):
        D_func = lambda s: eps_cap * (np.clip(s, 0.0, 1.0)**p_exp) * ((1.0 - np.clip(s, 0.0, 1.0))**p_exp)
        u = np.zeros(N); u[0] = 1.0
        rec_times = [0.10, 0.20, 0.30]
        rec_steps = [int(round(t / dt)) for t in rec_times]
        history = {}
        
        for n in range(1, Nt + 1):
            u = solve_imex_ssp2_step(u, dt, dx, f_func=f_bl, df_func=df_bl, D_func=D_func, limiter='superbee', bc_left=1.0)
            if n in rec_steps:
                idx = rec_steps.index(n)
                history[rec_times[idx]] = u.copy()
        return u, history

    # Generate synthetic observations with 2% noise
    np.random.seed(42)
    _, true_history = simulate_bl(eps_true, p_true)
    sensor_idx = np.where((x > 0.05) & (x < 0.60))[0][::2]
    sensor_obs = {}
    for tk, u_ex in true_history.items():
        noise = 0.02 * np.random.randn(len(sensor_idx))
        sensor_obs[tk] = np.clip(u_ex[sensor_idx] + noise, 0.0, 1.0)

    print(f"Ground Truth Parameters       : eps_c = {eps_true:.4f}, p = {p_true:.2f}")
    print(f"Initial Guess Parameters      : eps_c = {eps_init:.4f}, p = {p_init:.2f} (eps: +200%, p: -40%)")
    print("Executing Nelder-Mead simplex optimization coupled to solve_imex_ssp2_step...")

    real_loss_history = []
    def bl_objective(params):
        eps_c, p_val = params
        _, sim_hist = simulate_bl(eps_c, p_val)
        loss = 0.0
        for tk, obs in sensor_obs.items():
            pred = sim_hist[tk][sensor_idx]
            loss += np.mean((pred - obs)**2)
        real_loss_history.append(loss)
        return loss

    loss_0 = bl_objective([eps_init, p_init])
    print(f"Initial Loss at (eps_c={eps_init}, p={p_init}): {loss_0:.6e}")

    t_start = time.time()
    res = minimize(
        bl_objective, [eps_init, p_init], method='Nelder-Mead',
        options={'maxiter': 15}
    )
    elapsed_time = time.time() - t_start

    eps_cal, p_cal = res.x[0], res.x[1]
    final_loss = res.fun
    err_eps = abs(eps_cal - eps_true) / eps_true * 100.0
    err_p = abs(p_cal - p_true) / p_true * 100.0

    print(f"\nOptimization Finished in {elapsed_time:.2f} seconds ({res.nit} iterations, {res.nfev} solver calls)")
    print(f"Calibrated Parameters (SciML) : eps_c = {eps_cal:.4f} (Error: {err_eps:.2f}%), p = {p_cal:.2f} (Error: {err_p:.2f}%)")
    print(f"Final Optimization Loss       : {final_loss:.6e}")

    # Generate Figure 10 from real optimization data
    u_bl = np.linspace(0.0, 1.0, 100)
    D_true_bl = eps_true * (u_bl**p_true) * ((1.0 - u_bl)**p_true)
    D_cal_bl = eps_cal * (u_bl**p_cal) * ((1.0 - u_bl)**p_cal)
    D_init_bl = eps_init * (u_bl**p_init) * ((1.0 - u_bl)**p_init)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    
    # (a) Capillary diffusivity curves
    ax = axes[0]
    ax.plot(u_bl, D_true_bl, 'k-', lw=2.2, label=rf'True $p={p_true:.2f}, \epsilon_c={eps_true:.3f}$')
    ax.plot(u_bl, D_cal_bl, 'b--', lw=1.8, label=rf'Calibrated $p={p_cal:.2f}, \epsilon_c={eps_cal:.4f}$')
    ax.plot(u_bl, D_init_bl, 'r:', lw=1.5, label=rf'Initial $p={p_init:.2f}, \epsilon_c={eps_init:.3f}$')
    ax.set_xlabel(r'Water Saturation $u$'); ax.set_ylabel(r'Capillary Diffusivity $D(u)$')
    ax.text(0.05, 0.90, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper right', framealpha=0.9, fontsize=8.0)

    # (b) REAL loss history recorded during optimization
    ax = axes[1]
    clean_loss = []
    min_so_far = np.inf
    for val in real_loss_history:
        if val < min_so_far:
            min_so_far = val
            clean_loss.append(val)
    iters_plot = np.arange(1, len(clean_loss) + 1)
    ax.semilogy(iters_plot, clean_loss, 'b-s', lw=1.8, ms=4)
    ax.set_xlabel(r'Optimization Iterations')
    ax.set_ylabel(r'Observation Loss $\mathcal{L}(\theta)$')
    ax.text(0.05, 0.90, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, which="both", linestyle=':', alpha=0.6)

    # (c) Real forward simulation with calibrated parameters
    ax = axes[2]
    colors_bl = ['#1f77b4', '#2ca02c', '#ff7f0e']
    _, cal_history = simulate_bl(eps_cal, p_cal)
    for idx, (tk, u_prof) in enumerate(cal_history.items()):
        ax.plot(x, u_prof, color=colors_bl[idx], lw=1.8, label=f'$t={tk:.2f}$')
        ax.plot(x[sensor_idx], sensor_obs[tk], 'x', color=colors_bl[idx], ms=5)
    ax.set_xlabel(r'Spatial coordinate $x$'); ax.set_ylabel(r'Saturation $u(x, t)$')
    ax.text(0.05, 0.90, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower left', framealpha=0.9, fontsize=8.0)

    fig.tight_layout()
    fig.savefig('figures/reconstruction_buckley_leverett.png', dpi=300)
    fig.savefig('/tmp/figures/reconstruction_buckley_leverett.png', dpi=300)
    plt.close(fig)
    print(">> Saved: figures/reconstruction_buckley_leverett.png")

if __name__ == "__main__":
    run_calibration()
