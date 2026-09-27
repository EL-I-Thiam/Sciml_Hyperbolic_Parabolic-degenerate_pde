import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.solvers import solve_imex_ssp2_step

os.makedirs('figures', exist_ok=True)
os.makedirs('/tmp/figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'font.family': 'serif',
    'mathtext.fontset': 'cm'
})

def run_benchmark():
    print("=" * 85)
    print("=== BENCHMARK 3: BUCKLEY-LEVERETT TWO-PHASE FLOW (2x4 MULTI-PANEL SUITE) ===")
    print("=" * 85)
    
    L = 1.0; Tf = 0.40; N = 200; dx = L / N; x = (np.arange(N) + 0.5) * dx
    Nt = int(round(Tf / 0.0010)); dt = Tf / Nt
    times_4 = [0.10, 0.20, 0.30, 0.40]
    eps_4 = [0.0, 0.005, 0.010, 0.050]
    all_eps = [0.0, 0.001, 0.005, 0.010, 0.025, 0.050]

    f_bl = lambda s: (np.clip(s, 0.0, 1.0)**2) / (np.clip(s, 0.0, 1.0)**2 + 0.5*((1.0 - np.clip(s, 0.0, 1.0))**2) + 1e-12)
    df_bl = lambda s: (np.clip(s, 0.0, 1.0)*(1.0 - np.clip(s, 0.0, 1.0))) / ((np.clip(s, 0.0, 1.0)**2 + 0.5*((1.0 - np.clip(s, 0.0, 1.0))**2))**2 + 1e-12)

    def simulate_bl(eps_cap):
        D_func = lambda s: eps_cap * (np.clip(s, 0.0, 1.0)**2) * ((1.0 - np.clip(s, 0.0, 1.0))**2)
        u = np.zeros(N); u[0] = 1.0
        history = {}
        rec_steps = [int(round(t / dt)) for t in times_4]
        
        for n in range(1, Nt + 1):
            u = solve_imex_ssp2_step(u, dt, dx, f_func=f_bl, df_func=df_bl, D_func=D_func, limiter='superbee', bc_left=1.0)
            if n in rec_steps:
                idx = rec_steps.index(n)
                history[times_4[idx]] = u.copy()
                
        return u, history

    print("Simulating Buckley-Leverett using unified solve_imex_ssp2_step from src/solvers.py...")
    sim_results = {}
    for ep in all_eps:
        u_f, hist = simulate_bl(ep)
        sim_results[ep] = {'final': u_f, 'history': hist}
        print(f" -> Done eps_c = {ep}")

    # MASTER FIGURE 6: 2 ROWS x 4 COLUMNS
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.8), sharex=True, sharey=True)

    # --- ROW 1: 4 times for given epsilon_c = 0.010 ---
    eps_given = 0.010
    colors_t = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    for j, tk in enumerate(times_4):
        ax = axes[0, j]
        u_t = sim_results[eps_given]['history'][tk]
        u_hyp = sim_results[0.0]['history'][tk]
        ax.plot(x, u_hyp, 'k--', lw=1.2, label=r'$\epsilon_c = 0$ (Hyperbolic)')
        ax.plot(x, u_t, color=colors_t[j], lw=2.2, label=rf'$\epsilon_c = {eps_given}$')
        ax.fill_between(x, 0, u_t, color=colors_t[j], alpha=0.10)
        
        x_s = 1.366 * tk
        ax.axvline(x_s, color='gray', linestyle=':', lw=1.0)
        ax.text(0.06, 0.90, f'({chr(97+j)}) $t = {tk:.2f}$', transform=ax.transAxes, fontsize=10.5, fontweight='bold')
        ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=':', alpha=0.55)
        ax.legend(loc='lower left', framealpha=0.90, fontsize=8.0)
        if j == 0:
            ax.set_ylabel(r'Water Saturation $u(x, t)$')

    # --- ROW 2: 4 epsilons for fixed time T_f = 0.40 ---
    labels_ep = [
        r'$\epsilon_c = 0.0$ (Shock)',
        r'$\epsilon_c = 0.005$ (Weak)',
        r'$\epsilon_c = 0.010$ (Baseline)',
        r'$\epsilon_c = 0.050$ (Strong)'
    ]
    colors_ep = ['#000000', '#2ca02c', '#1f77b4', '#ff7f0e']
    for j, ep_val in enumerate(eps_4):
        ax = axes[1, j]
        u_fin = sim_results[ep_val]['final']
        ls = '--' if ep_val == 0.0 else '-'
        ax.plot(x, u_fin, color=colors_ep[j], linestyle=ls, lw=2.2, label=labels_ep[j])
        ax.fill_between(x, 0, u_fin, color=colors_ep[j], alpha=0.10)
        
        axins = inset_axes(ax, width="40%", height="38%", loc="upper right", borderpad=1.0)
        axins.plot(x, u_fin, color=colors_ep[j], linestyle=ls, lw=2.0)
        axins.set_xlim(0.46, 0.64); axins.set_ylim(-0.05, 0.70)
        axins.grid(True, linestyle=':', alpha=0.5); axins.tick_params(labelsize=7)
        
        ax.text(0.06, 0.90, f'({chr(101+j)}) $\\epsilon_c = {ep_val}$', transform=ax.transAxes, fontsize=10.5, fontweight='bold')
        ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=':', alpha=0.55)
        ax.legend(loc='lower left', framealpha=0.90, fontsize=8.0)
        ax.set_xlabel(r'Spatial coordinate $x$')
        if j == 0:
            ax.set_ylabel(r'Water Saturation $u(x, T_f)$')

    fig.subplots_adjust(top=0.96, bottom=0.09, left=0.06, right=0.98, hspace=0.18, wspace=0.12)
    fig.savefig('figures/benchmark_3_buckley_leverett.png', dpi=300, bbox_inches='tight')
    fig.savefig('/tmp/figures/benchmark_3_buckley_leverett.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(">> Saved: figures/benchmark_3_buckley_leverett.png (Unified IMEX-Superbee Solver)")

    # Quantitative Summary Table for Buckley-Leverett across eps_c
    print("\n" + "=" * 85)
    print("=== QUANTITATIVE SUMMARY: BUCKLEY-LEVERETT REGULARIZATION (T_f = 0.40) ===")
    print("=" * 85)
    print(f"{'eps_c':<10} | {'Max Saturation':<16} | {'Min Saturation':<16} | {'Mass Deviation (%)':<20}")
    print("-" * 85)
    for ep in all_eps:
        u_fin = sim_results[ep]['final']
        mass_fin = np.sum(u_fin) * dx
        mass_dev = abs(mass_fin - Tf) / Tf * 100.0
        print(f"{ep:<10.4f} | {np.max(u_fin):<16.4f} | {np.min(u_fin):<16.4f} | {mass_dev:<20.2f}%")
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
