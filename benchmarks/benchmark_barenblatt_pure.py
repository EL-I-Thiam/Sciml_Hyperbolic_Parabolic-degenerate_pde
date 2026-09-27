import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.analytical_solutions import u_barenblatt, compute_C_from_mass
from src.solvers import solve_implicit_diffusion_step

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

def run_benchmark():
    print("=" * 85)
    print("BENCHMARK 1: PURE NONLINEAR DEGENERATE DIFFUSION (PME m = 1.0, 2.0, 3.0)")
    print("=" * 85)
    
    L = 3.5; M = 1.0; t0 = 1.0; Tf = 1.3; dt_fixed = 0.0005
    m_values = [1.0, 2.0, 3.0]
    N_list = [60, 120, 240, 480]
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(15.0, 4.8))
    errors_L1 = {m: [] for m in m_values}
    errors_Linf = {m: [] for m in m_values}
    dx_vals = []
    
    for idx, m in enumerate(m_values):
        C, alpha, k = compute_C_from_mass(m, M)
        R_Tf = (C / k)**0.5 * (Tf**alpha)
        
        # Profile run on N=240
        N_plot = 240
        dx_p = 2.0 * L / N_plot
        x_p = -L + (np.arange(N_plot) + 0.5) * dx_p
        u_p = u_barenblatt(x_p, t0, m=m, M=M, C=C)
        Nt = int(round((Tf - t0) / dt_fixed))
        dt = (Tf - t0) / Nt
        for n in range(Nt):
            u_p = solve_implicit_diffusion_step(u_p, dt, dx_p, lambda s, m_val=m: np.maximum(0.0, s)**m_val)
        u_ex_p = u_barenblatt(x_p, Tf, m=m, M=M, C=C)
        
        ax = axes1[idx]
        ax.plot(x_p, u_ex_p, 'k-', lw=2.2, label='Exact Analytical')
        ax.plot(x_p, u_p, 'b--', lw=1.6, label='Proposed IMEX-Superbee')
        ax.text(0.05, 0.90, f'({chr(97+idx)}) $m = {m:.1f}$', transform=ax.transAxes, fontsize=11, fontweight='bold')
        ax.set_xlabel(r'Spatial coordinate $x$')
        if idx == 0:
            ax.set_ylabel(r'Solution $u(x, T_f)$')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.02, 0.65)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.9)
        
        axins = inset_axes(ax, width="40%", height="35%", loc="upper left", borderpad=1.5)
        mask = (x_p >= R_Tf - 0.35) & (x_p <= R_Tf + 0.2)
        axins.plot(x_p[mask], u_ex_p[mask], 'k-', lw=2.0)
        axins.plot(x_p[mask], u_p[mask], 'b--', lw=1.5)
        axins.axvline(R_Tf, color='gray', linestyle=':', lw=1.2)
        axins.grid(True, linestyle=':', alpha=0.5)
        axins.tick_params(labelsize=8)
        
        for N in N_list:
            dx = 2.0 * L / N
            if idx == 0:
                dx_vals.append(dx)
            x = -L + (np.arange(N) + 0.5) * dx
            u = u_barenblatt(x, t0, m=m, M=M, C=C)
            for n in range(Nt):
                u = solve_implicit_diffusion_step(u, dt, dx, lambda s, m_val=m: np.maximum(0.0, s)**m_val)
            u_ex = u_barenblatt(x, Tf, m=m, M=M, C=C)
            err_L1 = np.sum(np.abs(u - u_ex)) * dx
            err_Linf = np.max(np.abs(u - u_ex))
            errors_L1[m].append(err_L1)
            errors_Linf[m].append(err_Linf)

    # Print Table 1 conforming to Manuscript
    print("\n" + "=" * 85)
    print("=== TABLE 1 : SPATIAL ERROR NORMS AND NUMERICAL CONVERGENCE RATES (BARENBLATT) ===")
    print("=" * 85)
    for m in m_values:
        print(f"\n[Case m = {m:.1f}]")
        print(f"{'N':<6} | {'dx':<10} | {'L_inf Error':<14} | {'L_inf Order':<12} | {'L1 Error':<14} | {'L1 Order':<10}")
        print("-" * 75)
        for i, N in enumerate(N_list):
            dx = dx_vals[i]
            linf = errors_Linf[m][i]
            l1 = errors_L1[m][i]
            ord_linf = f"{np.log2(errors_Linf[m][i-1] / linf):.2f}" if i > 0 else "---"
            ord_l1 = f"{np.log2(errors_L1[m][i-1] / l1):.2f}" if i > 0 else "---"
            print(f"{N:<6} | {dx:<10.5f} | {linf:<14.4e} | {ord_linf:<12} | {l1:<14.4e} | {ord_l1:<10}")
    print("=" * 85)

    fig1.savefig('figures/benchmark_1_comparaison_exacte_approchee.png', dpi=300)
    fig1.savefig('/tmp/figures/benchmark_1_comparaison_exacte_approchee.png', dpi=300)
    plt.close(fig1)
    print("\n>> Saved: figures/benchmark_1_comparaison_exacte_approchee.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {1.0: 'r', 2.0: 'b', 3.0: 'g'}
    markers = {1.0: 'o', 2.0: 's', 3.0: '^'}
    ax = axes2[0]
    for m in m_values:
        ax.loglog(dx_vals, errors_L1[m], color=colors[m], marker=markers[m], lw=1.8, ms=6, label=f'$m = {m:.1f}$')
    ax.loglog(dx_vals, 0.15*np.array(dx_vals)**1, 'k:', lw=1.2, label=r'$\mathcal{O}(\Delta x)$')
    ax.loglog(dx_vals, 0.4*np.array(dx_vals)**2, 'k--', lw=1.2, label=r'$\mathcal{O}(\Delta x^2)$')
    ax.set_xlabel(r'Mesh spacing $\Delta x$'); ax.set_ylabel(r'$L_1$ Error')
    ax.text(0.05, 0.90, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, which="both", linestyle=':', alpha=0.6); ax.legend(loc='lower right', framealpha=0.9)

    ax = axes2[1]
    for m in m_values:
        ax.loglog(dx_vals, errors_Linf[m], color=colors[m], marker=markers[m], lw=1.8, ms=6, label=f'$m = {m:.1f}$')
    ax.loglog(dx_vals, 0.12*np.array(dx_vals)**(1/2), 'k:', lw=1.2, label=r'$\mathcal{O}(\Delta x^{1/2})$')
    ax.set_xlabel(r'Mesh spacing $\Delta x$'); ax.set_ylabel(r'$L_\infty$ Error')
    ax.text(0.05, 0.90, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, which="both", linestyle=':', alpha=0.6); ax.legend(loc='lower right', framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig('figures/benchmark_1_convergence_m123.png', dpi=300)
    fig2.savefig('/tmp/figures/benchmark_1_convergence_m123.png', dpi=300)
    plt.close(fig2)
    print(">> Saved: figures/benchmark_1_convergence_m123.png")

if __name__ == "__main__":
    run_benchmark()
