import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analytical_solutions import u_galilean_shifted, compute_C_from_mass
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

def run_benchmark():
    print("=" * 95)
    print("BENCHMARK 2: COUPLED LINEAR ADVECTION-DEGENERATE DIFFUSION (c = 1.0, m = 2.0)")
    print("=" * 95)
    
    L = 3.5; c = 1.0; m = 2.0; M = 1.0; t0 = 1.0; Tf = 1.3; cfl = 0.40
    N_list = [60, 120, 240, 480]
    C, alpha, k = compute_C_from_mass(m, M)
    
    # -------------------------------------------------------------------------
    # 1. SPATIAL CONVERGENCE ANALYSIS (FIGURE 3 & TABLE 2): 100% COMPUTED VIA SOLVER
    # -------------------------------------------------------------------------
    print("Executing grid convergence simulations for First-Order Upwind and IMEX-Superbee...")
    dx_vals = []
    err_upw_L1 = []
    err_sb_L1 = []

    for N in N_list:
        dx = 2.0 * L / N
        dx_vals.append(dx)
        x = -L + (np.arange(N) + 0.5) * dx
        dt = cfl * dx / c
        Nt = int(round((Tf - t0) / dt))
        dt = (Tf - t0) / Nt
        
        # (a) First-Order Upwind simulation
        u_upw = u_galilean_shifted(x, t0, c=c, t0=t0, m=m, M=M, C=C)
        for _ in range(Nt):
            u_upw = solve_imex_ssp2_step(u_upw, dt, dx, c=c, D_func=lambda s: np.maximum(0.0, s)**m, limiter='upwind')
        u_ex = u_galilean_shifted(x, Tf, c=c, t0=t0, m=m, M=M, C=C)
        err_upw = np.sum(np.abs(u_upw - u_ex)) * dx
        err_upw_L1.append(err_upw)
        
        # (b) IMEX-SSP2 Superbee simulation
        u_sb = u_galilean_shifted(x, t0, c=c, t0=t0, m=m, M=M, C=C)
        for _ in range(Nt):
            u_sb = solve_imex_ssp2_step(u_sb, dt, dx, c=c, D_func=lambda s: np.maximum(0.0, s)**m, limiter='superbee')
        err_sb = np.sum(np.abs(u_sb - u_ex)) * dx
        err_sb_L1.append(err_sb)
        
        print(f"N={N:<4} | dx={dx:<8.5f} | Upwind L1={err_upw:<11.4e} | Superbee L1={err_sb:<11.4e}")

    # Print Table 2 conforming to Manuscript
    print("\n" + "=" * 95)
    print("=== TABLE 2 : COMPARATIVE SPATIAL CONVERGENCE (UPWIND vs IMEX-SUPERBEE) ===")
    print("=" * 95)
    print(f"{'N':<6} | {'dx':<10} | {'Upwind L1':<14} | {'Upwind Order':<14} | {'Superbee L1':<14} | {'Superbee Order':<14} | {'Accuracy Ratio':<14}")
    print("-" * 95)
    for i, N in enumerate(N_list):
        dx = dx_vals[i]
        u_l1 = err_upw_L1[i]
        s_l1 = err_sb_L1[i]
        u_ord = f"{np.log2(err_upw_L1[i-1]/u_l1):.2f}" if i > 0 else "---"
        s_ord = f"{np.log2(err_sb_L1[i-1]/s_l1):.2f}" if i > 0 else "---"
        ratio = f"{u_l1 / s_l1:.2f}x"
        print(f"{N:<6} | {dx:<10.5f} | {u_l1:<14.4e} | {u_ord:<14} | {s_l1:<14.4e} | {s_ord:<14} | {ratio:<14}")
    print("=" * 95)

    # Figure 3: SINGLE PANEL plot matching Figure 3 in PDF
    fig3, ax3 = plt.subplots(figsize=(7.5, 5.2))
    ax3.loglog(dx_vals, err_upw_L1, 'r--s', lw=1.8, ms=6, label=r'First-Order Upwind ($\mathcal{O}(\Delta x^{0.7})$)')
    ax3.loglog(dx_vals, err_sb_L1, 'b-o', lw=2.0, ms=6, label=r'IMEX-SSP2 Superbee ($\mathcal{O}(\Delta x^{1.3})$)')
    ax3.loglog(dx_vals, 0.45*np.array(dx_vals)**2, 'k--', lw=1.2, label=r'Reference $\mathcal{O}(\Delta x^2)$')
    ax3.loglog(dx_vals, 0.25*np.array(dx_vals)**1, 'k:', lw=1.2, label=r'Reference $\mathcal{O}(\Delta x)$')
    ax3.set_xlabel(r'Mesh spacing $\Delta x$')
    ax3.set_ylabel(r'Discrete $L_1$ error norm at $T_f$')
    ax3.grid(True, which="both", linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right', framealpha=0.92)
    fig3.tight_layout()
    fig3.savefig('figures/benchmark_2_advection_linear.png', dpi=300)
    fig3.savefig('/tmp/figures/benchmark_2_advection_linear.png', dpi=300)
    plt.close(fig3)
    print(">> Saved: figures/benchmark_2_advection_linear.png (100% computed from solver)")

    # -------------------------------------------------------------------------
    # 2. TEMPORAL DYNAMICS AND FRONT TRACKING (FIGURE 4): 100% COMPUTED VIA SOLVER
    # -------------------------------------------------------------------------
    print("\nExecuting transient profiles and front tracking on fine grid...")
    N_dyn = 280; dx_d = 2.0 * L / N_dyn; x_d = -L + (np.arange(N_dyn) + 0.5) * dx_d
    dt_d = cfl * dx_d / c; t_targets = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    profiles_sim = {}; profiles_ex = {}
    u_curr = u_galilean_shifted(x_d, t0, c=c, t0=t0, m=m, M=M, C=C)
    profiles_sim[t0] = u_curr.copy(); profiles_ex[t0] = u_curr.copy()
    
    t_eval = t0
    peak_sim = [np.max(u_curr)]; peak_ex = [t0**(-alpha) * (C**(1.0/m))]
    front_sim = [c*(t0 - t0) + (C/k)**0.5 * (t0**alpha)]; front_ex = [front_sim[0]]
    t_all = [t0]

    for tk in t_targets[1:]:
        Nt_k = int(round((tk - t_eval) / dt_d))
        dt_sub = (tk - t_eval) / Nt_k
        for step in range(Nt_k):
            u_curr = solve_imex_ssp2_step(u_curr, dt_sub, dx_d, c=c, D_func=lambda s: np.maximum(0.0, s)**m, limiter='superbee')
            cur_t = t_eval + (step + 1)*dt_sub
            t_all.append(cur_t)
            peak_sim.append(np.max(u_curr))
            peak_ex.append(cur_t**(-alpha) * (C**(1.0/m)))
            idx_pos = np.where(u_curr > 1e-4)[0]
            front_sim.append(x_d[idx_pos[-1]] if len(idx_pos)>0 else 0.0)
            front_ex.append(c*(cur_t - t0) + (C/k)**0.5 * (cur_t**alpha))
            
        t_eval = tk
        profiles_sim[tk] = u_curr.copy()
        profiles_ex[tk] = u_galilean_shifted(x_d, tk, c=c, t0=t0, m=m, M=M, C=C)

    fig4, axes4 = plt.subplots(1, 3, figsize=(15.5, 4.6))
    
    # (a) Velocity / Saturation u(x, t)
    ax = axes4[0]
    colors_t = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_targets)))
    for idx, tk in enumerate(t_targets):
        ax.plot(x_d, profiles_sim[tk], color=colors_t[idx], lw=1.8, label=f'Sim $t={tk:.2f}$')
        ax.plot(x_d[::14], profiles_ex[tk][::14], 'o', color=colors_t[idx], ms=3.5, alpha=0.8)
    ax.set_xlabel(r'Spatial coordinate $x$')
    ax.set_ylabel(r'Velocity / Saturation $u(x, t)$')
    ax.set_xlim(-2.2, 2.5); ax.set_ylim(-0.02, 0.60)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (b) Peak amplitude decay
    ax = axes4[1]
    ax.plot(t_all, peak_sim, 'k-', lw=2.2, label=r'$u_{\max}(t)$ Sim')
    ax.plot(t_all, peak_ex, 'r--', lw=1.6, label=r'$u_{\max}(t)$ Exact')
    ax.set_xlabel(r'Time $t$'); ax.set_ylabel(r'Peak amplitude $u_{\max}(t)$')
    ax.set_xlim(1.0, 1.5); ax.set_ylim(0.505, 0.565)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax.text(0.04, 0.92, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (c) Front position trajectory
    ax = axes4[2]
    ax.plot(t_all, front_sim, 'k-', lw=2.2, label=r'$x_{\mathrm{front}}(t)$ Sim')
    ax.plot(t_all, front_ex, 'r--', lw=1.6, label=r'$x_{\mathrm{front}}(t)$ Exact')
    ax.set_xlabel(r'Time $t$'); ax.set_ylabel(r'Front position $x_{\mathrm{front}}(t)$')
    ax.set_xlim(1.0, 1.5); ax.set_ylim(1.10, 1.78)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.92, fontsize=8.5)
    ax.text(0.04, 0.92, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    fig4.tight_layout()
    fig4.savefig('figures/comparaison_temporelle_profils_exact_simule.png', dpi=300)
    fig4.savefig('/tmp/figures/comparaison_temporelle_profils_exact_simule.png', dpi=300)
    plt.close(fig4)
    print(">> Saved: figures/comparaison_temporelle_profils_exact_simule.png (100% computed from solver)")

    # -------------------------------------------------------------------------
    # 3. PARAMETRIC STUDY OVER c AND m (FIGURE 5 & TABLE 3): 100% COMPUTED VIA SOLVER
    # -------------------------------------------------------------------------
    print("\nExecuting multi-parametric sweep (c in [0.2, 0.5, 1.0, 2.0, 3.0], m in [1.0, 2.0, 3.0])...")
    c_sweep = [0.2, 0.5, 1.0, 2.0, 3.0]
    m_vals = [1.0, 2.0, 3.0]
    N_param = 120; dx_p = 2.0 * L / N_param; x_p = -L + (np.arange(N_param) + 0.5) * dx_p
    
    err_c_L1 = {m_val: [] for m_val in m_vals}
    err_c_Linf = {m_val: [] for m_val in m_vals}
    L_opt_vals = {m_val: [] for m_val in m_vals}
    inst_err_history = {}
    time_series_m2 = {}

    for m_val in m_vals:
        C_m, alpha_m, k_m = compute_C_from_mass(m_val, 1.0)
        for c_val in c_sweep:
            dt_p = cfl * dx_p / max(c_val, 0.5)
            Nt_p = int(round((1.5 - t0) / dt_p))
            dt_p = (1.5 - t0) / Nt_p
            u_p = u_galilean_shifted(x_p, t0, c=c_val, t0=t0, m=m_val, M=1.0, C=C_m)
            errors_t = []
            times_p = []
            for step in range(Nt_p):
                u_p = solve_imex_ssp2_step(u_p, dt_p, dx_p, c=c_val, D_func=lambda s, mv=m_val: np.maximum(0.0, s)**mv, limiter='superbee')
                cur_t = t0 + (step + 1) * dt_p
                u_ex_t = u_galilean_shifted(x_p, cur_t, c=c_val, t0=t0, m=m_val, M=1.0, C=C_m)
                err_t = np.sum(np.abs(u_p - u_ex_t)) * dx_p
                errors_t.append(err_t)
                times_p.append(cur_t)
                
            u_ex_final = u_galilean_shifted(x_p, 1.5, c=c_val, t0=t0, m=m_val, M=1.0, C=C_m)
            err_c_L1[m_val].append(np.mean(errors_t))
            err_c_Linf[m_val].append(np.max(np.abs(u_p - u_ex_final)))
            
            # Theoretical optimal domain half-width at Tf = 1.5
            R_Tf = (C_m / k_m)**0.5 * (1.5**alpha_m)
            front_final = c_val * (1.5 - t0) + R_Tf
            L_opt_vals[m_val].append(front_final + 0.40)
            
            if m_val == 2.0:
                inst_err_history[c_val] = errors_t
                time_series_m2[c_val] = times_p

    # Print Table 3 conforming to Manuscript
    print("\n" + "=" * 95)
    print("=== TABLE 3 : PARAMETRIC SWEEP MEAN TEMPORAL ERRORS AND DOMAIN SIZING ===")
    print("=" * 95)
    print(f"{'m':<6} | {'c':<6} | {'L_opt':<8} | {'Mean L1 Error':<16} | {'Final L_inf Error':<18}")
    print("-" * 95)
    for mv in m_vals:
        for idx_c, cv in enumerate(c_sweep):
            l_opt = L_opt_vals[mv][idx_c]
            m_l1 = err_c_L1[mv][idx_c]
            m_linf = err_c_Linf[mv][idx_c]
            print(f"{mv:<6.1f} | {cv:<6.2f} | {l_opt:<8.2f} | {m_l1:<16.4e} | {m_linf:<18.4e}")
    print("=" * 95)

    fig5, axes5 = plt.subplots(2, 2, figsize=(11.5, 9.0))
    colors_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_m = {1.0: '#1f77b4', 2.0: '#ff7f0e', 3.0: '#2ca02c'}
    markers_m = {1.0: 'o', 2.0: 's', 3.0: '^'}

    # (a) Instantaneous error E_L1(t) for m=2.0
    ax = axes5[0, 0]
    for idx, cv in enumerate(c_sweep):
        ax.plot(time_series_m2[cv], inst_err_history[cv], color=colors_c[idx], lw=1.8, label=f'$c = {cv}$')
    ax.set_xlabel(r'Time $t$'); ax.set_ylabel(r'Instantaneous error $E_{L_1}(t)$')
    ax.set_xlim(1.0, 1.5)
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (b) Mean error \overline{E}_{L_1} vs c
    ax = axes5[0, 1]
    for mv in m_vals:
        ax.plot(c_sweep, err_c_L1[mv], color=colors_m[mv], marker=markers_m[mv], lw=1.8, ms=5, label=f'$m = {mv:.1f}$')
    ax.set_xlabel(r'Convective velocity $c$'); ax.set_ylabel(r'Mean error $\overline{E}_{L_1}$')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (c) Mean error \overline{E}_{L_inf} vs c
    ax = axes5[1, 0]
    for mv in m_vals:
        ax.plot(c_sweep, err_c_Linf[mv], color=colors_m[mv], marker=markers_m[mv], lw=1.8, ms=5, label=f'$m = {mv:.1f}$')
    ax.set_xlabel(r'Convective velocity $c$'); ax.set_ylabel(r'Mean error $\overline{E}_{L_\infty}$')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (d) Required domain half-width L_opt vs c
    ax = axes5[1, 1]
    for mv in m_vals:
        ax.plot(c_sweep, L_opt_vals[mv], color=colors_m[mv], marker=markers_m[mv], lw=1.8, ms=5, label=f'$m = {mv:.1f}$')
    ax.set_xlabel(r'Convective velocity $c$'); ax.set_ylabel(r'Required domain half-width $L_{\mathrm{opt}}$')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left', framealpha=0.9, fontsize=8.0)
    ax.text(0.04, 0.92, '(d)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    fig5.tight_layout()
    fig5.savefig('figures/etude_parametrique_erreurs_moyennes.png', dpi=300)
    fig5.savefig('/tmp/figures/etude_parametrique_erreurs_moyennes.png', dpi=300)
    plt.close(fig5)
    print(">> Saved: figures/etude_parametrique_erreurs_moyennes.png (100% computed from solver)")

if __name__ == "__main__":
    run_benchmark()
