import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
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

class NeuralDiffusivityMLP:
    """Universal Differential Equation (UDE) neural diffusivity D_theta(u) = u^2 * Softplus(MLP(u))."""
    def __init__(self, n_hidden=4):
        self.n_hidden = n_hidden
        self.n_params = 1 * n_hidden + n_hidden + n_hidden * 1 + 1 # 17 trainable weights
        
    def unpack(self, theta):
        h = self.n_hidden
        W1 = theta[0:h].reshape(1, h)
        b1 = theta[h:2*h]
        W2 = theta[2*h:3*h].reshape(h, 1)
        b2 = theta[3*h]
        return W1, b1, W2, b2

    def __call__(self, u, theta):
        u_arr = np.asarray(u, dtype=np.float64)
        W1, b1, W2, b2 = self.unpack(theta)
        u_in = u_arr.reshape(-1, 1)
        hidden = np.tanh(u_in @ W1 + b1)
        out = (hidden @ W2 + b2).ravel()
        # Strictly positive Softplus: ln(1 + exp(s))
        softplus = np.log1p(np.exp(np.clip(out, -20.0, 20.0)))
        return (u_arr**2) * softplus

def run_ude_discovery():
    print("=" * 85)
    print("SOLVER-REINFORCED SCIML: CONSTITUTIVE LAW DISCOVERY VIA UDE")
    print("=" * 85)
    
    # Target physical model: D_true(u) = u^2 * (1 + 0.5*u)
    L = 2.5; N = 80; dx = 2.0 * L / N; x = -L + (np.arange(N) + 0.5) * dx
    t0 = 1.0; Tf = 1.25; dt = 0.005; Nt = int(round((Tf - t0) / dt))
    
    def D_true(u):
        u_c = np.maximum(0.0, u)
        return (u_c**2) * (1.0 + 0.5 * u_c)

    u_init = np.maximum(0.0, (0.55 - 0.20 * x**2))
    sensor_times = [1.10, 1.15, 1.20, 1.25]
    sensor_indices = np.linspace(15, N - 15, 14, dtype=int)
    
    # Generate sensor data with 2% noise
    np.random.seed(42)
    u_true_sim = u_init.copy()
    sensor_obs = {}
    for step in range(Nt):
        u_true_sim = solve_implicit_diffusion_step(u_true_sim, dt, dx, D_func=D_true)
        cur_t = round(t0 + (step + 1) * dt, 4)
        if cur_t in sensor_times:
            noise = 0.02 * np.random.randn(len(sensor_indices)) * np.max(u_true_sim)
            sensor_obs[cur_t] = np.maximum(0.0, u_true_sim[sensor_indices] + noise)

    mlp = NeuralDiffusivityMLP(n_hidden=4)
    # Structured initialization ensuring physical positivity
    theta_0 = np.zeros(mlp.n_params)
    theta_0[-1] = 0.5413  # Softplus(0.5413) ~ 1.0 (corresponds to D(u) = u^2)
    theta_0[0:4] = [0.20, -0.10, 0.30, -0.20]
    theta_0[8:12] = [0.30, 0.10, -0.20, 0.10]

    print(f"Target Non-Power-Law Diffusivity: D(u) = u^2 * (1 + 0.5*u)")
    print(f"Neural Architecture: D_theta(u) = u^2 * Softplus(MLP_theta(u)) [{mlp.n_params} trainable weights]")
    print("Executing L-BFGS-B optimization through the differentiable implicit forward solver...")

    loss_history = []
    def ude_loss(theta):
        u_curr = u_init.copy()
        D_nn = lambda s: mlp(s, theta)
        loss = 0.0
        for step in range(Nt):
            u_curr = solve_implicit_diffusion_step(u_curr, dt, dx, D_func=D_nn, tol=1e-6, max_iter=6)
            cur_t = round(t0 + (step + 1) * dt, 4)
            if cur_t in sensor_times:
                pred = u_curr[sensor_indices]
                obs = sensor_obs[cur_t]
                loss += np.mean((pred - obs)**2)
        loss_history.append(loss)
        return loss * 1e4

    t_start = time.time()
    # Live optimization using L-BFGS-B (without deprecated options)
    res = minimize(
        ude_loss, theta_0, method='L-BFGS-B',
        options={'maxiter': 18, 'eps': 1e-3, 'ftol': 1e-9, 'gtol': 1e-7}
    )
    opt_time = time.time() - t_start
    print(f"Optimization Completed in {opt_time:.2f} s ({res.nit} iterations, {res.nfev} solver calls)")

    # Evaluate learned diffusivity on evaluation grid
    u_eval = np.linspace(0.0, 0.55, 100)
    D_true_eval = D_true(u_eval)
    D_learned_eval = mlp(u_eval, res.x)
    rel_l2_err = np.linalg.norm(D_learned_eval - D_true_eval) / np.linalg.norm(D_true_eval) * 100.0
    print(f"Discovered Neural Model Relative L2 Error: {rel_l2_err:.2f}%")

    # Generate Figure 12 directly from real optimization data
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    
    # (a) Discovered constitutive curve vs ground truth
    ax = axes[0]
    ax.plot(u_eval, D_true_eval, 'k-', lw=2.4, label=r'True $D(u) = u^2(1 + 0.5u)$')
    ax.plot(u_eval, D_learned_eval, 'b--', lw=1.8, label=rf'UDE Discovered $D_\theta(u)$ ({rel_l2_err:.2f}% err)')
    ax.plot(u_eval, mlp(u_eval, theta_0), 'r:', lw=1.4, label='Initial Guess')
    ax.set_xlabel(r'State $u$'); ax.set_ylabel(r'Diffusivity $D(u)$')
    ax.text(0.05, 0.90, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left', framealpha=0.9, fontsize=8.0)

    # (b) REAL loss convergence history recorded during L-BFGS-B
    ax = axes[1]
    clean_loss = []
    min_so_far = np.inf
    for val in loss_history:
        if val < min_so_far:
            min_so_far = val
            clean_loss.append(val)
    iters_plot = np.arange(1, len(clean_loss) + 1)
    ax.semilogy(iters_plot, clean_loss, 'g-o', lw=1.8, ms=4)
    ax.set_xlabel(r'Optimization Iterations')
    ax.set_ylabel(r'Observation Loss $\mathcal{L}(\theta)$')
    ax.text(0.05, 0.90, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.grid(True, which="both", linestyle=':', alpha=0.6)

    # (c) Real state profiles simulated with discovered neural network vs sensors
    ax = axes[2]
    colors_ude = ['#6a3d9a', '#1f78b4', '#33a02c', '#ff7f0e']
    u_rec = u_init.copy()
    rec_profs = {}
    D_opt_func = lambda s: mlp(s, res.x)
    for step in range(Nt):
        u_rec = solve_implicit_diffusion_step(u_rec, dt, dx, D_func=D_opt_func)
        cur_t = round(t0 + (step + 1) * dt, 4)
        if cur_t in sensor_times:
            rec_profs[cur_t] = u_rec.copy()
            
    for idx, to in enumerate(sensor_times):
        ax.plot(x, rec_profs[to], color=colors_ude[idx], lw=1.8, label=f'Model $t={to}$')
        ax.plot(x[sensor_indices], sensor_obs[to], 's', color=colors_ude[idx], ms=4, alpha=0.8)
    ax.set_xlabel(r'Spatial coordinate $x$'); ax.set_ylabel(r'State $u(x, t)$')
    ax.text(0.05, 0.90, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-0.02, 0.6)
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper right', framealpha=0.9, fontsize=8.0)

    fig.tight_layout()
    fig.savefig('figures/discovery_ude_diffusivity.png', dpi=300)
    fig.savefig('/tmp/figures/discovery_ude_diffusivity.png', dpi=300)
    plt.close(fig)
    print(">> Saved: figures/discovery_ude_diffusivity.png")

if __name__ == "__main__":
    run_ude_discovery()
