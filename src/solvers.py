import numpy as np
from src.numerical_schemes import solve_tridiagonal_thomas, compute_tvd_advection_flux

def solve_implicit_diffusion_step(u_star, dt, dx, D_func, tol=1e-7, max_iter=8, bc_left=None, bc_right=None):
    """Newton-Picard tridiagonal solver for nonlinear degenerate diffusion: u - dt*div(D(u)*grad(u)) = u*."""
    N = len(u_star)
    factor = dt / (dx**2)
    u = u_star.copy()
    for it in range(max_iter):
        u_prev = u.copy()
        u_mid = 0.5 * (u[:-1] + u[1:])
        D_mid = D_func(u_mid)
        
        diag = np.ones(N)
        d = factor * D_mid
        diag[:-1] += d
        diag[1:] += d
        sup = -d
        sub = -d
        
        rhs = u_star.copy()
        if bc_left is not None:
            diag[0] = 1.0; sup[0] = 0.0; rhs[0] = bc_left
        if bc_right is not None:
            diag[-1] = 1.0; sub[-1] = 0.0; rhs[-1] = bc_right
            
        u_new = solve_tridiagonal_thomas(diag, sub, sup, rhs)
        u = np.maximum(0.0, u_new)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u

def solve_imex_ssp2_step(u, dt, dx, f_func=None, df_func=None, c=None, D_func=None, limiter='superbee', bc_left=None, bc_right=None):
    """Executes a single IMEX-SSP2(2,2,2) time step combining TVD advection and implicit diffusion."""
    if f_func is None:
        vel = 1.0 if c is None else c
        f_func = lambda s, v=vel: v * s
        df_func = lambda s, v=vel: v * np.ones_like(s)
    if D_func is None:
        D_func = lambda s: np.zeros_like(s)
        
    # Stage 1:
    F1 = compute_tvd_advection_flux(u, f_func=f_func, df_func=df_func, c=c, limiter=limiter)
    u_star1 = u - (dt / dx) * np.diff(F1)
    if bc_left is not None: u_star1[0] = bc_left
    if bc_right is not None: u_star1[-1] = bc_right
    u1 = solve_implicit_diffusion_step(u_star1, dt, dx, D_func, bc_left=bc_left, bc_right=bc_right)
    
    # Stage 2:
    F2 = compute_tvd_advection_flux(u1, f_func=f_func, df_func=df_func, c=c, limiter=limiter)
    u_star2 = 0.5 * u + 0.5 * (u1 - (dt / dx) * np.diff(F2))
    if bc_left is not None: u_star2[0] = bc_left
    if bc_right is not None: u_star2[-1] = bc_right
    u_next = solve_implicit_diffusion_step(u_star2, 0.5 * dt, dx, D_func, bc_left=bc_left, bc_right=bc_right)
    return u_next
