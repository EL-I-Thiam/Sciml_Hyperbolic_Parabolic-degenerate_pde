import numpy as np

def superbee_limiter(r):
    """Compressive TVD Superbee flux limiter: phi(r) = max(0, min(2r, 1), min(r, 2))."""
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))

def solve_tridiagonal_thomas(diag, sub, sup, rhs):
    """Solves a tridiagonal linear system Ax = rhs in O(N) operations via the Thomas algorithm."""
    n = len(diag)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - sub[i-1] * cp[i-1]
        if i < n - 1:
            cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i-1] * dp[i-1]) / denom
        
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def compute_tvd_advection_flux(u, f_func=None, df_func=None, c=None, limiter='superbee'):
    """Vectorized MUSCL TVD numerical face flux F_{i+1/2} with Rusanov Riemann solver for general f(u)."""
    N = len(u)
    if f_func is None:
        vel = 1.0 if c is None else c
        f_func = lambda s, v=vel: v * s
        df_func = lambda s, v=vel: v * np.ones_like(s)
        
    u_ext = np.zeros(N + 4)
    u_ext[2:N+2] = u
    u_ext[0] = u[0]; u_ext[1] = u[0]
    u_ext[N+2] = u[-1]; u_ext[N+3] = u[-1]
    
    diff = np.diff(u_ext)
    r = np.zeros(len(diff))
    den = diff[1:]
    num = diff[:-1]
    mask = np.abs(den) > 1e-12
    r[1:][mask] = num[mask] / den[mask]
    
    if limiter == 'superbee':
        phi = np.maximum(0.0, np.maximum(np.minimum(2.0*r, 1.0), np.minimum(r, 2.0)))
    elif limiter == 'upwind':
        phi = np.zeros_like(r)
    else:
        phi = np.maximum(0.0, np.minimum(1.0, r)) # minmod
        
    idx = np.arange(N + 1) + 2
    u_L = u_ext[idx-1] + 0.5 * phi[idx-1] * (u_ext[idx] - u_ext[idx-1])
    u_R = u_ext[idx] - 0.5 * phi[idx] * (u_ext[idx+1] - u_ext[idx])
    
    ws = np.maximum(np.abs(df_func(u_L)), np.abs(df_func(u_R)))
    F = 0.5 * (f_func(u_L) + f_func(u_R)) - 0.5 * ws * (u_R - u_L)
    return F
