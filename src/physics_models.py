import numpy as np

def D_power_law(u, m=2.0):
    """Diffusivity for Porous Medium Equation D(u) = u^m."""
    return np.maximum(0.0, u)**m

def D_power_law_prime(u, m=2.0):
    """Derivative of power-law diffusivity."""
    if m <= 1.0:
        return np.ones_like(u) if m == 1.0 else 0.0
    return m * (np.maximum(0.0, u)**(m - 1.0))

def f_linear_advection(u, c=1.0):
    """Linear advective flux f(u) = c*u."""
    return c * u

def df_linear_advection(u, c=1.0):
    return c * np.ones_like(u)

def f_buckley_leverett(u, M_ratio=2.0):
    """S-shaped non-convex fractional flow function."""
    u_c = np.clip(u, 0.0, 1.0)
    return (u_c**2) / (u_c**2 + (1.0 / M_ratio) * ((1.0 - u_c)**2) + 1e-14)

def df_buckley_leverett(u, M_ratio=2.0):
    u_c = np.clip(u, 0.0, 1.0)
    denom = (u_c**2 + (1.0 / M_ratio) * ((1.0 - u_c)**2))**2 + 1e-14
    return (2.0 * u_c * (1.0 - u_c) / M_ratio) / denom

def D_capillary(u, eps_cap=0.01, p=2.0, q=2.0):
    """Doubly-degenerate capillary diffusivity D(u) = eps_c * u^p * (1-u)^q."""
    u_c = np.clip(u, 0.0, 1.0)
    return eps_cap * (u_c**p) * ((1.0 - u_c)**q)
