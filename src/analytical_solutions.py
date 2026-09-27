import numpy as np
from scipy.special import beta as beta_func

def compute_C_from_mass(m, M=1.0):
    """Inverts the exact integration constant C from total mass M via Euler's Beta function."""
    alpha = 1.0 / (m + 2.0)
    k = m / (2.0 * (m + 2.0))
    B_val = beta_func(0.5, 1.0 / m + 1.0)
    C = ((M * np.sqrt(k)) / B_val)**((2.0 * m) / (m + 2.0))
    return C, alpha, k

def u_barenblatt(x, t, m=2.0, M=1.0, C=None):
    """Evaluates the exact self-similar Barenblatt-Pattle solution."""
    t = np.maximum(t, 1e-10)
    if C is None:
        C, alpha, k = compute_C_from_mass(m, M)
    else:
        alpha = 1.0 / (m + 2.0)
        k = m / (2.0 * (m + 2.0))
    
    arg = C - k * (x**2) / (t**(2.0 * alpha))
    pos = np.maximum(0.0, arg)
    return (t**(-alpha)) * (pos**(1.0 / m))

def u_galilean_shifted(x, t, c=1.0, t0=1.0, m=2.0, M=1.0, C=None):
    """Evaluates the exact Galilean-shifted advection-diffusion solution."""
    x_shifted = x - c * (t - t0)
    return u_barenblatt(x_shifted, t, m=m, M=M, C=C)
