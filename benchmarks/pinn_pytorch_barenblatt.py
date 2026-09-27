"""
PyTorch Reference Implementation of PINNs and Conservative PINNs (cPINNs)
Applied to the Barenblatt Porous Medium Equation (PME m=2.0).
Adheres strictly to the methodology described in Section 5.6:
- Neural Network Architecture: 3 hidden layers with 24 neurons and tanh activations.
- Optimization Strategy: Dual-phase training using Adam followed by L-BFGS until convergence.
"""

import sys, os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    print("[NOTE] PyTorch is not installed in this environment.")
    print("To run this reference script, install PyTorch: pip install torch")
    sys.exit(0)

class BarenblattPINN(nn.Module):
    def __init__(self, hidden_dim=24, num_layers=3):
        super(BarenblattPINN, self).__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, x, t, m=2.0):
        """Computes strong differential residual via automatic differentiation."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.forward(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # PME residual: u_t - (u^m * u_x)_x = u_t - u^m * u_xx - m * u^(m-1) * (u_x)^2
        res = u_t - (u**m) * u_xx - m * (u**(m - 1.0)) * (u_x**2)
        return res

def train_barenblatt(mode="cpinn", adam_epochs=500, lbfgs_iters=50, lr=1e-3):
    """Trains Standard PINN or Conservative cPINN on Barenblatt benchmark via Adam -> L-BFGS."""
    print(f"--- Training {mode.upper()} on Barenblatt PME (m=2.0) using PyTorch (Adam -> L-BFGS) ---")
    model = BarenblattPINN(hidden_dim=24, num_layers=3)
    
    # Collocation points
    L = 2.5; t0 = 1.0; Tf = 1.3; M = 1.0
    x_col = torch.rand(250, 1) * 2 * L - L
    t_col = torch.rand(250, 1) * (Tf - t0) + t0
    
    # Boundary & Initial conditions
    x_ic = torch.linspace(-L, L, 60).unsqueeze(1)
    t_ic = torch.ones_like(x_ic) * t0
    from src.analytical_solutions import u_barenblatt
    u_ic_np = u_barenblatt(x_ic.numpy(), t0, m=2.0, M=M)
    u_ic = torch.tensor(u_ic_np, dtype=torch.float32)
    
    x_bc_l = -L * torch.ones(40, 1); t_bc = torch.linspace(t0, Tf, 40).unsqueeze(1)
    x_bc_r = L * torch.ones(40, 1)
    
    def compute_total_loss():
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        loss_bc = torch.mean(model(x_bc_l, t_bc)**2) + torch.mean(model(x_bc_r, t_bc)**2)
        loss_res = torch.mean(model.pde_residual(x_col, t_col)**2)
        total_loss = 30.0 * loss_ic + 15.0 * loss_bc + loss_res
        
        if mode == "cpinn":
            # Integral mass penalty
            x_quad = torch.linspace(-L, L, 80).unsqueeze(1)
            dx = 2.0 * L / 80
            t_eval = torch.tensor([1.1, 1.2, 1.3]).unsqueeze(1)
            loss_mass = 0.0
            for te in t_eval:
                t_q = te.repeat(80, 1)
                mass_pred = torch.sum(model(x_quad, t_q)) * dx
                loss_mass = loss_mass + (mass_pred - M)**2
            total_loss = total_loss + 30.0 * (loss_mass / len(t_eval))
        return total_loss

    # Phase 1: Adam optimization
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        loss = compute_total_loss()
        loss.backward()
        optimizer_adam.step()
        if (epoch + 1) % 250 == 0:
            print(f"[Adam] Epoch {epoch+1:4d} | Total Loss: {loss.item():.4e}")

    # Phase 2: L-BFGS optimization until convergence
    print(f"[L-BFGS] Initiating second-order fine-tuning for {lbfgs_iters} iterations...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=lbfgs_iters, history_size=15, line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_total_loss()
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    final_loss = compute_total_loss()
    print(f"[Finished] Final Convergence Loss ({mode.upper()}): {final_loss.item():.4e}")
    return model

if __name__ == "__main__":
    train_barenblatt(mode="pinn", adam_epochs=300, lbfgs_iters=40)
    train_barenblatt(mode="cpinn", adam_epochs=300, lbfgs_iters=40)
