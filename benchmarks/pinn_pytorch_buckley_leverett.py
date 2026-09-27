"""
PyTorch Reference Implementation of PINNs and cPINNs
Applied to Buckley-Leverett Two-Phase Flow with Doubly-Degenerate Capillarity.
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

class BuckleyLeverettPINN(nn.Module):
    def __init__(self, hidden_dim=24, num_layers=3):
        super(BuckleyLeverettPINN, self).__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, x, t, M_ratio=2.0, eps_cap=0.01):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.forward(x, t)
        u_c = torch.clamp(u, 0.0, 1.0)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Flux derivative df/du
        denom = (u_c**2 + (1.0 / M_ratio) * ((1.0 - u_c)**2))**2 + 1e-12
        df_du = (2.0 * u_c * (1.0 - u_c) / M_ratio) / denom
        
        # Capillary diffusion: D(u) = eps_c * u^2 * (1-u)^2
        D = eps_cap * (u_c**2) * ((1.0 - u_c)**2)
        dD_du = 2.0 * eps_cap * u_c * (1.0 - u_c) * (1.0 - 2.0 * u_c)
        
        res = u_t + df_du * u_x - D * u_xx - dD_du * (u_x**2)
        return res

def train_buckley_leverett(mode="cpinn", adam_epochs=500, lbfgs_iters=50, lr=1e-3):
    print(f"--- Training {mode.upper()} on Buckley-Leverett Flow using PyTorch (Adam -> L-BFGS) ---")
    model = BuckleyLeverettPINN(hidden_dim=24, num_layers=3)
    
    L = 1.0; Tf = 0.40
    x_col = torch.rand(300, 1) * L
    t_col = torch.rand(300, 1) * Tf
    
    x_ic = torch.linspace(0.01, L, 50).unsqueeze(1); t_ic = torch.zeros_like(x_ic)
    x_bc = torch.zeros(40, 1); t_bc = torch.linspace(0.0, Tf, 40).unsqueeze(1)
    
    def compute_total_loss():
        loss_ic = torch.mean(model(x_ic, t_ic)**2) # u(x, 0) = 0
        loss_bc = torch.mean((model(x_bc, t_bc) - 1.0)**2) # u(0, t) = 1
        loss_res = torch.mean(model.pde_residual(x_col, t_col)**2)
        total_loss = 30.0 * loss_ic + 20.0 * loss_bc + loss_res
        
        if mode == "cpinn":
            # Total injected water mass = t
            x_quad = torch.linspace(0.0, L, 60).unsqueeze(1); dx = L / 60
            t_eval = torch.tensor([0.1, 0.2, 0.3, 0.4]).unsqueeze(1)
            loss_mass = 0.0
            for te in t_eval:
                t_q = te.repeat(60, 1)
                mass_pred = torch.sum(model(x_quad, t_q)) * dx
                loss_mass = loss_mass + (mass_pred - te)**2
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
    train_buckley_leverett(mode="pinn", adam_epochs=300, lbfgs_iters=40)
    train_buckley_leverett(mode="cpinn", adam_epochs=300, lbfgs_iters=40)
