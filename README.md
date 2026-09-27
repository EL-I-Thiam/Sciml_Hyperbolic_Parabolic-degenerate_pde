# Differentiable IMEX-Superbee Framework for Hyperbolic Conservation Laws with Degenerate Diffusion

**A High-Order, Conservative, and Physics-Differentiable SciML Framework**

*Author:* Pr. Elhadji Ibrahima Thiam  
*Affiliation:* Laboratoire d'Analyse Numérique et d'Informatique (LANI), Université Gaston Berger, Saint-Louis, Senegal  
*Journal Submission:* *Computers & Mathematics with Applications* (Elsevier)

---

## 1. Overview & Mathematical Context

Nonlinear hyperbolic-parabolic conservation laws with degenerate diffusion govern two-phase Darcy transport, nonlinear filtration, and sedimentation:
$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = \frac{\partial}{\partial x}\left( D(u) \frac{\partial u}{\partial x} \right)$$

While standard Physics-Informed Neural Networks (PINNs) have become popular in scientific machine learning, they suffer from structural breakdowns on degenerate operators:
1. **Breakdown of Automatic Differentiation across Interfaces**: For Barenblatt moving free boundaries where the derivative diverges ($\partial_x u \to -\infty$), strong residuals diverge.
2. **Spectral Bias & Non-Entropic Shock Smearing**: Neural networks smear non-convex hyperbolic shocks (Welge fronts in Buckley-Leverett).
3. **Violation of Physical Invariants**: Mass conservation drifts by $28\%$ to $71\%$, and solutions violate positivity ($u < 0$) or maximum saturation ($u > 1$).

This repository implements a **Differentiable Physics / Grey-Box SciML framework** that provides:
* **$100\%$ Exact Discrete Mass Conservation** by conservative finite volume construction.
* **Unconditional Positivity Preservation ($u \ge 0$)** via $M$-matrix properties.
* **High-Order TVD Superbee Reconstruction** resolving sharp fronts without Gibbs oscillations.
* **Linear Time $\mathcal{O}(N)$ Tridiagonal Solver** via the vectorized Thomas algorithm.

---

## 2. Directory Structure

```text
sciml_hyperbolic_degenerate_pde/
├── src/
│   ├── analytical_solutions.py          # Barenblatt, Galilean shift, Beta mass inversion
│   ├── numerical_schemes.py             # TVD Superbee limiter, MUSCL, Thomas solver
│   ├── physics_models.py                # Diffusivity & flux laws (PME, Buckley-Leverett, Burgers)
│   └── solvers.py                       # IMEX-SSP2(2,2,2) and Newton-Raphson solvers
├── benchmarks/
│   ├── benchmark_barenblatt_pure.py     # Benchmark 1: m=1, 2, 3 convergence study
│   ├── benchmark_advection_linear.py    # Benchmark 2: Upwind vs IMEX-Superbee accuracy gain
│   ├── benchmark_buckley_leverett.py    # Benchmark 3: Capillary sensitivity study (eps_c)
│   ├── benchmark_burgers_sciml.py       # Benchmark 4: Viscous Burgers data assimilation
│   ├── benchmark_pinn_cpinn_comparison.py # Standalone dual-engine PINN/cPINN benchmark
│   ├── pinn_pytorch_barenblatt.py       # PyTorch reference PINN & cPINN for Barenblatt PME
│   └── pinn_pytorch_buckley_leverett.py # PyTorch reference PINN & cPINN for Buckley-Leverett
├── inverse_problem/
│   ├── calibrate_diffusivity_pme.py     # Inverse calibration for Porous Medium Equation
│   └── calibrate_buckley_leverett.py    # Inverse calibration for Buckley-Leverett capillarity
├── figures/                             # High-resolution 300 DPI figures (English labels, no titles)
├── requirements.txt                     # Dependencies (torch, numpy, scipy, matplotlib)
├── run_all_experiments.py               # Master execution script
├── LICENSE                              # MIT License
└── README.md
```

---

## 3. PyTorch PINN and cPINN Reference Implementations

In strict accordance with foundational literature:
- **PINNs**: Raissi, Perdikaris, Karniadakis, *Journal of Computational Physics* (2019)
- **cPINNs**: Jagtap, Kharazmi, Karniadakis, *Computer Methods in Applied Mechanics and Engineering* (2020)

To execute the PyTorch implementations:
```bash
pip install -r requirements.txt

# Run PyTorch PINN/cPINN on Barenblatt PME
python benchmarks/pinn_pytorch_barenblatt.py

# Run PyTorch PINN/cPINN on Buckley-Leverett Flow
python benchmarks/pinn_pytorch_buckley_leverett.py

# Run unified comparison and generate publication figures
python benchmarks/benchmark_pinn_cpinn_comparison.py
```
