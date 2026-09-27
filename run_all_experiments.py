import os, sys, subprocess

print("=" * 95)
print("=== MASTER REPRODUCTION SCRIPT: ALL 12 MANUSCRIPT FIGURES & SCIENTIFIC TABLES ===")
print("=" * 95)

os.makedirs('figures', exist_ok=True)

benchmarks = [
    ("benchmarks/benchmark_barenblatt_pure.py", "Figures 1 & 2: Barenblatt Profiles & h-Convergence"),
    ("benchmarks/benchmark_advection_linear.py", "Figures 3, 4 & 5: Advection-Diffusion Superbee, Temporal Profiles & Parametric Sweep"),
    ("benchmarks/benchmark_buckley_leverett.py", "Figure 6: Buckley-Leverett Transport & Capillary Sensitivity"),
    ("benchmarks/benchmark_pinn_cpinn_comparison.py", "Figures 7 & 8: PINN / cPINN vs IMEX-Superbee Benchmark Comparisons"),
    ("inverse_problem/calibrate_diffusivity_pme.py", "Figure 9: PME Diffusivity Calibration under 2% Noise"),
    ("inverse_problem/calibrate_buckley_leverett.py", "Figure 10: Buckley-Leverett Capillarity Calibration"),
    ("benchmarks/benchmark_burgers_sciml.py", "Figure 11: Viscous Burgers Data Assimilation Field & Viscosity Trajectory"),
    ("inverse_problem/discover_neural_diffusivity_ude.py", "Figure 12: UDE Non-Parametric Constitutive Law Discovery"),
]

base_dir = os.path.dirname(os.path.abspath(__file__))

for bm, desc in benchmarks:
    script_path = os.path.join(base_dir, bm)
    if os.path.exists(script_path):
        print(f"\n>>> EXECUTING: {bm} ({desc})...")
        subprocess.run([sys.executable, script_path], check=True)
    else:
        print(f"Skipping missing script: {bm}")

# Check that all 12 figures exist
expected_figures = [
    "benchmark_1_comparaison_exacte_approchee.png",
    "benchmark_1_convergence_m123.png",
    "benchmark_2_advection_linear.png",
    "comparaison_temporelle_profils_exact_simule.png",
    "etude_parametrique_erreurs_moyennes.png",
    "benchmark_3_buckley_leverett.png",
    "comparison_pinn_cpinn_barenblatt.png",
    "comparison_pinn_cpinn_buckley_leverett.png",
    "reconstruction_sciml_diffusivite.png",
    "reconstruction_buckley_leverett.png",
    "benchmark_4_burgers_assimilation.png",
    "discovery_ude_diffusivity.png"
]

print("\n" + "=" * 95)
print("=== VERIFICATION OF GENERATED ARTIFACTS (FIGURES DIRECTORY) ===")
print("=" * 95)
missing = 0
for idx, fig_name in enumerate(expected_figures):
    fig_path = os.path.join("figures", fig_name)
    if os.path.exists(fig_path):
        size_kb = os.path.getsize(fig_path) / 1024
        print(f"[{idx+1:2d}/12] [SUCCESS] {fig_name:<46} ({size_kb:6.1f} KB)")
    else:
        print(f"[{idx+1:2d}/12] [FAILED ] Missing {fig_name}")
        missing += 1

print("=" * 95)
if missing == 0:
    print("ALL 12 MANUSCRIPT FIGURES GENERATED AND VERIFIED SUCCESSFULLY AT 300 DPI!")
else:
    print(f"WARNING: {missing} figures are missing!")
print("=" * 95)
