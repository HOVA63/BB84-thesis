"""
Experiment: Secret-key rate vs. detector efficiency (isolated)
==================================================================
UPDATED: 100 repeats per point, mean +/- 1 std dev error bars.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb84_realistic import run_bb84_realistic
from experiment1_distance_sweep import secret_key_rate, repeat_run

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 100_000
    N_REPEATS = 100

    efficiencies = np.linspace(0.1, 1.0, 19)
    results = []

    for eta in efficiencies:
        stats = repeat_run(
            rng, n, N_REPEATS,
            distance_km=0, detector_efficiency=eta,
            dark_count_prob=0.0, misalignment_prob=0.0,
        )
        results.append((eta, stats["qber_mean"], stats["qber_std"],
                         stats["key_rate_mean"], stats["key_rate_std"]))

    print(f"{'Efficiency':>10} {'QBER_mean':>10} {'QBER_std':>10} {'KR_mean':>10} {'KR_std':>10}")
    for eta, qm, qs, krm, krs in results:
        print(f"{eta:>10.2f} {qm:>10.4f} {qs:>10.4f} {krm:>10.5f} {krs:>10.5f}")

    with open("experiment2_efficiency_sweep.csv", "w") as f:
        f.write("efficiency,qber_mean,qber_std,key_rate_mean,key_rate_std\n")
        for eta, qm, qs, krm, krs in results:
            f.write(f"{eta},{qm},{qs},{krm},{krs}\n")

    x = [row[0] for row in results]
    y_mean = [row[3] for row in results]
    y_std = [row[4] for row in results]

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y_mean, yerr=y_std, marker="o", color="darkorange",
                 markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="gray")
    plt.xlabel("Detector efficiency")
    plt.ylabel("Secret-key rate (per pulse)")
    plt.title(f"BB84 Secret-Key Rate vs. Detector Efficiency\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats per point)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment2_keyrate_vs_efficiency.png", dpi=150)
    print("\nSaved: experiment2_efficiency_sweep.csv")
    print("Saved: experiment2_keyrate_vs_efficiency.png")
