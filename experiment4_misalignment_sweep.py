"""
Experiment: QBER and secret-key rate vs. misalignment probability (isolated)
================================================================================
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

    misalignments = np.linspace(0.0, 0.15, 16)
    results = []

    for e_mis in misalignments:
        stats = repeat_run(
            rng, n, N_REPEATS,
            distance_km=0, detector_efficiency=1.0,
            dark_count_prob=0.0, misalignment_prob=e_mis,
        )
        results.append((e_mis, stats["qber_mean"], stats["qber_std"],
                         stats["key_rate_mean"], stats["key_rate_std"]))

    print(f"{'Misalign':>10} {'QBER_mean':>10} {'QBER_std':>10} {'KR_mean':>10} {'KR_std':>10}")
    for e_mis, qm, qs, krm, krs in results:
        print(f"{e_mis:>10.3f} {qm:>10.4f} {qs:>10.4f} {krm:>10.5f} {krs:>10.5f}")

    with open("experiment4_misalignment_sweep.csv", "w") as f:
        f.write("misalignment_prob,qber_mean,qber_std,key_rate_mean,key_rate_std\n")
        for e_mis, qm, qs, krm, krs in results:
            f.write(f"{e_mis},{qm},{qs},{krm},{krs}\n")

    x = [row[0] for row in results]
    y_qber_mean = [row[1] for row in results]
    y_qber_std = [row[2] for row in results]
    y_kr_mean = [row[3] for row in results]
    y_kr_std = [row[4] for row in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(x, y_qber_mean, yerr=y_qber_std, marker="o", color="crimson",
                 markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="lightcoral", label="QBER")
    ax1.set_xlabel("Misalignment probability")
    ax1.set_ylabel("QBER", color="crimson")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.errorbar(x, y_kr_mean, yerr=y_kr_std, marker="s", color="teal",
                 markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="paleturquoise", label="Key rate")
    ax2.set_ylabel("Secret-key rate (per pulse)", color="teal")

    plt.title(f"QBER and Key Rate vs. Misalignment Probability\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats per point)")
    plt.tight_layout()
    plt.savefig("experiment4_qber_keyrate_vs_misalignment.png", dpi=150)
    print("\nSaved: experiment4_misalignment_sweep.csv")
    print("Saved: experiment4_qber_keyrate_vs_misalignment.png")
