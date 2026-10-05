"""
Experiment: QBER and secret-key rate vs. dark-count probability (isolated)
=============================================================================
UPDATED: 100 repeats per point, mean +/- 1 std dev error bars.
Baseline distance of 30km retained - dark counts need a loss window
to be visible (see original comment: at distance=0 with perfect
detection, every pulse is already a real detection, so a dark count
adds nothing).
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

    dark_probs = np.logspace(-7, -3, 17)
    results = []

    for p_dc in dark_probs:
        stats = repeat_run(
            rng, n, N_REPEATS,
            distance_km=30, alpha_db_per_km=0.2,
            detector_efficiency=1.0, dark_count_prob=p_dc, misalignment_prob=0.0,
        )
        results.append((p_dc, stats["qber_mean"], stats["qber_std"],
                         stats["key_rate_mean"], stats["key_rate_std"]))

    print(f"{'DarkProb':>12} {'QBER_mean':>10} {'QBER_std':>10} {'KR_mean':>10} {'KR_std':>10}")
    for p_dc, qm, qs, krm, krs in results:
        print(f"{p_dc:>12.2e} {qm:>10.4f} {qs:>10.4f} {krm:>10.5f} {krs:>10.5f}")

    with open("experiment3_darkcount_sweep.csv", "w") as f:
        f.write("dark_count_prob,qber_mean,qber_std,key_rate_mean,key_rate_std\n")
        for p_dc, qm, qs, krm, krs in results:
            f.write(f"{p_dc},{qm},{qs},{krm},{krs}\n")

    x = [row[0] for row in results]
    y_qber_mean = [row[1] for row in results]
    y_qber_std = [row[2] for row in results]
    y_kr_mean = [row[3] for row in results]
    y_kr_std = [row[4] for row in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(x, y_qber_mean, yerr=y_qber_std, marker="o", color="crimson",
                 markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="lightcoral", label="QBER")
    ax1.set_xscale("log")
    ax1.set_xlabel("Dark-count probability (log scale)")
    ax1.set_ylabel("QBER", color="crimson")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.errorbar(x, y_kr_mean, yerr=y_kr_std, marker="s", color="teal",
                 markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="paleturquoise", label="Key rate")
    ax2.set_ylabel("Secret-key rate (per pulse)", color="teal")

    plt.title(f"QBER and Key Rate vs. Dark-Count Probability\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats per point)")
    plt.tight_layout()
    plt.savefig("experiment3_qber_keyrate_vs_darkcount.png", dpi=150)
    print("\nSaved: experiment3_darkcount_sweep.csv")
    print("Saved: experiment3_qber_keyrate_vs_darkcount.png")
