"""
Experiment: With vs. Without Eve (intercept-resend)
=======================================================
UPDATED: each scenario now run 100 times independently, reporting
mean +/- 1 std dev instead of a single run per scenario.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb84_realistic import run_bb84_realistic
from experiment1_distance_sweep import secret_key_rate, repeat_run

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 200_000
    N_REPEATS = 100

    hardware = dict(
        distance_km=30, alpha_db_per_km=0.2,
        detector_efficiency=0.8, dark_count_prob=1e-5, misalignment_prob=0.02,
    )

    scenarios = [
        ("No Eve", dict(eve_present=False)),
        ("Eve, full intercept (100%)", dict(eve_present=True, eve_intercept_prob=1.0)),
        ("Eve, partial intercept (50%)", dict(eve_present=True, eve_intercept_prob=0.5)),
        ("Eve, partial intercept (10%)", dict(eve_present=True, eve_intercept_prob=0.1)),
    ]

    print(f"{'Scenario':<30}{'QBER (mean\u00b1std)':>22}{'KeyRate (mean\u00b1std)':>24}")

    labels, qber_means, qber_stds, kr_means, kr_stds = [], [], [], [], []

    for label, eve_params in scenarios:
        stats = repeat_run(rng, n, N_REPEATS, **hardware, **eve_params)
        print(f"{label:<30}{stats['qber_mean']:>10.4f}\u00b1{stats['qber_std']:<9.4f}"
              f"{stats['key_rate_mean']:>10.5f}\u00b1{stats['key_rate_std']:<11.5f}")
        labels.append(label)
        qber_means.append(stats["qber_mean"])
        qber_stds.append(stats["qber_std"])
        kr_means.append(stats["key_rate_mean"])
        kr_stds.append(stats["key_rate_std"])

    with open("experiment5_eavesdropper_comparison.csv", "w") as f:
        f.write("scenario,qber_mean,qber_std,key_rate_mean,key_rate_std\n")
        for label, qm, qs, krm, krs in zip(labels, qber_means, qber_stds, kr_means, kr_stds):
            f.write(f"{label},{qm},{qs},{krm},{krs}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(labels, qber_means, yerr=qber_stds, capsize=6, color="crimson",
            error_kw=dict(elinewidth=1.5, capthick=1.5))
    ax1.set_ylabel("QBER")
    ax1.set_title(f"QBER: With vs. Without Eve\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats)")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, axis="y")

    ax2.bar(labels, kr_means, yerr=kr_stds, capsize=6, color="teal",
            error_kw=dict(elinewidth=1.5, capthick=1.5))
    ax2.set_ylabel("Secret-key rate (per pulse)")
    ax2.set_title(f"Key Rate: With vs. Without Eve\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats)")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("experiment5_eavesdropper_comparison.png", dpi=150)
    print("\nSaved: experiment5_eavesdropper_comparison.csv")
    print("Saved: experiment5_eavesdropper_comparison.png")
