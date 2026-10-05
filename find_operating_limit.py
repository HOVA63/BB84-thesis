"""
Operating "Limit": distance at which finite-key effects make the
asymptotic model statistically unreliable
==========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb84_realistic import run_bb84_realistic
from experiment1_distance_sweep import secret_key_rate

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 200_000
    N_REPEATS = 100
    MIN_SIFTED = 1000  # minimum sifted bits needed to trust a QBER estimate

    hardware = dict(
        alpha_db_per_km=0.2, detector_efficiency=0.8,
        dark_count_prob=1e-5, misalignment_prob=0.02,
    )

    distances = np.linspace(0, 400, 41)

    print(f"{'Distance(km)':>12} {'MeanSifted':>12} {'KeyRate_mean':>14}")
    L_break = None
    rows = []
    for L in distances:
        sifted_counts = []
        key_rates = []
        for _ in range(N_REPEATS):
            r = run_bb84_realistic(n, distance_km=L, rng=rng, **hardware)
            sifted_counts.append(r["n_sifted"])
            key_rates.append(secret_key_rate(r["sifting_rate"], r["qber"]))
        mean_sifted = np.mean(sifted_counts)
        mean_kr = np.mean(key_rates)
        rows.append((L, mean_sifted, mean_kr))
        print(f"{L:>12.1f} {mean_sifted:>12.1f} {mean_kr:>14.7f}")
        if mean_sifted < MIN_SIFTED and L_break is None:
            L_break = L

    with open("operating_limit_distance_sweep.csv", "w") as f:
        f.write("distance_km,mean_sifted,mean_key_rate\n")
        for L, ms, kr in rows:
            f.write(f"{L},{ms},{kr}\n")
    print("\nSaved: operating_limit_distance_sweep.csv")

    # Plotting Figure 4.5
    dists = [r[0] for r in rows]
    sifted = [r[1] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.semilogy(dists, sifted, marker="o", color="navy", label="Mean sifted bits")
    plt.axhline(MIN_SIFTED, color="crimson", linestyle="--", label=f"Reliability threshold ({MIN_SIFTED} bits)")
    
    if L_break is not None:
        plt.axvline(L_break, color="gray", linestyle=":", alpha=0.7, label=f"Breakdown (~{L_break:.0f} km)")

    plt.xlabel("Distance (km)")
    plt.ylabel("Mean sifted bits (log scale)")
    plt.title(f"Mean Sifted Sample Size vs. Distance\n(mean across {N_REPEATS} repeats, N={n:,} sent/repeat)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("operating_limit_sifted_vs_distance.png", dpi=150)
    print("Saved: operating_limit_sifted_vs_distance.png")

    if L_break is not None:
        print(f"\nMean sifted sample size drops below {MIN_SIFTED} bits at approximately {L_break:.0f} km.")