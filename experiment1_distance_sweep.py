"""
Experiment: Secret-key rate vs. distance (channel loss only)
===============================================================
Detector efficiency, dark counts, and misalignment are held at their
IDEAL values here (1.0, 0, 0) so this experiment isolates ONE effect:
channel loss.

UPDATED: each distance is now run N_REPEATS=100 times independently.
Mean and 1 std dev are reported and plotted as error bars, per
advisor's request. A single run is one random sample; repeating it
100 times shows how much that single number could vary by chance.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bb84_realistic import run_bb84_realistic


def binary_entropy(q):
    """h2(Q) - used for the simplified secret-fraction key-rate model."""
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def secret_key_rate(sifted_rate, qber):
    """R_key ~= R_sifted * (1 - 2*h2(Q))  [simplified BB84 model]"""
    if np.isnan(qber):
        return 0.0
    r = 1 - 2 * binary_entropy(qber)
    r = max(r, 0.0)
    return sifted_rate * r


def repeat_run(rng, n_bits, n_repeats, **params):
    """Run the same configuration n_repeats times; return mean/std."""
    qbers = np.zeros(n_repeats)
    key_rates = np.zeros(n_repeats)
    for i in range(n_repeats):
        r = run_bb84_realistic(n_bits, rng=rng, **params)
        qbers[i] = r["qber"] if not np.isnan(r["qber"]) else 0.0
        key_rates[i] = secret_key_rate(r["sifting_rate"], r["qber"])
    return {
        "qber_mean": qbers.mean(), "qber_std": qbers.std(),
        "key_rate_mean": key_rates.mean(), "key_rate_std": key_rates.std(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 100_000
    N_REPEATS = 100

    distances = np.linspace(0, 100, 21)
    results = []

    for L in distances:
        stats = repeat_run(
            rng, n, N_REPEATS,
            distance_km=L, alpha_db_per_km=0.2,
            detector_efficiency=1.0, dark_count_prob=0.0, misalignment_prob=0.0,
        )
        results.append((L, stats["qber_mean"], stats["qber_std"],
                         stats["key_rate_mean"], stats["key_rate_std"]))

    print(f"{'Distance(km)':>12} {'QBER_mean':>10} {'QBER_std':>10} {'KR_mean':>10} {'KR_std':>10}")
    for L, qm, qs, krm, krs in results:
        print(f"{L:>12.1f} {qm:>10.4f} {qs:>10.4f} {krm:>10.5f} {krs:>10.5f}")

    with open("experiment1_distance_sweep.csv", "w") as f:
        f.write("distance_km,qber_mean,qber_std,key_rate_mean,key_rate_std\n")
        for L, qm, qs, krm, krs in results:
            f.write(f"{L},{qm},{qs},{krm},{krs}\n")

    distances_arr = [row[0] for row in results]
    kr_mean_arr = [row[3] for row in results]
    kr_std_arr = [row[4] for row in results]

    plt.figure(figsize=(7, 5))
    plt.errorbar(distances_arr, kr_mean_arr, yerr=kr_std_arr,
                 marker="o", markersize=4, markerfacecolor="white",
                 capsize=5, elinewidth=1.5, ecolor="gray")
    plt.xlabel("Distance (km)")
    plt.ylabel("Secret-key rate (per pulse)")
    plt.title(f"BB84 Secret-Key Rate vs. Distance\n(mean \u00b1 1 std dev, n={N_REPEATS} repeats per point)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment1_keyrate_vs_distance.png", dpi=150)
    print("\nSaved: experiment1_distance_sweep.csv")
    print("Saved: experiment1_keyrate_vs_distance.png")
