"""
Sensitivity Ranking: No Eve vs. With Eve
============================================
UPDATED: every point in every sweep is now the MEAN of 100 independent
repeats, not a single run. This makes the % drop figures themselves
more stable, and lets us report the std dev of key rate at the best
and worst points of each sweep for context (see printed output).
"""

import numpy as np
from bb84_realistic import run_bb84_realistic
from experiment1_distance_sweep import secret_key_rate, repeat_run

N_REPEATS = 100

baseline = dict(
    distance_km=30, alpha_db_per_km=0.2,
    detector_efficiency=0.8, dark_count_prob=1e-5, misalignment_prob=0.02,
)

sweeps = {
    "distance_km": np.linspace(0, 100, 6),
    "detector_efficiency": np.linspace(0.5, 1.0, 6),
    "dark_count_prob": np.linspace(1e-6, 1e-4, 6),
    "misalignment_prob": np.linspace(0.0, 0.05, 6),
}


def run_sensitivity(eve_present, eve_intercept_prob=0.0, seed=42, n=100_000):
    rng = np.random.default_rng(seed=seed)  # fresh, identical seed every call
    results = {}
    detail = {}
    for param_name, values in sweeps.items():
        means, stds = [], []
        for v in values:
            params = baseline.copy()
            params[param_name] = v
            stats = repeat_run(
                rng, n, N_REPEATS,
                eve_present=eve_present, eve_intercept_prob=eve_intercept_prob,
                **params,
            )
            means.append(stats["key_rate_mean"])
            stds.append(stats["key_rate_std"])
        means = np.array(means)
        stds = np.array(stds)
        best_idx, worst_idx = means.argmax(), means.argmin()
        best, worst = means[best_idx], means[worst_idx]
        percent_drop = (best - worst) / best * 100 if best > 0 else float("nan")
        results[param_name] = percent_drop
        detail[param_name] = (best, stds[best_idx], worst, stds[worst_idx])
    return results, detail


if __name__ == "__main__":
    no_eve, no_eve_detail = run_sensitivity(eve_present=False)
    print(f"=== No Eve (baseline), n={N_REPEATS} repeats per point ===")
    no_eve_ranked = sorted(no_eve.items(), key=lambda kv: kv[1], reverse=True)
    for i, (name, pct) in enumerate(no_eve_ranked, 1):
        best, best_std, worst, worst_std = no_eve_detail[name]
        print(f"{i}. {name:<22}{pct:>6.1f}%   (best={best:.5f}\u00b1{best_std:.5f}, "
              f"worst={worst:.5f}\u00b1{worst_std:.5f})")
    order_no_eve = [name for name, _ in no_eve_ranked]

    all_with_eve = {}
    for EVE_INTERCEPT in (0.15, 0.20, 0.25):
        with_eve, with_eve_detail = run_sensitivity(eve_present=True, eve_intercept_prob=EVE_INTERCEPT)
        all_with_eve[EVE_INTERCEPT] = with_eve

        with_eve_ranked = sorted(with_eve.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n=== With Eve ({EVE_INTERCEPT:.0%} intercept) ===")
        for i, (name, pct) in enumerate(with_eve_ranked, 1):
            best, best_std, worst, worst_std = with_eve_detail[name]
            print(f"{i}. {name:<22}{pct:>6.1f}%   (best={best:.5f}\u00b1{best_std:.5f}, "
                  f"worst={worst:.5f}\u00b1{worst_std:.5f})")

        order_with_eve = [name for name, _ in with_eve_ranked]
        print(f"Rank order changed from no-Eve baseline: {order_no_eve != order_with_eve}")

    print(f"\n=== Trend across intercept levels (all parameters, % drop) ===")
    print(f"{'Parameter':<22}{'0%':>8}{'15%':>8}{'20%':>8}{'25%':>8}")
    for name in sweeps:
        print(f"{name:<22}{no_eve[name]:>7.1f}%{all_with_eve[0.15][name]:>7.1f}%"
              f"{all_with_eve[0.20][name]:>7.1f}%{all_with_eve[0.25][name]:>7.1f}%")

    with open("experiment6_sensitivity_with_without_eve.csv", "w") as f:
        f.write("parameter,pct_0,pct_15,pct_20,pct_25\n")
        for name in sweeps:
            f.write(f"{name},{no_eve[name]},{all_with_eve[0.15][name]},"
                    f"{all_with_eve[0.20][name]},{all_with_eve[0.25][name]}\n")
    print("\nSaved: experiment6_sensitivity_with_without_eve.csv")
