"""
Shared helper: repeat one configuration N times, report mean and std.
==========================================================================
Your advisor asked for every experiment to be repeated ~100 times, with
error bars (1 std dev) shown across those repeats. This is standard
Monte Carlo practice: a single simulation run is one sample from a
random process, and its result could land a little high or low by
chance. Running the SAME configuration 100 times and looking at the
spread tells you how much a single run's result could be trusted, and
lets you show that uncertainty honestly on every plot.

Use this instead of calling run_bb84_realistic() directly in any sweep.
"""

import numpy as np
from bb84_realistic import run_bb84_realistic
from experiment1_distance_sweep import secret_key_rate


def repeat_run(rng, n_bits, n_repeats=100, **params):
    """
    Run run_bb84_realistic() n_repeats times with the SAME parameters,
    letting the RNG advance naturally between repeats (so each repeat
    is an independent random sample, not a duplicate).

    Returns a dict with mean and std (1 std dev) of QBER and key rate
    across the n_repeats runs.
    """
    qbers = np.zeros(n_repeats)
    key_rates = np.zeros(n_repeats)

    for i in range(n_repeats):
        r = run_bb84_realistic(n_bits, rng=rng, **params)
        qbers[i] = r["qber"] if not np.isnan(r["qber"]) else 0.0
        key_rates[i] = secret_key_rate(r["sifting_rate"], r["qber"])

    return {
        "qber_mean": qbers.mean(),
        "qber_std": qbers.std(),   # ddof=0 (population std) - standard for reporting spread of observed runs
        "key_rate_mean": key_rates.mean(),
        "key_rate_std": key_rates.std(),
    }
