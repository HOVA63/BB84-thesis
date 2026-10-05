"""
Step 4 - Ideal BB84 Simulator
==============================
No channel loss, no noise, no eavesdropper yet. This is the baseline
you verify in Step 5 before adding realistic imperfections in Step 6+.
"""

import numpy as np


def run_bb84(n_bits, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, size=n_bits)
    alice_bases = rng.integers(0, 2, size=n_bits)

    bob_bases = rng.integers(0, 2, size=n_bits)

    same_basis = (alice_bases == bob_bases)

    bob_bits = np.where(
        same_basis,
        alice_bits,
        rng.integers(0, 2, size=n_bits)
    )

    sifted_alice = alice_bits[same_basis]
    sifted_bob = bob_bits[same_basis]

    if len(sifted_alice) > 0:
        errors = (sifted_alice != sifted_bob)
        qber = np.sum(errors) / len(sifted_alice)
    else:
        qber = float("nan")

    return {
        "alice_bits": alice_bits,
        "alice_bases": alice_bases,
        "bob_bases": bob_bases,
        "bob_bits": bob_bits,
        "sifting_mask": same_basis,
        "sifted_alice": sifted_alice,
        "sifted_bob": sifted_bob,
        "qber": qber,
        "sifting_rate": np.mean(same_basis),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 10_000

    result = run_bb84(n, rng=rng)

    print(f"Raw bits sent:       {n}")
    print(f"Sifted key length:   {len(result['sifted_alice'])}")
    print(f"Sifting rate:        {result['sifting_rate']:.3f}  (expect ~0.5)")
    print(f"QBER:                {result['qber']:.5f}  (expect ~0.0 for an ideal channel)")
