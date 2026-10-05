"""
Step 6 + Step 7 + Step 9 - Realistic BB84 Simulator, with optional Eve
========================================================================
Builds on the ideal simulator (Step 4/5). Adds channel loss, detector
efficiency, dark counts, and misalignment, combined into one function.
Also supports an optional intercept-resend eavesdropper (Step 9).
"""

import numpy as np


def run_bb84_realistic(
    n_bits,
    distance_km=0.0,          # 6A: channel loss
    alpha_db_per_km=0.2,      # typical fibre attenuation
    detector_efficiency=1.0,  # 6B: 1.0 = perfect detector
    dark_count_prob=0.0,      # 6C: false-click probability per pulse
    misalignment_prob=0.0,    # 6D: probability a "correct basis" bit still flips
    eve_present=False,        # Step 9: intercept-resend eavesdropper
    eve_intercept_prob=1.0,   # fraction of pulses Eve intercepts (1.0 = all of them)
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # --- Alice ---
    alice_bits = rng.integers(0, 2, size=n_bits)
    alice_bases = rng.integers(0, 2, size=n_bits)

    # --- Step 9: Eve (intercept-resend) ---
    # Eve measures the qubit in her OWN random basis, then resends a
    # fresh qubit prepared according to what SHE measured. If her
    # basis happens to match Alice's, she learns the bit perfectly and
    # resends it correctly - no disturbance introduced on that qubit.
    # If her basis is wrong, she gets a RANDOM outcome (measurement
    # disturbance, same idea as Bob's own measurement) and resends a
    # qubit in the WRONG basis - which is what introduces errors later,
    # even when Bob's basis matches Alice's original one.
    if eve_present:
        eve_intercepts = rng.random(n_bits) < eve_intercept_prob
        eve_bases = rng.integers(0, 2, size=n_bits)
        eve_measured_bits = np.where(
            eve_bases == alice_bases,
            alice_bits,
            rng.integers(0, 2, size=n_bits),
        )
        # What actually continues down the channel toward Bob:
        # Eve's resent qubit where she intercepted, Alice's original
        # qubit everywhere else.
        transmitted_bits = np.where(eve_intercepts, eve_measured_bits, alice_bits)
        transmitted_bases = np.where(eve_intercepts, eve_bases, alice_bases)
    else:
        transmitted_bits = alice_bits
        transmitted_bases = alice_bases

    # --- 6A: Channel loss ---
    transmittance = 10 ** (-alpha_db_per_km * distance_km / 10)
    photon_arrives = rng.random(n_bits) < transmittance

    # --- 6B: Detector efficiency ---
    real_detection = photon_arrives & (rng.random(n_bits) < detector_efficiency)

    # --- 6C: Dark counts ---
    dark_click = rng.random(n_bits) < dark_count_prob
    detected = real_detection | dark_click

    # --- Bob ---
    bob_bases = rng.integers(0, 2, size=n_bits)
    # IMPORTANT: Bob's bit correctness depends on matching the basis of
    # whatever qubit actually arrives (transmitted_bases - Eve's, if she
    # intercepted), NOT Alice's original basis directly.
    same_as_incoming = (bob_bases == transmitted_bases)

    bob_bits_if_detected = np.where(
        same_as_incoming,
        transmitted_bits,
        rng.integers(0, 2, size=n_bits),
    )

    # --- 6D: Misalignment ---
    misalignment_flip = (rng.random(n_bits) < misalignment_prob) & real_detection
    bob_bits_if_detected = np.where(
        misalignment_flip,
        1 - bob_bits_if_detected,
        bob_bits_if_detected,
    )

    bob_bits_if_detected = np.where(
        dark_click & ~real_detection,
        rng.integers(0, 2, size=n_bits),
        bob_bits_if_detected,
    )

    # --- Step 7: Sifting ---
    # Sifting is a PUBLIC step: Alice and Bob compare their announced
    # bases. Alice announces her ORIGINAL basis (she has no idea Eve
    # touched anything), so sifting still uses alice_bases, not
    # transmitted_bases.
    same_basis_public = (alice_bases == bob_bases)
    keep_mask = detected & same_basis_public
    sifted_alice = alice_bits[keep_mask]
    sifted_bob = bob_bits_if_detected[keep_mask]

    if len(sifted_alice) > 0:
        errors = (sifted_alice != sifted_bob)
        qber = np.sum(errors) / len(sifted_alice)
    else:
        qber = float("nan")

    return {
        "n_bits": n_bits,
        "n_detected": int(np.sum(detected)),
        "n_sifted": int(np.sum(keep_mask)),
        "sifted_alice": sifted_alice,
        "sifted_bob": sifted_bob,
        "qber": qber,
        "detection_rate": np.mean(detected),
        "sifting_rate": np.mean(keep_mask),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    n = 100_000

    print("=== Sanity check 1: distance=0, everything ideal -> should match Step 4/5 ===")
    r = run_bb84_realistic(n, distance_km=0, rng=rng)
    print(f"Sifting rate: {r['sifting_rate']:.3f} (expect ~0.5)")
    print(f"QBER:         {r['qber']:.5f} (expect ~0.0)\n")

    print("=== Sanity check 2: add realistic loss + detector + dark counts + misalignment ===")
    r = run_bb84_realistic(
        n,
        distance_km=50,
        alpha_db_per_km=0.2,
        detector_efficiency=0.8,
        dark_count_prob=1e-5,
        misalignment_prob=0.02,
        rng=rng,
    )
    print(f"Detection rate: {r['detection_rate']:.4f}")
    print(f"Sifted key len: {r['n_sifted']}")
    print(f"QBER:           {r['qber']:.4f} (should now be > 0, driven by misalignment)\n")

    print("=== Sanity check 3: ideal channel, but Eve fully intercepts everything ===")
    r = run_bb84_realistic(n, distance_km=0, eve_present=True, eve_intercept_prob=1.0, rng=rng)
    print(f"QBER: {r['qber']:.4f} (classic intercept-resend result: expect ~0.25)")
