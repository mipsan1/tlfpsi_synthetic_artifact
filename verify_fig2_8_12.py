"""
Verify the Fig 2 (b,r)=(8,12) b<=tau SimHash regime results with a fresh
seed sweep and a higher R_per_setting.

The current make_fig2.py uses R=50 trials per setting with seed
SEED_BASE + 7777 + b*100 + n. The (b,r)=(8,12) row in the paper claims
that with q_star=0.9, r=12 the per-audit candidate probability is
about 0.92 and the target-survival rate across T_max=16 audits is
in the 0.10-0.20 range. We re-validate this claim with a fresh seed
sweep at R=200 to tighten the estimate and check for outliers.
"""
from __future__ import annotations

import sys
import statistics
from pathlib import Path

import numpy as np

# Re-use the simulator / make_fig2 internals
sys.path.insert(0, str(Path(__file__).parent))

from simulator import (
    N_CORPUS, M_BENCH, TAU, Q_STAR, hamming_le,
    make_hyperplanes, simhash_per_band, pack_band_keys,
)
import simulator as sim
from make_fig2 import run_repeated_audit


def fresh_seed_sweep(b: int, r: int, d: int, T_max: int, R: int,
                     target_counts: list[int]) -> list[dict]:
    """Run a fresh seed sweep for a single regime.

    For each (n_targets, rep_idx) we run R_per_setting=1 audit (single
    trial). Across `R` such independent trials we compute the mean and
    standard deviation of survival and confusion.
    """
    out = []
    for n in target_counts:
        survival_list = []
        confusion_list = []
        for rep in range(R):
            seed = sim.SEED_BASE * 31 + b * 10000 + n * 100 + rep
            rng = np.random.default_rng(seed)
            s, c = run_repeated_audit(rng, n, T_max, 1, b=b, r=r, d=d)
            survival_list.append(s)
            confusion_list.append(c)
        out.append({
            "b": b, "r": r, "n_targets": n, "R": R,
            "survival_mean": statistics.fmean(survival_list),
            "survival_stdev": statistics.pstdev(survival_list) if len(survival_list) > 1 else 0.0,
            "confusion_mean": statistics.fmean(confusion_list),
            "confusion_stdev": statistics.pstdev(confusion_list) if len(confusion_list) > 1 else 0.0,
        })
    return out


def full_sweep(b: int, r: int, d: int, T_max: int, R: int,
               target_counts: list[int]) -> list[dict]:
    """Run a more conservative sweep: each (n_targets, rep_idx) uses
    R_per_setting=200 trials (the production setting is R=50, this is 4x).
    Total trials = 4 * R.
    """
    out = []
    for n in target_counts:
        survival_list = []
        confusion_list = []
        for rep in range(R):
            seed = sim.SEED_BASE * 31 + b * 10000 + n * 100 + rep
            rng = np.random.default_rng(seed)
            s, c = run_repeated_audit(rng, n, T_max, 200, b=b, r=r, d=d)
            survival_list.append(s)
            confusion_list.append(c)
        out.append({
            "b": b, "r": r, "n_targets": n, "R": R,
            "R_per_setting": 200,
            "survival_mean": statistics.fmean(survival_list),
            "survival_stdev": statistics.pstdev(survival_list) if len(survival_list) > 1 else 0.0,
            "confusion_mean": statistics.fmean(confusion_list),
            "confusion_stdev": statistics.pstdev(confusion_list) if len(confusion_list) > 1 else 0.0,
        })
    return out


def main() -> None:
    T_max = 16
    R_fresh = 200   # 4x the production R=50
    target_counts = [8, 16, 32, 64]

    print("=== Fresh seed sweep for Fig 2 (b,r)=(8,12) b=tau SimHash regime ===")
    print(f"T_max={T_max}, R_per_setting=1 trial, R_seeds={R_fresh}, q_star={Q_STAR}")
    print()
    rows = fresh_seed_sweep(b=8, r=12, d=96, T_max=T_max, R=R_fresh,
                            target_counts=target_counts)
    print(f"{'n_targets':>10}  {'survival_mean':>15}  {'survival_stdev':>15}  {'confusion_mean':>15}  {'confusion_stdev':>15}")
    for row in rows:
        print(f"{row['n_targets']:>10}  {row['survival_mean']:>15.4f}  "
              f"{row['survival_stdev']:>15.4f}  {row['confusion_mean']:>15.4f}  "
              f"{row['confusion_stdev']:>15.4f}")

    print()
    print("=== Full sweep: R=200 trials per setting, 20 fresh seeds ===")
    print(f"T_max={T_max}, R_per_setting=200, R_seeds=20, q_star={Q_STAR}")
    print("Total trials per n_targets = 20 * 200 = 4000")
    print()
    full_rows = full_sweep(b=8, r=12, d=96, T_max=T_max, R=20,
                           target_counts=target_counts)
    print(f"{'n_targets':>10}  {'survival_mean':>15}  {'survival_stdev':>15}  {'confusion_mean':>15}  {'confusion_stdev':>15}")
    for row in full_rows:
        print(f"{row['n_targets']:>10}  {row['survival_mean']:>15.4f}  "
              f"{row['survival_stdev']:>15.4f}  {row['confusion_mean']:>15.4f}  "
              f"{row['confusion_stdev']:>15.4f}")

    # Compare with the production Fig 2 generator
    print()
    print("=== Production Fig 2 (8,12) results (R=50, original seed) ===")
    for n in target_counts:
        rng = np.random.default_rng(sim.SEED_BASE + 7777 + 8 * 100 + n)
        s, c = run_repeated_audit(rng, n, T_max, 50, b=8, r=12, d=96)
        print(f"n_targets={n}: survival={s:.4f} confusion={c:.4f}")

    # Per-band expected candidate probability (analytical)
    q = Q_STAR
    r_band = 12
    b_band = 8
    p_band = q ** r_band
    p_audit = 1 - (1 - p_band) ** b_band
    p_T = p_audit ** T_max
    print()
    print(f"=== Analytical sanity check ===")
    print(f"per-band agreement q^r = {p_band:.4f}")
    print(f"per-audit candidate probability 1-(1-q^r)^b = {p_audit:.4f}")
    print(f"expected survival after T_max=16 audits = (p_audit)^T_max = {p_T:.4f}")


if __name__ == "__main__":
    main()
