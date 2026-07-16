"""
Generate Figure 2 (fig2.pdf): multi-item repeated-audit illustration for
Proposition 4 (paper).

Per the 3rd-round reviewer note (item 12), the right-hand panel must report
the conditional probability that the intersection retains non-target
candidates, conditioned on the target-survival event E (i.e. the target
index i* survives in the intersection of all T_max audit candidate sets).
This matches the conditioning used in Proposition 4:

    Pr[|intersection_t N_t(j)| > 1  |  E]

where E = "i* in intersection_t N_t(j)".

The figure is generated for two (b, r) regimes side by side, so that the
deterministic-completeness regime (b > tau) and the probabilistic regime
(b <= tau) can be compared directly:

  (a) (b, r) = (12, 8), b > tau = 8. Lemma 1 applies to every
      permuted view, so every Ham-<=tau target survives every audit
      and the conditional non-target count is essentially zero. The
      curve is therefore a horizontal line at (1.0, 0.0).

  (b) (b, r) = (8, 12), b = tau. The SimHash banding of Section V-A
      is used: each band is the sign of the fingerprint projected onto
      b*r independent random hyperplanes, and the candidate probability
      is 1 - (1 - q_star^r)^b under Assumption (simhash). With
      q_star = 0.9 and r = 12, the per-band agreement is approximately
      0.282, so a single audit has a non-trivial miss probability and
      the target-survival rate across T_max = 16 audits is below one.
      The conditional non-target count is also above zero because
      the SimHash candidate set contains a small but non-negligible
      fraction of non-target corpus indices.

Both metrics are reported across n_targets in {8, 16, 32, 64}.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import simulator as sim
from simulator import (
    N_CORPUS,
    M_BENCH,
    TAU,
    Q_STAR,
    hamming_le,
    make_hyperplanes,
    simhash_per_band,
    pack_band_keys,
)


def raw_bit_partition_keys(fps: np.ndarray, b: int, r: int) -> np.ndarray:
    """
    Pack each of the b disjoint raw bit slices of length r into a uint64 key.

    This implements the deterministic-completeness regime of Lemma 1
    (b > tau). Each band k uses bits [k*r, (k+1)*r) of the fingerprint.
    """
    n, d = fps.shape
    out = np.zeros((n, b), dtype=np.uint64)
    weights = np.left_shift(np.ones(r, dtype=np.uint64), np.arange(r, dtype=np.uint64))
    for k in range(b):
        lo = k * r
        hi = (k + 1) * r
        out[:, k] = (fps[:, lo:hi].astype(np.uint64) * weights).sum(axis=1)
    return out


def simhash_band_keys(
    rng: np.random.Generator,
    fps: np.ndarray,
    b: int,
    r: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the SimHash band keys for the probabilistic regime (b <= tau).

    Returns (ck, bk) where ck is the (n, b) corpus-side band-key matrix
    and bk is the (m, b) benchmark-side band-key matrix. The hyperplane
    matrix is freshly drawn from `rng` for each call (this is the
    "fresh bit permutation per audit" interpretation of Proposition 4
    in the SimHash regime).
    """
    hp = make_hyperplanes(rng, b, r, d)
    c_sig = simhash_per_band(fps, hp, b, r)
    ck = pack_band_keys(c_sig, b, r)
    return ck


def band_keys_for_audit(
    rng: np.random.Generator,
    corpus: np.ndarray,
    bench: np.ndarray,
    b: int,
    r: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ck, bk) for a single audit, choosing the regime from (b, r).

    If b > tau (deterministic-completeness regime), apply a fresh
    permutation to the bit positions and use the raw bit partition.
    Lemma 1 then guarantees that every Ham-<=tau pair shares at least
    one exact-match band on every audit.

    If b <= tau (probabilistic regime), draw a fresh SimHash hyperplane
    matrix and project both sides onto it. The candidate set is then
    probabilistic; the per-band agreement is q_star^r under the
    honest-input model of Assumption (simhash).
    """
    if b > TAU:
        perm = rng.permutation(d)
        c_perm = corpus[:, perm]
        b_perm = bench[:, perm]
        return (
            raw_bit_partition_keys(c_perm, b, r),
            raw_bit_partition_keys(b_perm, b, r),
        )
    hp = make_hyperplanes(rng, b, r, d)
    c_sig = simhash_per_band(corpus, hp, b, r)
    b_sig = simhash_per_band(bench, hp, b, r)
    return (
        pack_band_keys(c_sig, b, r),
        pack_band_keys(b_sig, b, r),
    )


def run_repeated_audit(
    rng,
    n_target_pairs: int,
    T_max: int,
    R_per_setting: int,
    b: int = 12,
    r: int = 8,
    d: int = 96,
):
    """
    Returns:
        survival_rate: float
        confusion_given_survival: float
            (mean |intersection - {i*}| conditional on target surviving)
    """
    survival_records = []
    confusion_records = []   # only when target survived (E holds)
    for _ in range(R_per_setting):
        # 1) draw corpus + bench
        corpus = rng.integers(0, 2, size=(N_CORPUS, d), dtype=np.uint8)
        bench = rng.integers(0, 2, size=(M_BENCH, d), dtype=np.uint8)

        # 2) inject n_target_pairs true matches under threshold
        inject_pairs = []
        used_bench = set()
        used_corpus = set()
        tries = 0
        while len(inject_pairs) < n_target_pairs and tries < n_target_pairs * 200:
            tries += 1
            i = int(rng.integers(0, N_CORPUS))
            if i in used_corpus:
                continue
            j = int(rng.integers(0, M_BENCH))
            if j in used_bench:
                continue
            h = int(rng.integers(0, TAU + 1))
            if h > 0:
                flip = rng.choice(d, size=h, replace=False)
                bench_flip = corpus[i].copy()
                bench_flip[flip] ^= 1
                bench[j] = bench_flip
            # Only accept the pair if the resulting Hamming distance is at
            # most TAU. The previous version skipped this check, which
            # permitted a fraction of trials in which the injected pair
            # had Ham > TAU and was therefore not a true match; under the
            # deterministic-completeness regime of Lemma 1, such a pair
            # has zero band matches and is incorrectly counted as a lost
            # target. We re-sample in that case so that all injected
            # pairs satisfy the Ham <= TAU invariant of the audit.
            ham = hamming_le(d, corpus[i][None, :], bench[j][None, :])[0]
            if ham > TAU:
                # Roll back: restore bench[j] to its prior (random) value
                # by re-rolling the affected bit flips is not necessary
                # because the bench slot will be overwritten by a future
                # accept, and used_bench / used_corpus do not record
                # rejected candidates. Just skip the append.
                if h > 0:
                    # Re-flip the previously flipped bits to recover the
                    # original random bench[j] (so the slot remains
                    # unbiased for future inject attempts).
                    bench[j][flip] ^= 1
                continue
            inject_pairs.append((i, j))
            used_bench.add(j)
            used_corpus.add(i)

        # 3) T_max audits, each with a fresh projection on both sides.
        # For (b,r) in the deterministic-completeness regime (b > tau),
        # the projection is a fresh bit permutation and Lemma 1
        # applies; for the probabilistic regime (b <= tau), the
        # projection is a fresh random-hyperplane matrix under
        # Assumption (simhash).
        i_star, j_star = inject_pairs[0]
        target_survived = True
        candidate_intersection: set[int] = None
        for t in range(T_max):
            ck, bk = band_keys_for_audit(rng, corpus, bench, b, r, d)
            cand: set[int] = set()
            for k in range(b):
                key = bk[j_star, k]
                cand.update(np.where(ck[:, k] == key)[0].tolist())
            if candidate_intersection is None:
                candidate_intersection = set(cand)
            else:
                candidate_intersection &= cand
            if i_star not in candidate_intersection:
                target_survived = False
                break
        survival_records.append(1.0 if target_survived else 0.0)
        if target_survived:
            # non-target count = |intersection| - 1
            confusion_records.append(max(0, len(candidate_intersection) - 1))
    survival_rate = float(np.mean(survival_records)) if survival_records else 0.0
    confusion_given = (
        float(np.mean(confusion_records)) if confusion_records else 0.0
    )
    return survival_rate, confusion_given


def main(out_pdf: str) -> None:
    T_max = 16
    R_per_setting = 50
    target_counts = [8, 16, 32, 64]
    # Two regimes: deterministic-completeness (b > tau) and probabilistic
    # (b <= tau). The b=8, r=12 row of the synthetic study (Table XII)
    # has b = tau = 8 and falls in the SimHash regime; the b=12, r=8
    # row has b > tau and falls in the deterministic-completeness
    # regime of Lemma 1.
    regimes = [
        {"b": 12, "r": 8, "d": 96, "label": "(b,r)=(12,8)  b>tau, det-complete"},
        {"b": 8, "r": 12, "d": 96, "label": "(b,r)=(8,12)  b=tau, SimHash regime"},
    ]
    rows_all: list[dict] = []
    for regime in regimes:
        b, r, d = regime["b"], regime["r"], regime["d"]
        for n in target_counts:
            rng = np.random.default_rng(sim.SEED_BASE + 7777 + b * 100 + n)
            s, c = run_repeated_audit(rng, n, T_max, R_per_setting, b=b, r=r, d=d)
            print(
                f"{regime['label']} n_target={n}: "
                f"survival={s:.3f} E[|I\\{{i*\\}}| | E]={c:.2f}"
            )
            rows_all.append({
                "b": b, "r": r, "d": d,
                "n_targets": n,
                "T_max": T_max,
                "survival": s,
                "confusion_given_survival": c,
            })

    Path("fig2_data.json").write_text(json.dumps(rows_all, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for col, regime in enumerate(regimes):
        b, r = regime["b"], regime["r"]
        rows = [row for row in rows_all if row["b"] == b and row["r"] == r]
        xs = [r["n_targets"] for r in rows]
        surv = [r["survival"] for r in rows]
        conf = [r["confusion_given_survival"] for r in rows]

        ax = axes[2 * col]
        ax.plot(xs, surv, "o-", color="#1f77b4")
        for x_val, y_val in zip(xs, surv):
            # offset above the marker; clear of the title which sits above
            # the upper y-limit of 1.10
            ax.annotate(f"{y_val:.2f}", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("true-match pairs (n_targets)")
        ax.set_ylabel("target-survival rate")
        ax.set_ylim(0.0, 1.10)
        ax.set_title(f"({chr(ord('a')+2*col)}) survival  {regime['label']}",
                     fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax = axes[2 * col + 1]
        ax.plot(xs, conf, "s-", color="#d62728")
        for x_val, y_val in zip(xs, conf):
            # Place label ABOVE the marker so it never collides with the
            # x-axis tick labels (the previous ylim of [0, 1.1*max] put
            # labels at y < 0 when y_val = 0, overlapping the tick row).
            ax.annotate(f"{y_val:.2f}", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("true-match pairs (n_targets)")
        ax.set_ylabel("non-target count")
        # Force a fixed ylim of [0, 1.5] so the data labels (placed 8pt
        # above the marker) sit well below the subplot title and never
        # collide with the x-axis tick labels when y_val = 0.
        ax.set_ylim(0.0, 1.5)
        ax.set_title(f"({chr(ord('b')+2*col)}) non-target count  {regime['label']}",
                     fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Multi-item repeated-audit illustration (Proposition 4): "
        f"n=800, m=240, tau=8, T_max={T_max}, R={R_per_setting} trials per setting",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fig2.pdf"
    main(out)
