"""
Generate Figure 1 (fig1.pdf) from the simulator summary.json output.

The figure has three panels:
  left:   blocked/full output agreement (non-overflowing runs)
  middle: per-parameter pair recall
  right:  per-parameter overflow rate

Missing (overflow=1) values for the (16,6) row are rendered as line breaks
in the left and middle panels and as 1.00 in the right panel.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(summary_path: str, out_pdf: str) -> None:
    s = json.loads(Path(summary_path).read_text())
    xi = s["table_xi"]
    xii = s["table_xii"]

    settings = [f"({r['b']},{r['r']})" for r in xi]
    ks_med = [r["k_pair_median"] for r in xi]
    overflow = [r["overflow_rate"] for r in xi]
    agreement = [r["output_agreement"] for r in xi]
    pair_recall = [r["pair_recall"] for r in xii]
    bench_recall = [r["bench_item_recall"] for r in xii]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --- left: output agreement (coarse summary) ---
    ax = axes[0]
    x = np.arange(len(settings))
    agree_plot = [a if not np.isnan(a) else None for a in agreement]
    yvals = [v if v is not None else np.nan for v in agree_plot]
    ax.plot(x[:3], yvals[:3], "o-", color="#1f77b4", label="output agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlabel("(b, r)")
    ax.set_ylabel("output agreement")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Blocked / full output agreement")
    ax.grid(True, linestyle=":", alpha=0.5)
    # line break for (16,6) -- drawn as gap in x-axis at position 3
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.0, ymin=0, ymax=1)
    ax.text(3.0, 0.5, "n/a\n(16,6)\naborts", ha="center", va="center", fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5))

    # --- middle: pair recall vs benchmark-item recall ---
    ax = axes[1]
    pr = [v if v is not None else np.nan for v in pair_recall]
    br = [v if v is not None else np.nan for v in bench_recall]
    ax.plot(x[:3], pr[:3], "o-", color="#2ca02c", label="pair recall")
    ax.plot(x[:3], br[:3], "s-", color="#d62728", label="benchmark-item recall")
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlabel("(b, r)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall on non-overflowing trials")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.0, ymin=0, ymax=1)
    ax.text(3.0, 0.5, "n/a\n(16,6)\naborts", ha="center", va="center", fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5))

    # --- right: overflow rate ---
    ax = axes[2]
    ax.bar(x, overflow, color=["#1f77b4", "#1f77b4", "#1f77b4", "#d62728"])
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlabel("(b, r)")
    ax.set_ylabel("overflow rate")
    ax.set_ylim(0.0, 1.10)
    ax.set_title("Overflow rate (chi_K = 1)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for i, v in enumerate(overflow):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(
        "TL-FPSI synthetic reference simulator: n=800, m=240, s=48, tau=8, "
        "K_max=12000, 100 trials per setting, q_star=0.9",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    summary = sys.argv[1] if len(sys.argv) > 1 else "summary.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "fig1.pdf"
    main(summary, out)
