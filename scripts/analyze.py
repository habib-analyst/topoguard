"""Comprehensive analysis of the TopoGuard evaluation matrix.

Tests the pre-registered hypotheses (HYPOTHESES.md) and generates
paper-ready findings. Run after all phases complete:
  python scripts/analyze.py --evaldir eval --out results/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load(evaldir):
    recs = []
    for path in sorted(glob.glob(os.path.join(evaldir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    return [r for r in recs if "error" not in r]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", default="eval")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = load(args.evaldir)
    main = [r for r in recs if r["cell"].get("phase") == "main"]
    print(f"main matrix cells: {len(main)}")
    if not main:
        print("no main results yet")
        return

    out = ["# TopoGuard hypothesis tests (pre-registered)\n"]

    # H1: method ranking overall
    by_method = defaultdict(list)
    for r in main:
        by_method[r["cell"]["defense"]].append(r["final_acc"])
    ranked = sorted(by_method.items(), key=lambda kv: -mean(kv[1]))
    out.append("## H1: overall method ranking (mean final_acc)\n")
    out.append("| method | mean_acc | n |")
    out.append("| --- | --- | --- |")
    for m, v in ranked:
        out.append(f"| {m} | {mean(v):.4f} | {len(v)} |")
    out.append("")

    # H2/H3: per-topology ranking + variance across topologies
    out.append("## H2/H3: per-topology ranking and cross-topology variance\n")
    out.append("| topology | " + " | ".join(m for m, _ in ranked) + " |")
    out.append("| --- | " + " | ".join(["---"] * len(ranked)) + " |")
    per_topo = {}
    for topo in ["ring", "star", "complete"]:
        row = []
        for m, _ in ranked:
            v = [r["final_acc"] for r in main
                 if r["cell"]["topology"] == topo and r["cell"]["defense"] == m]
            row.append(mean(v))
        per_topo[topo] = row
        out.append(f"| {topo} | " + " | ".join(f"{x:.4f}" for x in row) + " |")
    out.append("")
    out.append("Cross-topology std per method (lower = more topology-robust):")
    out.append("| method | std_across_topos |")
    out.append("| --- | --- |")
    for i, (m, _) in enumerate(ranked):
        vals = [per_topo[t][i] for t in ["ring", "star", "complete"]]
        vals = [v for v in vals if v is not None and v == v]  # drop missing/nan
        s = statistics.pstdev(vals) if len(vals) >= 2 else float("nan")
        note = " (⚠️ <3 topologies — incomplete data)" if len(vals) < 3 else ""
        out.append(f"| {m} | {s:.4f} |{note}")
    out.append("")

    # H7: selection value (topoguard - tg_trustonly)
    out.append("## H7: selection value (topoguard minus tg_trustonly)\n")
    out.append("| topology | attack | delta | n |")
    out.append("| --- | --- | --- | --- |")
    for topo in ["ring", "star", "complete"]:
        for atk in ["label_flip", "sign_flip", "gaussian", "backdoor", "alie"]:
            a = [r["final_acc"] for r in main if r["cell"]["topology"] == topo
                 and r["cell"]["attack"] == atk and r["cell"]["defense"] == "topoguard"]
            b = [r["final_acc"] for r in main if r["cell"]["topology"] == topo
                 and r["cell"]["attack"] == atk and r["cell"]["defense"] == "tg_trustonly"]
            if a and b:
                out.append(f"| {topo} | {atk} | {mean(a)-mean(b):+.4f} | {len(a)} |")
    out.append("")

    # H4: backdoor ASR
    out.append("## H4: backdoor ASR by defense (lower is better)\n")
    out.append("| defense | mean_ASR | mean_acc | n |")
    out.append("| --- | --- | --- | --- |")
    for m, _ in ranked:
        v = [r for r in main if r["cell"]["defense"] == m and r["cell"]["attack"] == "backdoor"]
        if v:
            out.append(f"| {m} | {mean([r['final_asr'] for r in v]):.4f} | "
                       f"{mean([r['final_acc'] for r in v]):.4f} | {len(v)} |")
    out.append("")

    # Fraction sensitivity
    out.append("## Byzantine fraction sensitivity (mean_acc)\n")
    out.append("| defense | 10% | 20% | 30% |")
    out.append("| --- | --- | --- | --- |")
    for m, _ in ranked:
        row = []
        for f in [0.1, 0.2, 0.3]:
            v = [r["final_acc"] for r in main
                 if r["cell"]["defense"] == m and r["cell"]["byz_frac"] == f]
            row.append(f"{mean(v):.4f}")
        out.append(f"| {m} | " + " | ".join(row) + " |")

    text = "\n".join(out)
    with open(os.path.join(args.out, "ANALYSIS.md"), "w") as f:
        f.write(text + "\n")
    print("wrote results/ANALYSIS.md")
    print(text[:3000])


if __name__ == "__main__":
    main()
