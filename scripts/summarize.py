"""Rebuild every paper table from measured eval/*.jsonl records.

No hand-typed numbers: PAPER.md's [FILL] cells come from here.
Usage: python scripts/summarize.py [--evaldir eval] [--out results/]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def load(evaldir: str) -> list[dict]:
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


def pivot(recs, row_key, col_key, val="final_acc"):
    rows, cols = set(), set()
    grid = defaultdict(list)
    for r in recs:
        c = r.get("cell", {})
        rk, ck = row_key(c), col_key(c)
        rows.add(rk)
        cols.add(ck)
        grid[(rk, ck)].append(r.get(val))
    rows, cols = sorted(rows), sorted(cols)
    table = [[rk] + [round(mean(grid[(rk, ck)]), 4) for ck in cols] for rk in rows]
    return ["variant/topology/etc"] + cols, table


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(map(str, header)) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", default="eval")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = load(args.evaldir)
    print(f"loaded {len(recs)} cell records")
    by_phase = defaultdict(list)
    for r in recs:
        by_phase[r.get("cell", {}).get("phase", "?")].append(r)
    print("phases:", {k: len(v) for k, v in by_phase.items()})

    report = []

    # ---- pilot: variant comparison ----
    if "pilot" in by_phase:
        p = by_phase["pilot"]
        h, t = pivot(p, lambda c: c.get("variant"), lambda c: f'{c["topology"]}/{c["attack"]}')
        report.append("## Pilot: mechanism selection (final_acc, mean of 2 seeds)\n"
                      + md_table(h, t))
        # margin of fused vs others
        agg = defaultdict(list)
        for r in p:
            agg[r["cell"]["variant"]].append(r["final_acc"])
        report.append("\n**Mean over pilot cells:** " +
                      ", ".join(f"{k}={mean(v):.4f}" for k, v in sorted(agg.items())))

    # ---- tune: hyperparameter grid ----
    if "tune" in by_phase:
        t = by_phase["tune"]
        agg = defaultdict(list)
        for r in t:
            kw = r["cell"]["topoguard_kw"]
            key = (f"c={kw['c']} kappa={kw['kappa']} "
                   f"tau3={kw.get('tau', [0.2, 0.45, 0.7])[2]}")
            agg[key].append(r["final_acc"])
        ranked = sorted(agg.items(), key=lambda kv: -mean(kv[1]))
        report.append("## Tune: hyperparameter grid (mean final_acc over "
                      "ring/star x sign_flip/backdoor, non-IID)\n"
                      + md_table(["hyperparams", "mean_acc", "n"],
                                 [[k, round(mean(v), 4), len(v)]
                                  for k, v in ranked]))

    # ---- main matrix: per-topology defense x attack ----
    if "main" in by_phase:
        m = by_phase["main"]
        for topo in sorted({r["cell"]["topology"] for r in m}):
            sub = [r for r in m if r["cell"]["topology"] == topo]
            h, t = pivot(sub, lambda c: c["defense"], lambda c: c["attack"])
            report.append(f"\n## Main: {topo} — defense x attack "
                          f"(final_acc, mean over frac/data/seeds)\n" + md_table(h, t))
            # backdoor ASR companion
            bd = [r for r in sub if r["cell"]["attack"] == "backdoor"]
            if bd:
                h2, t2 = pivot(bd, lambda c: c["defense"],
                               lambda c: f'frac={c["byz_frac"]}', val="final_asr")
                report.append(f"\n### {topo} backdoor ASR (lower is better)\n"
                              + md_table(h2, t2))

    # ---- heldout: random_geometric ----
    if "heldout" in by_phase:
        h0 = by_phase["heldout"]
        h, t = pivot(h0, lambda c: c["defense"], lambda c: c["attack"])
        report.append("\n## Held-out family: random_geometric (never tuned on)\n"
                      + md_table(h, t))

    # ---- ablations ----
    if "ablation" in by_phase:
        a = by_phase["ablation"]
        h, t = pivot(a, lambda c: c.get("variant"), lambda c: f'{c["topology"]}/{c["attack"]}')
        report.append("\n## Ablations (final_acc, non-IID, 20% Byzantine)\n"
                      + md_table(h, t))

    # ---- time-varying ----
    if "timevary" in by_phase:
        tv = by_phase["timevary"]
        h, t = pivot(tv, lambda c: c["defense"], lambda c: c["attack"])
        report.append("\n## Time-varying topology stress\n" + md_table(h, t))

    text = "\n\n".join(report)
    with open(os.path.join(args.out, "TABLES.md"), "w") as f:
        f.write("# TopoGuard results (measured)\n\n" + text + "\n")
    print("wrote results/TABLES.md")
    print(text[:2000])


if __name__ == "__main__":
    main()
