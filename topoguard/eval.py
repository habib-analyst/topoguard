"""Resumable evaluation-matrix runner.

Usage:
  python -m topoguard.eval --phase pilot --shard 0 --nshards 2 --out eval/pilot.jsonl
  python -m topoguard.eval --phase main  --shard 1 --nshards 2 --out eval/main.jsonl --resume

Each cell appends one JSON line (with cell_id) to --out. --resume skips
cells already present. A DONE marker file (<out>.done) is written when the
shard's cells are all complete. Launch shards detached (setsid+nohup); the
orchestrator relaunches with --resume after daemon kills.

Phases:
  pilot      mechanism selection: trust-only vs select-only vs fused
             (2 topo x 2 attacks x 20% x iid x 2 seeds = 24 cells)
  main       full topology x defense x attack x fraction matrix
             (3 topo x 6 methods x 5 attacks x 3 frac x iid/noniid x 3 seeds)
  heldout    random_geometric (never tuned on): 3 methods x 5 attacks x
             2 frac x iid/noniid x 3 seeds
  ablations  A1..A4 + full, on 3 topo x 2 attacks x 20% x noniid x 2 seeds
  timevary   time-varying graphs: 3 methods x 2 attacks x 20% x 2 seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from .simulate import run_cell

ATTACKS = ["label_flip", "sign_flip", "gaussian", "backdoor", "alie"]
METHODS = ["fedavg", "krum", "trimmedmean", "geomedian", "balance_fixed",
           "tg_trustonly", "topoguard"]
TUNE_TOPOS = ["ring", "star", "complete"]  # random_geometric is HELD OUT

# Cost controls for the large phases (pilot uses its own explicit values).
N_NODES = 10
ROUNDS = 15
SAMPLES = 120

V_TRUST = {"select": False, "fixed_level": 0, "topology": True, "trust": True}
V_SELECT = {"select": True, "topology": False, "trust": True}
V_FUSED = {"select": True, "topology": True, "trust": True}
ABLATIONS = {
    "A1_no_select": {"select": False, "fixed_level": 2, "topology": True, "trust": True},
    "A2_no_topology": {"select": True, "topology": False, "trust": True},
    "A3_no_trust": {"select": True, "topology": True, "trust": False},
    "A4_small_portfolio": {
        "select": True, "topology": True, "trust": True, "portfolio": "small",
    },
}


def cell_id(cfg: dict) -> str:
    canon = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def pilot_cells():
    cells = []
    variants = {"V_trust": V_TRUST, "V_select": V_SELECT, "V_fused": V_FUSED}
    for vname, kw in variants.items():
        for topo in ["ring", "complete"]:
            for attack in ["sign_flip", "backdoor"]:
                for seed in [0, 1]:
                    cells.append({
                        "phase": "pilot", "variant": vname, "defense": "topoguard",
                        "topoguard_kw": kw, "topology": topo, "attack": attack,
                        "byz_frac": 0.2, "noniid": False, "seed": seed,
                        "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                    })
    return cells


def main_cells(tg_kw: dict):
    cells = []
    for topo in TUNE_TOPOS:
        for method in METHODS:
            for attack in ATTACKS:
                for frac in [0.1, 0.2, 0.3]:
                    for noniid in [False, True]:
                        for seed in [0, 1, 2]:
                            kw = (dict(tg_kw) if method in ("topoguard", "tg_trustonly")
                                  else {})
                            cells.append({
                                "phase": "main", "defense": method,
                                "topoguard_kw": kw, "topology": topo,
                                "attack": attack, "byz_frac": frac,
                                "noniid": noniid, "seed": seed,
                                "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                            })
    return cells


def heldout_cells(tg_kw: dict):
    cells = []
    for method in ["fedavg", "balance_fixed", "topoguard"]:
        for attack in ATTACKS:
            for frac in [0.2, 0.3]:
                for noniid in [False, True]:
                    for seed in [0, 1, 2]:
                        kw = dict(tg_kw) if method == "topoguard" else {}
                        cells.append({
                            "phase": "heldout", "defense": method,
                            "topoguard_kw": kw, "topology": "random_geometric",
                            "attack": attack, "byz_frac": frac,
                            "noniid": noniid, "seed": seed,
                            "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                        })
    return cells


def ablation_cells(tg_kw: dict):
    cells = []
    variants = {"full": dict(tg_kw), **{k: v for k, v in ABLATIONS.items()}}
    for vname, kw in variants.items():
        for topo in TUNE_TOPOS:
            for attack in ["sign_flip", "backdoor"]:
                for seed in [0, 1]:
                    cells.append({
                        "phase": "ablation", "variant": vname, "defense": "topoguard",
                        "topoguard_kw": kw, "topology": topo, "attack": attack,
                        "byz_frac": 0.2, "noniid": True, "seed": seed,
                        "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                    })
    return cells


def timevary_cells(tg_kw: dict):
    cells = []
    for method in ["fedavg", "balance_fixed", "topoguard"]:
        for attack in ["sign_flip", "backdoor"]:
            for seed in [0, 1]:
                kw = dict(tg_kw) if method == "topoguard" else {}
                cells.append({
                    "phase": "timevary", "defense": method,
                    "topoguard_kw": kw, "topology": "time_varying",
                    "attack": attack, "byz_frac": 0.2, "noniid": False,
                    "seed": seed, "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                })
    return cells


def tune_cells(_tg_kw: dict):
    """Hyperparameter grid for the fused mechanism.

    Tuned ONLY on ring/star (never on the held-out random_geometric).
    Grid: c x kappa x tau3; gamma/ldown/lup fixed on principle.
    tau3 is the L3 reservation threshold: L3 (geometric median) is the
    rule of last resort for *unidentifiable* threats (murky trust);
    clearly-identified Byzantine are better handled by trust-exclusion
    (L0/L2), so tau3=0.9 reserves L3 for high-exposure murky cases.
    18 combos x (2 topo x 2 attacks x 20% x noniid x 1 seed) = 72 cells.
    """
    cells = []
    for c in [0.25, 0.5, 0.75]:
        for kappa in [0.5, 1.0, 2.0]:
            for tau3 in [0.7, 0.9]:
                kw = {"select": True, "topology": True, "trust": True,
                      "c": c, "kappa": kappa, "tau": [0.2, 0.45, tau3]}
                for topo in ["ring", "star"]:
                    for attack in ["sign_flip", "backdoor"]:
                        cells.append({
                            "phase": "tune", "defense": "topoguard",
                            "topoguard_kw": kw, "topology": topo,
                            "attack": attack, "byz_frac": 0.2,
                            "noniid": True, "seed": 0,
                            "n_nodes": N_NODES, "rounds": ROUNDS,
                        "samples_per_client": SAMPLES,
                        })
    return cells


PHASES = {
    "pilot": pilot_cells,
    "tune": tune_cells,
    "main": main_cells,
    "heldout": heldout_cells,
    "ablation": ablation_cells,
    "timevary": timevary_cells,
}


def load_done(out_path: str) -> set:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["cell_id"])
                    except Exception:
                        pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tgkw", default="{}",
                    help="JSON overrides for TopoGuard hyperparams (main/heldout/ablation/timevary)")
    ap.add_argument("--tgkw-file", default=None,
                    help="Path to a JSON file with TopoGuard hyperparams (avoids shell quoting)")
    args = ap.parse_args()

    tg_kw = json.loads(args.tgkw)
    if args.tgkw_file:
        with open(args.tgkw_file) as f:
            tg_kw = json.load(f)
    fn = PHASES[args.phase]
    cells = fn(tg_kw) if args.phase != "pilot" else fn()
    shard = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]
    print(f"phase={args.phase} cells={len(cells)} shard={args.shard}/{args.nshards} "
          f"shard_cells={len(shard)} -> {args.out}", flush=True)

    done = load_done(args.out) if args.resume else set()
    todo = [c for c in shard if cell_id(c) not in done]
    print(f"already done={len(done)} todo={len(todo)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        for k, cfg in enumerate(todo):
            cid = cell_id(cfg)
            try:
                res = run_cell(cfg)
            except Exception as e:  # never let one cell kill the shard
                res = {"error": f"{type(e).__name__}: {e}"}
            res["cell_id"] = cid
            res["cell"] = cfg
            f.write(json.dumps(res) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(todo)} cells", flush=True)

    remaining = [c for c in shard if cell_id(c) not in load_done(args.out)]
    if not remaining:
        open(f"{args.out}.done.shard{args.shard}", "w").write("done\n")
        print("SHARD COMPLETE", flush=True)
    else:
        print(f"shard incomplete: {len(remaining)} cells remain", flush=True)


if __name__ == "__main__":
    main()
