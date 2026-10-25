"""Finish the e1 tuning that tune_e1.py left at the edge of its grid.

tune_e1.py's second pass chose `gain_smooth_frac = 0.35` and `feather_width = 8`
-- both the smallest values it offered.  A grid whose winner sits on its own
boundary has not converged, and the oracle decomposition in
baseline/results/ (and RESULTS.md) says why it matters: the Stage 5
parameterisation, not the multiplicative model, is what caps this method.
With the true gain field from the GT, PSNR_S over the shadow region is

    sigma=15, clamp 8, feathered  (Stage 5 as written)   22.0 dB
    sigma=3                                              24.4 dB
    no smoothing                                         26.8 dB
    no smoothing, clamp 50                               30.2 dB
    no smoothing, clamp 50, hard mask                    44.4 dB

so smoothing and feathering throw away more headroom than any estimator
improvement could recover.  This pass searches that corner properly, on TRAIN,
and rewrites `chosen_config` in tuning_train_e1.json.

    python experiments/e1/refine_e1.py --n 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np

from common import metrics as M
from common.istd import load_split
from experiments.e1.iat import IAT, IATConfig

RES = os.path.join(HERE, "results")
TUNING = os.path.join(RES, "tuning_train_e1.json")

GRID = {
    "gain_smooth_frac": [0.05, 0.10, 0.20, 0.35],
    "feather_width": [4, 8, 12],
    "gain_max": [6.0, 12.0],
}


def score(cfg, samples):
    fn = IAT(cfg)
    ps, pns, mae = [], [], []
    for s in samples:
        img, mask, gt = s.load()
        pred, _ = fn(img, mask)
        m = M.compute_metrics(pred, gt, mask, with_lpips=False)
        ps.append(m["psnr_S"]); pns.append(m["psnr_NS"]); mae.append(m["mae_lab_ALL"])
    return float(np.nanmean(ps)), float(np.nanmean(pns)), float(np.nanmean(mae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(TUNING):
        print("run experiments/e1/tune_e1.py first"); return 1
    tun = json.load(open(TUNING, encoding="utf-8"))
    base_cfg = dict(tun["chosen_config"])
    print("starting from:", base_cfg)

    samples = random.Random(args.seed).sample(load_split("train"), args.n)
    base = IATConfig(**{**base_cfg, "verify": False})
    s0, ns0, mae0 = score(base, samples)
    print(f"incumbent -> S {s0:.3f}  NS {ns0:.3f}  mae {mae0:.3f}")

    rows = [{"cfg": {k: getattr(base, k) for k in GRID},
             "psnr_S": s0, "psnr_NS": ns0, "mae_lab_ALL": mae0,
             "incumbent": True}]
    t0 = time.perf_counter()
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo))
        if all(getattr(base, k) == v for k, v in kw.items()):
            continue
        cfg = IATConfig(**{**base_cfg, **kw, "verify": False})
        s_, ns_, mae_ = score(cfg, samples)
        # a config that wins shadow PSNR by wrecking the lit region is not a win
        ok = ns_ >= ns0 - 0.3
        rows.append({"cfg": kw, "psnr_S": s_, "psnr_NS": ns_, "mae_lab_ALL": mae_,
                     "admissible": bool(ok)})
        print(f"  {kw} -> S {s_:.3f}  NS {ns_:.3f}  mae {mae_:.3f}"
              f"{'' if ok else '   [rejected: damages lit region]'}", flush=True)

    adm = [r for r in rows if r.get("admissible", True)]
    best = max(adm, key=lambda r: r["psnr_S"])
    gain = best["psnr_S"] - s0
    print(f"\nBEST {best['cfg']}  S={best['psnr_S']:.3f}  ({gain:+.3f} dB vs incumbent)")

    on_edge = [k for k, v in best["cfg"].items()
               if v in (min(GRID[k]), max(GRID[k])) and len(GRID[k]) > 1]
    if on_edge:
        print(f"NOTE: winner still on the grid edge for {on_edge} -- "
              f"widen GRID and rerun if you want this converged")

    tun["refine_pass"] = {
        "grid": GRID, "n_images": args.n, "seed": args.seed,
        "incumbent": {"psnr_S": s0, "psnr_NS": ns0, "mae_lab_ALL": mae0},
        "rows": rows, "best": best, "delta_psnr_S": gain,
        "winner_on_grid_edge": on_edge,
        "seconds": time.perf_counter() - t0,
    }
    tun["chosen_config"] = {**base_cfg, **best["cfg"]}
    json.dump(tun, open(TUNING, "w", encoding="utf-8"), indent=2)
    print("chosen_config ->", tun["chosen_config"])
    print("->", TUNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
