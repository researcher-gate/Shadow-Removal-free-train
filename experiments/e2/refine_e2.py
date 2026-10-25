"""Second tuning pass for e2, on TRAIN.

Three things need it:

  * `strip` and `max_brighten` both won on the EDGE of tune_e2.py's grid, so
    that search had not converged.
  * The self-check thresholds (`trust_lo`, `trust_hi`) are new. Hand-set values
    recovered the worst failures by +6.5 dB but over-triggered on a legitimate
    case, which is exactly what a fitted threshold is for.

Selection criterion is deliberately NOT mean PSNR. A gate whose whole job is
to stop catastrophic failures must be scored on the tail, so configurations are
ranked by mean PSNR_S with a hard constraint on the 5th percentile and on how
many images end up worse than the untouched input.

    python experiments/e2/refine_e2.py --n 90
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
from experiments.e2.iar import IAR, IARConfig

RES = os.path.join(HERE, "results")
TUNING = os.path.join(RES, "tuning_train_e2.json")

GRID = {
    "trust_lo": [0.20, 0.25, 0.30],
    "trust_hi": [0.40, 0.50, 0.65],
    "strip": [4, 6, 8],
    "max_brighten": [32.0, 64.0],
}


def score(cfg, samples, base):
    fn = IAR(cfg)
    ps, pns, mae = [], [], []
    for s in samples:
        img, mask, gt = s.load()
        pred, _ = fn(img, mask)
        m = M.compute_metrics(pred, gt, mask, with_lpips=False)
        ps.append(m["psnr_S"]); pns.append(m["psnr_NS"]); mae.append(m["mae_lab_ALL"])
    ps = np.array(ps)
    worse = int((ps < base).sum())
    return {"psnr_S": float(ps.mean()), "psnr_S_p05": float(np.percentile(ps, 5)),
            "psnr_S_min": float(ps.min()), "psnr_NS": float(np.nanmean(pns)),
            "mae_lab_ALL": float(np.nanmean(mae)),
            "n_worse_than_input": worse,
            "frac_worse_than_input": float(worse / len(ps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    tun = json.load(open(TUNING, encoding="utf-8"))
    base_cfg = dict(tun["chosen_config"])
    print("starting from:", base_cfg)

    samples = random.Random(args.seed).sample(load_split("train"), args.n)
    base_ps, base_ns = [], []
    for s in samples:
        img, mask, gt = s.load()
        m = M.compute_metrics(img, gt, mask, with_lpips=False)
        base_ps.append(m["psnr_S"]); base_ns.append(m["psnr_NS"])
    base_ps = np.array(base_ps)
    input_ns = float(np.mean(base_ns))
    print(f"input: PSNR_S {base_ps.mean():.3f}  PSNR_NS {input_ns:.3f}")

    rows = []
    t0 = time.perf_counter()
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo))
        if kw["trust_hi"] <= kw["trust_lo"]:
            continue
        r = score(IARConfig(**{**base_cfg, **kw}), samples, base_ps)
        r["cfg"] = kw
        r["admissible"] = bool(r["psnr_NS"] >= input_ns - 0.3)
        rows.append(r)
        print(f"  {kw} -> S {r['psnr_S']:.3f}  p05 {r['psnr_S_p05']:.2f}  "
              f"min {r['psnr_S_min']:.2f}  NS {r['psnr_NS']:.3f}  "
              f"worse_than_input {r['n_worse_than_input']}"
              f"{'' if r['admissible'] else '   [rejected]'}", flush=True)

    adm = [r for r in rows if r["admissible"]] or rows
    # tail-aware: among configs within 0.15 dB of the best mean, take the one
    # with the best 5th percentile. A gate against catastrophes is pointless if
    # it is selected on the mean alone.
    top = max(adm, key=lambda r: r["psnr_S"])["psnr_S"]
    near = [r for r in adm if r["psnr_S"] >= top - 0.15]
    best = max(near, key=lambda r: (r["psnr_S_p05"], r["psnr_S"]))
    print(f"\nBEST {best['cfg']}  S={best['psnr_S']:.3f}  p05={best['psnr_S_p05']:.2f}  "
          f"worse_than_input={best['n_worse_than_input']}/{len(samples)}")
    edge = [k for k, v in best["cfg"].items() if v in (min(GRID[k]), max(GRID[k]))]
    if edge:
        print(f"NOTE: still on the grid edge for {edge}")

    tun["refine_pass"] = {"grid": GRID, "n_images": args.n, "seed": args.seed,
                          "input_psnr_S": float(base_ps.mean()),
                          "input_psnr_NS": input_ns,
                          "selection": "max PSNR_S_p05 among configs within 0.15 dB "
                                       "of the best mean PSNR_S",
                          "rows": rows, "best": best,
                          "winner_on_grid_edge": edge,
                          "seconds": time.perf_counter() - t0}
    tun["chosen_config"] = {**base_cfg, **best["cfg"]}
    json.dump(tun, open(TUNING, "w", encoding="utf-8"), indent=2)
    print("chosen_config ->", tun["chosen_config"])


if __name__ == "__main__":
    main()
