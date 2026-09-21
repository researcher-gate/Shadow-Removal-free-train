"""Select e1's free parameters on the ISTD+ TRAIN split.

Stage 0 fixes C (experiments/e1/calibrate.py).  What is left are the two scale
multipliers a_s (patch size) and a_b (band cutoff), which C1 of CORRECTIONS.md
insists are independent, plus the estimator switches.  All of them are chosen
here, on train, and then frozen for the single test run.

    python experiments/e1/tune_e1.py --n 50
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


def load_C(default=1.524):
    p = os.path.join(RES, "stage0_calibration.json")
    if not os.path.exists(p):
        return default, "default (calibration not run)"
    d = json.load(open(p, encoding="utf-8"))
    est = d["best_estimator_by_abs_spearman"]
    return d["fits"][est]["C_through_origin"], est


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

    C, est = load_C()
    print(f"Stage 0: C = {C:.4f}  (estimator: {est})")

    samples = random.Random(args.seed).sample(load_split("train"), args.n)
    base = IATConfig(C=C, verify=False)

    grid = {
        "a_s": [0.5, 1.0, 2.0],
        "a_b": [0.5, 1.0, 2.0, 4.0],
        "use_prior": [True, False],
    }
    rows = []
    t0 = time.perf_counter()
    for combo in itertools.product(*grid.values()):
        cfg = IATConfig(**{**base.__dict__, **dict(zip(grid, combo))})
        s_, ns_, mae_ = score(cfg, samples)
        rows.append({"cfg": dict(zip(grid, combo)), "psnr_S": s_, "psnr_NS": ns_,
                     "mae_lab_ALL": mae_})
        print(f"  {dict(zip(grid, combo))} -> S {s_:.3f}  NS {ns_:.3f}  "
              f"mae {mae_:.3f}", flush=True)
    best = max(rows, key=lambda r: r["psnr_S"])
    print("BEST:", best)

    # second pass: a few estimator knobs around the winner
    rows2 = []
    for topk, gsf, fw in itertools.product([2, 4, 8], [0.35, 0.5, 0.8], [8, 12]):
        kw = {"topk": topk, "gain_smooth_frac": gsf, "feather_width": fw}
        cfg = IATConfig(**{**base.__dict__, **best["cfg"], **kw})
        s_, ns_, mae_ = score(cfg, samples)
        rows2.append({"cfg": kw, "psnr_S": s_, "psnr_NS": ns_, "mae_lab_ALL": mae_})
        print(f"  {kw} -> S {s_:.3f} NS {ns_:.3f}", flush=True)
    best2 = max(rows2, key=lambda r: r["psnr_S"])

    chosen = {**best["cfg"], **best2["cfg"], "C": C}
    out = {"split": "train", "n_images": args.n, "seed": args.seed,
           "stage0_C": C, "stage0_estimator": est,
           "pass1_grid": grid, "pass1": rows, "pass1_best": best,
           "pass2": rows2, "pass2_best": best2,
           "chosen_config": chosen,
           "seconds": time.perf_counter() - t0}
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "tuning_train_e1.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
    print("chosen:", chosen)
    print("->", p)


if __name__ == "__main__":
    main()
