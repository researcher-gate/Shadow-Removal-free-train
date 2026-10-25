"""Select e2's scalars on the ISTD+ TRAIN split, and run the design ablation.

Pass 1 is the ablation that justifies the design: colour domain x affine-vs-
multiplicative x measured-vs-fixed matte, all 8 combinations. Pass 2 tunes the
sampling geometry around the winner.

Nothing here ever sees the test split.

    python experiments/e2/tune_e2.py --n 60
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

PASS1 = {
    "linear": [False, True],
    "affine": [True, False],
    "measured_matte": [True, False],
}
PASS2 = {
    "strip": [8, 14, 22],
    "profile_radius": [20, 28, 40],
    "max_brighten": [8.0, 16.0, 32.0],
}


def score(cfg, samples):
    fn = IAR(cfg)
    ps, pns, pa, mae = [], [], [], []
    for s in samples:
        img, mask, gt = s.load()
        pred, _ = fn(img, mask)
        m = M.compute_metrics(pred, gt, mask, with_lpips=False)
        ps.append(m["psnr_S"]); pns.append(m["psnr_NS"])
        pa.append(m["psnr_ALL"]); mae.append(m["mae_lab_ALL"])
    return (float(np.nanmean(ps)), float(np.nanmean(pns)),
            float(np.nanmean(pa)), float(np.nanmean(mae)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = random.Random(args.seed).sample(load_split("train"), args.n)
    print(f"tuning on {len(samples)} TRAIN images (seed {args.seed})")

    # the untouched input fixes the non-shadow ceiling we must not damage
    ps0, ns0 = [], []
    for s in samples:
        img, mask, gt = s.load()
        m = M.compute_metrics(img, gt, mask, with_lpips=False)
        ps0.append(m["psnr_S"]); ns0.append(m["psnr_NS"])
    input_ns = float(np.nanmean(ns0))
    print(f"input PSNR_S {np.nanmean(ps0):.3f}  PSNR_NS {input_ns:.3f} "
          f"(guard: configs must stay above {input_ns - 0.3:.3f})")

    out = {"split": "train", "n_images": args.n, "seed": args.seed,
           "input_psnr_NS": input_ns,
           "criterion": "max mean PSNR_S subject to PSNR_NS >= input - 0.3 dB"}

    rows = []
    t0 = time.perf_counter()
    print("\n--- pass 1: design ablation ---")
    for combo in itertools.product(*PASS1.values()):
        kw = dict(zip(PASS1, combo))
        s_, ns_, pa_, mae_ = score(IARConfig(**kw), samples)
        ok = ns_ >= input_ns - 0.3
        rows.append({"cfg": kw, "psnr_S": s_, "psnr_NS": ns_, "psnr_ALL": pa_,
                     "mae_lab_ALL": mae_, "admissible": bool(ok)})
        print(f"  {kw} -> S {s_:.3f}  NS {ns_:.3f}  ALL {pa_:.3f}  mae {mae_:.3f}"
              f"{'' if ok else '   [rejected: damages lit region]'}", flush=True)
    adm = [r for r in rows if r["admissible"]] or rows
    best1 = max(adm, key=lambda r: r["psnr_S"])
    print("BEST pass1:", best1["cfg"], f"S={best1['psnr_S']:.3f}")

    rows2 = []
    print("\n--- pass 2: sampling geometry ---")
    for combo in itertools.product(*PASS2.values()):
        kw = dict(zip(PASS2, combo))
        cfg = IARConfig(**{**best1["cfg"], **kw})
        s_, ns_, pa_, mae_ = score(cfg, samples)
        ok = ns_ >= input_ns - 0.3
        rows2.append({"cfg": kw, "psnr_S": s_, "psnr_NS": ns_, "psnr_ALL": pa_,
                      "mae_lab_ALL": mae_, "admissible": bool(ok)})
        print(f"  {kw} -> S {s_:.3f}  NS {ns_:.3f}  mae {mae_:.3f}"
              f"{'' if ok else '   [rejected]'}", flush=True)
    adm2 = [r for r in rows2 if r["admissible"]] or rows2
    best2 = max(adm2, key=lambda r: r["psnr_S"])

    chosen = {**best1["cfg"], **best2["cfg"]}
    edge = [k for k, v in best2["cfg"].items() if v in (min(PASS2[k]), max(PASS2[k]))]
    if edge:
        print(f"NOTE: winner on the grid edge for {edge}; widen and rerun to converge")

    out.update({"pass1_grid": PASS1, "pass1": rows, "pass1_best": best1,
                "pass2_grid": PASS2, "pass2": rows2, "pass2_best": best2,
                "winner_on_grid_edge": edge,
                "chosen_config": chosen,
                "seconds": time.perf_counter() - t0})
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "tuning_train_e2.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
    print("\nchosen:", chosen)
    print("->", p)


if __name__ == "__main__":
    main()
