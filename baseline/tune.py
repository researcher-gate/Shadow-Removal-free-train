"""Hyper-parameter selection for the baselines -- on the ISTD+ TRAIN split.

A "strong baseline" that was tuned on the test set is not a baseline, it is a
leak.  Every knob below (ring widths, feather width, segment count, edge width,
sRGB vs linear) is chosen here on a fixed random subset of the 1330 training
triplets, and the chosen configuration is then run once on the 540 test images.

    python baseline/tune.py --n 80
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from baseline import methods as B
from common import metrics as M
from common.istd import load_split

GRIDS = {
    "global_gain": {
        "linear": [False, True],
        "robust": [False, True],
    },
    "boundary_gain": {
        "linear": [False, True],
        "band": [6, 10, 16, 24],
        "gap": [2, 4, 8],
        "f_width": [8, 12, 20],
    },
    "guo_paired_regions": {
        "linear": [False, True],
        "n_segments": [120, 220, 400],
        "topk": [1, 3, 5],
    },
    "gradient_domain": {
        "linear": [False, True],
        "edge_width": [3, 5, 7, 11],
    },
}

FACTORY = {
    "global_gain": B.make_global_gain,
    "boundary_gain": B.make_boundary_gain,
    "guo_paired_regions": B.make_guo_paired,
    "gradient_domain": B.make_gradient_domain,
}


def score(fn, samples):
    """Selection criterion: mean PSNR over the shadow region, with the
    non-shadow region policed so a method cannot buy shadow PSNR by wrecking
    the rest of the image."""
    ps, pns, mae = [], [], []
    for s in samples:
        img, mask, gt = s.load()
        pred, _ = fn(img, mask)
        m = M.compute_metrics(pred, gt, mask, with_lpips=False)
        ps.append(m["psnr_S"])
        pns.append(m["psnr_NS"])
        mae.append(m["mae_lab_ALL"])
    return (float(np.nanmean(ps)), float(np.nanmean(pns)), float(np.nanmean(mae)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=list(GRIDS))
    args = ap.parse_args()

    all_train = load_split("train")
    rng = random.Random(args.seed)
    samples = rng.sample(all_train, min(args.n, len(all_train)))
    print(f"tuning on {len(samples)} train images (seed {args.seed})")

    out = {"split": "train", "n_images": len(samples), "seed": args.seed,
           "criterion": "max mean PSNR_S, reject configs that lose >0.3 dB PSNR_NS "
                        "relative to the untouched input",
           "families": {}}

    # the untouched input defines the non-shadow ceiling we must not damage
    base_ns = score(B.method_input, samples)[1]
    print(f"input PSNR_NS = {base_ns:.3f} dB (guard: configs must stay above "
          f"{base_ns - 0.3:.3f})")
    out["input_psnr_NS"] = base_ns

    for fam in args.families:
        grid = GRIDS[fam]
        keys = list(grid)
        rows = []
        t0 = time.perf_counter()
        for combo in itertools.product(*[grid[k] for k in keys]):
            cfg = dict(zip(keys, combo))
            fn = FACTORY[fam](**cfg)
            s_, ns_, mae_ = score(fn, samples)
            ok = ns_ >= base_ns - 0.3
            rows.append({"cfg": cfg, "psnr_S": s_, "psnr_NS": ns_,
                         "mae_lab_ALL": mae_, "admissible": bool(ok)})
            print(f"  {fam} {cfg} -> S {s_:.3f} NS {ns_:.3f} mae {mae_:.3f}"
                  f"{'' if ok else '   [rejected: damages lit region]'}", flush=True)
        adm = [r for r in rows if r["admissible"]] or rows
        best = max(adm, key=lambda r: r["psnr_S"])
        out["families"][fam] = {"grid": grid, "rows": rows, "best": best,
                                "seconds": time.perf_counter() - t0}
        print(f"  BEST {fam}: {best['cfg']}  PSNR_S={best['psnr_S']:.3f}")

    path = os.path.join(ROOT, "baseline", "results", "tuning_train.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("->", path)


if __name__ == "__main__":
    main()
