"""Run e2 on the ISTD+ test split through the same harness as baseline/ and e1,
so metrics AND latency are directly comparable.

Produces:
  results/e2_main.json              headline run, predictions saved
  results/ablation_design_*.json    the four design arms, on the full test set
  results/mask_sensitivity_*.json   robustness to mask dilation/erosion
  outputs/e2_iar/*.png              per-image predictions

    python experiments/e2/run_e2.py
    python experiments/e2/run_e2.py --stages main ablation masksens
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import cv2

from common.imgutil import disk
from common.runner import run_method
from experiments.e2.iar import IAR, IARConfig

RES = os.path.join(HERE, "results")
OUTIMG = os.path.join(HERE, "outputs")


def chosen_config(**over) -> IARConfig:
    p = os.path.join(RES, "tuning_train_e2.json")
    kw = {}
    if os.path.exists(p):
        kw = dict(json.load(open(p, encoding="utf-8"))["chosen_config"])
    else:
        print("WARNING: no tuning_train_e2.json; using defaults")
    kw.update(over)
    return IARConfig(**kw)


def mask_perturb(fn, n):
    def g(img, mask):
        if n:
            k = disk(abs(n))
            mask = cv2.dilate(mask, k) if n > 0 else cv2.erode(mask, k)
        return fn(img, mask)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=["main", "ablation", "masksens"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    cfg = chosen_config()
    print("config:", asdict(cfg))

    if "main" in args.stages:
        run_method("e2_iar", IAR(cfg),
                   out_json=os.path.join(RES, f"e2_main{args.tag}.json"),
                   save_dir=os.path.join(OUTIMG, "e2_iar"),
                   limit=args.limit, with_lpips=not args.no_lpips,
                   extra_meta={"config": asdict(cfg),
                               "config_selected_on": "ISTD+ train, see "
                                                     "results/tuning_train_e2.json"})

    if "ablation" in args.stages:
        arms = {
            "affine_measured":      dict(affine=True, measured_matte=True),
            "affine_fixedmatte":    dict(affine=True, measured_matte=False),
            "mult_measured":        dict(affine=False, measured_matte=True),
            "mult_fixedmatte":      dict(affine=False, measured_matte=False),
            "affine_measured_lin":  dict(affine=True, measured_matte=True, linear=True),
        }
        for name, kw in arms.items():
            c = chosen_config(**kw)
            run_method(f"e2_{name}", IAR(c),
                       out_json=os.path.join(RES, f"ablation_design_{name}{args.tag}.json"),
                       limit=args.limit, with_lpips=False, eval_full=False,
                       extra_meta={"config": asdict(c), "arm": name,
                                   "note": "ablation arms scored at 256 only"})

    if "masksens" in args.stages:
        for n in (-12, -6, 0, 6, 12):
            run_method(f"e2_mask{n:+d}", mask_perturb(IAR(cfg), n),
                       out_json=os.path.join(RES, f"mask_sensitivity_{n:+d}{args.tag}.json"),
                       limit=args.limit, with_lpips=False, eval_full=False,
                       extra_meta={"mask_morphology_px": n,
                                   "note": "positive = dilate, negative = erode; "
                                           "metrics always use the UNPERTURBED mask"})

    print("done")


if __name__ == "__main__":
    main()
