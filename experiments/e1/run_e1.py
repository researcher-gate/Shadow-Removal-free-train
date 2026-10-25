"""Run e1 on the ISTD+ test split -- identical harness, protocol and metrics as
baseline/run_baseline.py, so the two result sets are directly comparable.

Produces:
  results/e1_main.json              headline run (verification on)
  results/e1_main_noverify.json     same estimator, verification off -> the
                                    latency split between inference and self-check
  results/ablation_8_1_*.json       gating 8.1, the four arms (re-specified)
  results/mask_sensitivity_8_5.json gating 8.5
  results/selfcheck_8_3.json        gating 8.3 -- does the reject signal predict
                                    true PSNR?  (the headline contribution)

    python experiments/e1/run_e1.py --stages main ablation masksens selfcheck
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
import numpy as np

from common.imgutil import disk
from common.runner import run_method
from experiments.e1.iat import IAT, IATConfig

RES = os.path.join(HERE, "results")
OUTIMG = os.path.join(HERE, "outputs")


def chosen_config(verify=True) -> IATConfig:
    p = os.path.join(RES, "tuning_train_e1.json")
    kw = {}
    if os.path.exists(p):
        kw = dict(json.load(open(p, encoding="utf-8"))["chosen_config"])
    else:
        c = os.path.join(RES, "stage0_calibration.json")
        if os.path.exists(c):
            d = json.load(open(c, encoding="utf-8"))
            kw["C"] = d["fits"][d["best_estimator_by_abs_spearman"]]["C_through_origin"]
    kw["verify"] = verify
    return IATConfig(**kw)


def mask_perturb(fn, n):
    """Wrap a method so the mask it receives is dilated (n>0) / eroded (n<0)."""
    def g(img, mask):
        if n != 0:
            k = disk(abs(n))
            mask = cv2.dilate(mask, k) if n > 0 else cv2.erode(mask, k)
        return fn(img, mask)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", nargs="*",
                    default=["main", "ablation", "masksens"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    cfg = chosen_config(verify=True)
    print("config:", asdict(cfg))

    if "main" in args.stages:
        r = run_method("e1_iat", IAT(cfg), out_json=os.path.join(RES, f"e1_main{args.tag}.json"),
                       save_dir=os.path.join(OUTIMG, "e1_iat"), limit=args.limit,
                       extra_meta={"config": asdict(cfg),
                                   "note": "verification ON; latency includes the "
                                           "synthetic-shadow round trip"})
        cfg0 = chosen_config(verify=False)
        run_method("e1_iat_noverify", IAT(cfg0),
                   out_json=os.path.join(RES, f"e1_main_noverify{args.tag}.json"),
                   limit=args.limit,
                   extra_meta={"config": asdict(cfg0),
                               "note": "identical output pixels to e1_iat; isolates "
                                       "the cost of self-verification"})
        del r

    if "ablation" in args.stages:
        arms = {
            "arm1_fixed_s_fixed_b":  dict(geometry_patch=False, geometry_band=False),
            "arm2_fixed_s_geo_b":    dict(geometry_patch=False, geometry_band=True),
            "arm3_geo_s_fixed_b":    dict(geometry_patch=True, geometry_band=False),
            "arm4_geo_s_geo_b":      dict(geometry_patch=True, geometry_band=True),
            "arm3lit_geo_s_no_band": dict(geometry_patch=True, full_band=True),
        }
        for name, kw in arms.items():
            c = IATConfig(**{**asdict(cfg), "verify": False, **kw})
            run_method(f"e1_{name}", IAT(c),
                       out_json=os.path.join(RES, f"ablation_8_1_{name}{args.tag}.json"),
                       limit=args.limit, with_lpips=False, eval_full=False,
                       save_dir=(os.path.join(OUTIMG, name)
                                 if name == "arm1_fixed_s_fixed_b" else None),
                       extra_meta={"config": asdict(c), "arm": name,
                                   "note": "ablation arms are scored at 256 only "
                                           "(PSNR/SSIM/MAE-Lab); LPIPS and full-res "
                                           "add cost without changing the ranking"})

    if "trainverify" in args.stages:
        # the reject-signal combiner must be fitted somewhere other than the
        # test set it is then evaluated on (guideline 5.5)
        run_method("e1_iat_train", IAT(cfg), split="train",
                   out_json=os.path.join(RES, f"e1_train_verify{args.tag}.json"),
                   limit=args.limit or 200,
                   extra_meta={"config": asdict(cfg),
                               "note": "train-split run used only to fit the "
                                       "reject-signal combiner"})

    if "masksens" in args.stages:
        c = IATConfig(**{**asdict(cfg), "verify": False})
        for n in (-12, -6, 0, 6, 12):
            run_method(f"e1_mask{n:+d}", mask_perturb(IAT(c), n),
                       out_json=os.path.join(RES, f"mask_sensitivity_{n:+d}{args.tag}.json"),
                       limit=args.limit, with_lpips=False, eval_full=False,
                       extra_meta={"mask_morphology_px": n,
                                   "note": "positive = dilate, negative = erode; "
                                           "metrics always use the UNPERTURBED mask"})

    print("done")


if __name__ == "__main__":
    main()
