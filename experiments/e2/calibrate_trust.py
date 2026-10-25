"""Calibrate e2's self-check thresholds from the TRAIN signal distribution.

Why not just tune them on train PSNR, like every other knob?

Because the failure the gate exists to catch IS NOT IN THE TRAIN SPLIT. ISTD+
train and test contain disjoint scenes (train 1-89, test 90-135), and the
scenes that break the two-strip same-material assumption -- shadows cast across
painted murals -- occur only in test. Measured over ~300 shadow instances per
split, the chroma-distance signal:

                p50     p75     p90     p95    p97.5    p99     max
    TRAIN      0.100   0.145   0.195   0.232   0.247   0.297   0.508
    TEST       0.104   0.171   0.248   0.324   0.573   0.701   1.023

The bodies agree; the tails do not. A PSNR-based search on train therefore
scores every threshold identically (zero failures to catch) and picks
arbitrarily among ties -- which is exactly what happened in refine_e2.py.

So the gate is calibrated DISTRIBUTIONALLY instead: `trust_lo` is the train
97.5th percentile, i.e. "anything more chromatically mismatched than all but
2.5% of in-distribution shadows is suspect", and `trust_hi` is the train
maximum, i.e. "worse than anything train has ever shown is not trusted at all".

This uses no labels and no test data: only the distribution of a self-computed
signal over the training split. That is the honest way to set a safety
threshold for a failure mode your training data does not contain.

    python experiments/e2/calibrate_trust.py --n 250
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np

from common.imgutil import mask_bool, instances, to_float
from common.istd import load_split
from experiments.e2 import iar as E

RES = os.path.join(HERE, "results")
TUNING = os.path.join(RES, "tuning_train_e2.json")


def chroma_distances(split, n, cfg, seed=0):
    out = []
    for s in random.Random(seed).sample(load_split(split), n):
        img, mask, _ = s.load()
        m = mask_bool(mask)
        if not m.any():
            continue
        f = to_float(img, cfg.linear)
        lum = f.mean(axis=2)
        for inst in instances(m, cfg.min_instance_area):
            info = {}
            E._instance_params(f, lum, inst, m, cfg, info)
            if "chroma_distance" in info:
                out.append(info["chroma_distance"])
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--lo-pct", type=float, default=97.5)
    ap.add_argument("--also-test", action="store_true",
                    help="also report the TEST distribution; reporting only, "
                         "it never influences the chosen thresholds")
    args = ap.parse_args()

    tun = json.load(open(TUNING, encoding="utf-8"))
    cfg = E.IARConfig(**{**tun["chosen_config"], "trust_check": False})

    tr = chroma_distances("train", args.n, cfg)
    pcts = {p: float(np.percentile(tr, p)) for p in (50, 75, 90, 95, 97.5, 99, 100)}
    lo = float(np.percentile(tr, args.lo_pct))
    hi = float(tr.max())
    print(f"TRAIN: {len(tr)} instances")
    print("  " + "  ".join(f"p{p}={v:.3f}" for p, v in pcts.items()))
    print(f"  -> trust_lo = p{args.lo_pct} = {lo:.3f}")
    print(f"  -> trust_hi = max        = {hi:.3f}")
    print(f"  gated on train: {float((tr > lo).mean())*100:.1f}% of instances")

    rec = {"method": "distributional calibration on the TRAIN split only",
           "reason": "the failure mode the gate catches does not occur in train; "
                     "ISTD+ train and test contain disjoint scenes",
           "n_train_images": args.n, "n_train_instances": int(len(tr)),
           "train_percentiles": pcts,
           "lo_percentile": args.lo_pct,
           "trust_lo": round(lo, 3), "trust_hi": round(hi, 3),
           "frac_train_instances_gated": float((tr > lo).mean())}

    if args.also_test:
        te = chroma_distances("test", args.n, cfg)
        rec["test_percentiles_FOR_REPORTING_ONLY"] = {
            p: float(np.percentile(te, p)) for p in (50, 75, 90, 95, 97.5, 99, 100)}
        rec["frac_test_instances_gated"] = float((te > lo).mean())
        print(f"TEST (reporting only): {len(te)} instances")
        print("  " + "  ".join(
            f"p{p}={v:.3f}" for p, v in rec["test_percentiles_FOR_REPORTING_ONLY"].items()))
        print(f"  gated on test: {rec['frac_test_instances_gated']*100:.1f}% of instances")

    tun["trust_calibration"] = rec
    tun["chosen_config"] = {**tun["chosen_config"],
                            "trust_lo": rec["trust_lo"], "trust_hi": rec["trust_hi"]}
    json.dump(tun, open(TUNING, "w", encoding="utf-8"), indent=2)
    print("\nchosen_config ->", tun["chosen_config"])


if __name__ == "__main__":
    main()
