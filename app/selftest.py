"""Sanity check for the PoC: does detect -> remove actually work end to end?

Runs the full pipeline on a few ISTD+ test images two ways -- once with the
ground-truth mask, once with the detector's own mask -- so the two sources of
error are separated. A PoC that merges them tells you nothing about which half
is wrong.

    python app/selftest.py
    python app/selftest.py --n 12 --checkpoint istd
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from app import detector as D
from app.pipeline import remove_shadow
from common import metrics as M
from common.istd import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--checkpoint", default="istd", choices=["istd", "sbu"])
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "silt", "classical"])
    args = ap.parse_args()

    print("backends available:", D.available_backends())

    samples = load_split("test")
    step = max(1, len(samples) // args.n)
    chosen = samples[::step][:args.n]

    print(f"\n{'image':10s} {'input':>7s} {'GT mask':>8s} {'detected':>9s} "
          f"{'IoU':>6s} {'det ms':>7s} {'rm ms':>6s} {'trust':>6s}")
    rows = []
    for s in chosen:
        img, gtmask, gt = s.load()
        out_gt, _, _ = remove_shadow(img, mask_override=gtmask)
        out_d, m_d, info = remove_shadow(img, backend=args.backend,
                                         checkpoint=args.checkpoint)
        p0 = M.compute_metrics(img, gt, gtmask, with_lpips=False)["psnr_S"]
        pg = M.compute_metrics(out_gt, gt, gtmask, with_lpips=False)["psnr_S"]
        pd = M.compute_metrics(out_d, gt, gtmask, with_lpips=False)["psnr_S"]
        g, p = gtmask > 127, m_d > 127
        iou = float((g & p).sum()) / max(float((g | p).sum()), 1.0)
        tr = info.get("e2", {}).get("trust", float("nan"))
        rows.append((p0, pg, pd, iou, info["detect_ms"], info["remove_ms"]))
        print(f"{s.name:10s} {p0:7.2f} {pg:8.2f} {pd:9.2f} {iou:6.3f} "
              f"{info['detect_ms']:7.0f} {info['remove_ms']:6.0f} {tr:6.2f}")

    a = np.array(rows, float)
    print(f"\n{'MEAN':10s} {a[:,0].mean():7.2f} {a[:,1].mean():8.2f} "
          f"{a[:,2].mean():9.2f} {a[:,3].mean():6.3f} "
          f"{a[:,4].mean():7.0f} {a[:,5].mean():6.0f}")
    print(f"\ncost of using a predicted mask instead of the annotation: "
          f"{a[:,2].mean() - a[:,1].mean():+.2f} dB")
    print("detector:", json.dumps(info["detector"]))


if __name__ == "__main__":
    main()
