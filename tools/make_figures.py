"""Qualitative result images: side-by-side panels with per-image PSNR burned in.

    python tools/make_figures.py                     # best / median / worst by e1 PSNR_S
    python tools/make_figures.py --names 91-1 103-5  # specific images
    python tools/make_figures.py --n 12              # more rows

Writes one PNG per selected image to figures/, plus a contact sheet.

Columns: input (shadow) | mask | each method's output | ground truth.
Each panel is labelled with that method's PSNR over the shadow region for THAT
image, computed with the same 256x256 protocol as the tables, so a panel and a
table row can never disagree.

Methods are read from whatever prediction directories exist under
baseline/outputs/ and experiments/*/outputs/ -- so a method only appears here if
it was run with --save-images (baseline) or is the e1 main run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from common import metrics as M
from common.istd import load_split

FIGDIR = os.path.join(ROOT, "figures")
LABEL_H = 26
PAD = 6


def discover(order_hint=("input",)):
    """{label: dir of per-image PNG predictions}, in a sensible column order."""
    found = {}
    for pat, prefix in ((os.path.join(ROOT, "baseline", "outputs", "*"), ""),
                        (os.path.join(ROOT, "experiments", "*", "outputs", "*"), "")):
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png")):
                found[prefix + os.path.basename(d)] = d
    # e1 last so it reads left-to-right as baselines -> ours -> GT
    keys = sorted(found, key=lambda k: (k.startswith("e1"), k))
    return {k: found[k] for k in keys}


def put_label(img, text, color=(255, 255, 255), bg=(32, 32, 32)):
    h, w = img.shape[:2]
    out = np.full((h + LABEL_H, w, 3), bg, np.uint8)
    out[LABEL_H:] = img
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    while tw > w - 8 and scale > 0.25:
        scale -= 0.03
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(out, text, (4, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)
    return out


def row_for(sample, preds, scale=0.5):
    img, mask, gt = sample.load()
    cells = []

    m0 = M.compute_metrics(img, gt, mask, with_lpips=False)
    cells.append((f"input  PSNR-S {m0['psnr_S']:.2f} dB", img))
    cells.append(("shadow mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)))

    for label, d in preds.items():
        p = os.path.join(d, sample.name + ".png")
        if not os.path.exists(p):
            continue
        pred = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
        m = M.compute_metrics(pred, gt, mask, with_lpips=False)
        cells.append((f"{label}  PSNR-S {m['psnr_S']:.2f} dB", pred))

    cells.append(("ground truth (ISTD+ adjusted)", gt))

    out = []
    for text, c in cells:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(put_label(c, text))
    h = max(c.shape[0] for c in out)
    strip = []
    for c in out:
        if c.shape[0] != h:
            c = np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0)))
        strip.append(c)
        strip.append(np.full((h, PAD, 3), 255, np.uint8))
    return np.hstack(strip[:-1])


def pick(names, n, preds, rank_by=None):
    """Best / median / worst thirds by the ranking method's shadow PSNR.

    Showing only the first n images by filename hides exactly the cases a
    reader needs: the failures. Ranking by a real per-image score and sampling
    from both ends is the honest selection.
    """
    samples = {s.name: s for s in load_split("test")}
    if names:
        return [samples[x] for x in names if x in samples]
    candidates = [rank_by] if rank_by else [
        os.path.join(ROOT, "experiments", "e2", "results", "e2_main.json"),
        os.path.join(ROOT, "experiments", "e1", "results", "e1_main.json"),
        os.path.join(ROOT, "baseline", "results", "boundary_gain.json"),
    ]
    p = next((c for c in candidates if os.path.exists(c)), None)
    if p is None:
        return list(samples.values())[:n]
    per = json.load(open(p, encoding="utf-8"))["per_image"]
    per = [r for r in per if np.isfinite(r["psnr_S"])]
    per.sort(key=lambda r: r["psnr_S"])
    k = max(1, n // 3)
    chosen = per[:k] + per[len(per) // 2 - k // 2: len(per) // 2 - k // 2 + k] + per[-k:]
    seen, out = set(), []
    for r in chosen:
        if r["name"] not in seen and r["name"] in samples:
            seen.add(r["name"]); out.append(samples[r["name"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rank-by", default=None,
                    help="result JSON whose per-image psnr_S orders the selection")
    args = ap.parse_args()

    preds = discover()
    if not preds:
        print("no prediction directories found. Run with --save-images "
              "(baseline) or run experiments/e1/run_e1.py --stages main first.")
        return 1
    print("columns:", ", ".join(["input", "mask"] + list(preds) + ["ground truth"]))

    os.makedirs(FIGDIR, exist_ok=True)
    chosen = pick(args.names, args.n, preds, args.rank_by)
    rows = []
    for s in chosen:
        r = row_for(s, preds, args.scale)
        cv2.imwrite(os.path.join(FIGDIR, f"compare_{s.name}.png"), r[:, :, ::-1])
        rows.append(r)
        print(f"  figures/compare_{s.name}.png")

    w = max(r.shape[1] for r in rows)
    sheet = []
    for r in rows:
        if r.shape[1] != w:
            r = np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)
        sheet.append(r)
        sheet.append(np.full((PAD * 2, w, 3), 255, np.uint8))
    out = np.vstack(sheet[:-1])
    p = os.path.join(FIGDIR, "contact_sheet.png")
    cv2.imwrite(p, out[:, :, ::-1])
    print(f"  {os.path.relpath(p, ROOT)}  ({out.shape[1]}x{out.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
