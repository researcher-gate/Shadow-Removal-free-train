"""detect -> remove, the whole PoC in one function.

    from app.pipeline import remove_shadow
    out, mask, info = remove_shadow(img_rgb)

The remover is e2 (experiments/e2/iar.py) with the configuration selected on
the ISTD+ train split. Nothing here re-tunes it.

A caveat the UI repeats, because it matters for a demo on your own photos:
e2's parameters were chosen on ISTD+, which is 1330+540 images of mostly
single cast shadows on near-uniform ground, shot on one camera. Two of its
choices may not transfer:

  * it fits in sRGB rather than linear radiance, which beat linear by 1.30 dB
    on ISTD+ but is backwards from the physics -- most likely because ISTD+'s
    ground truth was itself built by an affine colour correction in the
    encoded domain (see experiments/e2/RESULTS.md section 6);
  * the self-check thresholds are percentiles of the ISTD+ train distribution.

So: good results on benchmark-like photos, less certain on arbitrary ones.
The pipeline reports its own trust score so the UI can say when it backed off.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from app import detector as D
from experiments.e2.iar import IAR, IARConfig

TUNING = os.path.join(ROOT, "experiments", "e2", "results", "tuning_train_e2.json")
MAX_SIDE = 1600          # beyond this the strips and profile get slow for no gain

_FN = {}


def e2_config(**over) -> IARConfig:
    kw = {}
    if os.path.exists(TUNING):
        kw = dict(json.load(open(TUNING, encoding="utf-8"))["chosen_config"])
    kw.update(over)
    return IARConfig(**kw)


def _remover(cfg: IARConfig):
    key = repr(cfg)
    if key not in _FN:
        _FN[key] = IAR(cfg)
    return _FN[key]


def _downscale(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA), s


def remove_shadow(img_rgb: np.ndarray, backend: str = "auto",
                  checkpoint: str = "istd", threshold: float = 0.7,
                  sensitivity: float = 0.5, mask_override: np.ndarray | None = None,
                  dilate: int = 0, trust_check: bool = True):
    """Returns (output RGB uint8, mask uint8, info dict).

    `mask_override` lets the UI hand in a user-drawn or uploaded mask and skip
    detection entirely -- which is also the honest way to show what e2 does
    when the mask is correct, separately from how well the detector works.
    """
    img_rgb = np.ascontiguousarray(img_rgb[:, :, :3])
    small, scale = _downscale(img_rgb)
    info = {"input_size": list(img_rgb.shape[:2]), "work_scale": round(scale, 4)}

    t0 = time.perf_counter()
    if mask_override is not None:
        m = mask_override
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape[:2] != small.shape[:2]:
            m = cv2.resize(m, (small.shape[1], small.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mask = np.where(m > 127, 255, 0).astype(np.uint8)
        info["detector"] = {"backend": "user-supplied mask", "n_parameters": 0,
                            "shadow_area_frac": float((mask > 127).mean())}
    else:
        mask, dmeta = D.detect(small, backend=backend, which=checkpoint,
                               threshold=threshold, sensitivity=sensitivity)
        info["detector"] = dmeta
    t1 = time.perf_counter()

    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * abs(dilate) + 1, 2 * abs(dilate) + 1))
        mask = cv2.dilate(mask, k) if dilate > 0 else cv2.erode(mask, k)

    if (mask > 127).sum() < 200:
        info.update({"detect_ms": (t1 - t0) * 1000, "remove_ms": 0.0,
                     "total_ms": (t1 - t0) * 1000,
                     "note": "no shadow region found; image returned unchanged"})
        return img_rgb.copy(), np.zeros(img_rgb.shape[:2], np.uint8), info

    fn = _remover(e2_config(trust_check=trust_check))
    t2 = time.perf_counter()
    out_small, aux = fn(small, mask)
    t3 = time.perf_counter()

    if scale < 1.0:
        # upsample the correction, not the pixels: the gain field is smooth by
        # construction, so this keeps the original image's full detail
        h, w = img_rgb.shape[:2]
        ratio = (out_small.astype(np.float32) + 1.0) / (small.astype(np.float32) + 1.0)
        ratio = cv2.resize(ratio, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(img_rgb.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        out, mask_full = out_small, mask

    info.update({
        "detect_ms": (t1 - t0) * 1000,
        "remove_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "e2": {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in (aux or {}).items()
               if k in ("n_instances", "penumbra_px", "trust", "trust_min",
                        "chroma_distance", "measured_matte_frac", "a_rgb",
                        "b_rgb", "shadow_area_frac", "degenerate")},
    })
    return out, mask_full, info


def overlay(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45):
    """Mask drawn over the image, for the UI."""
    m = (mask > 127)
    out = img_rgb.astype(np.float32).copy()
    tint = np.array([255.0, 64.0, 64.0], np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * tint
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0
    out[edge] = tint
    return out.astype(np.uint8)
