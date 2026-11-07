"""Shadow mask prediction, run locally.

e2 is mask-guided: it needs to be told where the shadow is. In the evaluation
(RESULT.md) that mask is the ISTD+ ground-truth annotation. For a PoC on the
user's own photos there is no annotation, so this module supplies one.

Two backends, in order of preference:

  silt       SILT (Hu et al., ICCV 2023), PVT-v2-B3, 49 M params. The official
             checkpoint, run on CPU. Reported BER 1.11 on ISTD, 4.36 on SBU.
             Weights: third_party/SILT/pretrained/*.pth  (198 MB each)

  classical  A weightless fallback so the app still runs with no download.
             Chromaticity-and-luminance thresholding in LAB with a
             ratio-map refinement. Much worse than SILT -- it exists so the
             PoC degrades instead of failing, and the UI says which ran.

Both return a float mask in [0,1] at the input image's resolution.

NOTE ON THE CHECKPOINTS
  The ISTD checkpoint is trained on the same dataset family e2 was tuned on,
  so the two agree about what counts as a shadow. The SBU checkpoint is
  trained on far more varied real-world scenes and generalises better to
  photos from a phone. For a PoC on arbitrary user images prefer SBU; for
  anything resembling the benchmark prefer ISTD. The UI exposes the choice.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILT_DIR = os.path.join(ROOT, "third_party", "SILT")
CKPTS = {
    "istd": os.path.join(SILT_DIR, "pretrained", "silt_pvtb3_istd.pth"),
    "sbu": os.path.join(SILT_DIR, "pretrained", "silt_pvtb5_sbu.pth"),
}

_CACHE: dict[str, object] = {}


# --------------------------------------------------------------- SILT backend


def silt_available(which: str = "istd") -> bool:
    return os.path.exists(CKPTS.get(which, ""))


def _load_silt(which: str):
    key = f"silt::{which}"
    if key in _CACHE:
        return _CACHE[key]
    import torch

    path = CKPTS[which]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run app/download_detector.sh")
    if SILT_DIR not in sys.path:
        sys.path.insert(0, SILT_DIR)
    from model.network import Network

    # weights_only=True refuses to execute pickle code while unpickling. This
    # checkpoint is a 198 MB file fetched from a third-party Google Drive link,
    # and torch.load's default (weights_only=False) would run whatever that
    # pickle says to. Tensors and plain dicts load fine under the safe reader;
    # if this file needs more than that, it is asking to run code and the
    # honest response is to stop and say so rather than silently allow it.
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            os.path.basename(path)
            + f" could not be loaded in safe mode ({type(e).__name__}: {e}).\n"
            "Loading it would require torch.load(weights_only=False), which "
            "executes arbitrary code from the downloaded file. Convert it once, "
            "deliberately, with:\n"
            "    python app/convert_checkpoint.py --i-trust-this-file\n"
            "which re-saves only the tensors in a safe format."
        ) from e
    cfg = ck["configs"]["training_config"]
    res = int(cfg["data"]["resolution"])
    net = Network(input_resolution=res, backbone=cfg["backbone"])
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(os.cpu_count() or 4)
    meta = {"backend": "silt", "checkpoint": which, "backbone": cfg["backbone"],
            "resolution": res, "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "n_parameters": int(sum(p.numel() for p in net.parameters()))}
    _CACHE[key] = (net, res, meta)
    return _CACHE[key]


def detect_silt(img_rgb: np.ndarray, which: str = "istd", threshold: float = 0.7):
    """Returns (soft mask in [0,1] at the image's size, meta dict)."""
    import torch
    net, res, meta = _load_silt(which)
    h, w = img_rgb.shape[:2]
    x = cv2.resize(img_rgb, (res, res), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(
        x.astype(np.float32).transpose(2, 0, 1) / 255.0))[None]
    with torch.no_grad():
        p = net(t)                       # (1,1,res,res), already sigmoid
    p = p[0, 0].numpy()
    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(p, 0.0, 1.0), {**meta, "threshold": threshold}


# ----------------------------------------------------- classical fallback


def detect_classical(img_rgb: np.ndarray, sensitivity: float = 0.5):
    """Weightless shadow detector. Here so the app runs with no download.

    Shadowed pixels are darker AND relatively bluer than their lit
    counterparts (ambient skylight dominates in shadow), so a combination of
    low L* and high b*-deficit separates them better than brightness alone.
    Crude, and it is meant to be: the UI labels any result produced this way.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    B = lab[:, :, 2] - 128.0

    # normalised "shadowness": dark, and blue-shifted relative to the image
    Ln = (np.median(L) - L) / max(L.std(), 1e-3)
    Bn = (np.median(B) - B) / max(B.std(), 1e-3)
    s = 0.75 * Ln + 0.25 * Bn
    s = cv2.GaussianBlur(s, (0, 0), 3.0)

    thr = np.quantile(s, 1.0 - 0.45 * sensitivity)
    m = (s > max(thr, 0.35)).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # drop specks: they are noise, and e2 skips instances under 200 px anyway
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(400, 0.0005 * m.size):
            keep[lab_ == i] = 1
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)
    return np.clip(soft, 0, 1), {"backend": "classical", "n_parameters": 0,
                                 "sensitivity": sensitivity}


# ------------------------------------------------------------------ frontend


def detect(img_rgb: np.ndarray, backend: str = "auto", which: str = "istd",
           threshold: float = 0.7, sensitivity: float = 0.5):
    """backend: 'auto' | 'silt' | 'classical'.  Returns (binary u8 mask, meta)."""
    if backend == "auto":
        backend = "silt" if silt_available(which) else "classical"
    if backend == "silt":
        soft, meta = detect_silt(img_rgb, which, threshold)
        hard = (soft > threshold).astype(np.uint8) * 255
    else:
        soft, meta = detect_classical(img_rgb, sensitivity)
        hard = (soft > 0.5).astype(np.uint8) * 255
    meta["shadow_area_frac"] = float((hard > 127).mean())
    return hard, meta


def available_backends() -> dict:
    return {"silt_istd": silt_available("istd"),
            "silt_sbu": silt_available("sbu"),
            "classical": True}
