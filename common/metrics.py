"""Shadow-removal metrics, ISTD+ community protocol + honest extras.

Conventions (documented because the literature is inconsistent about them):

* Evaluation resolution.  Almost every ISTD/ISTD+ paper resizes prediction and
  ground truth to 256x256 before measuring.  We report that ("@256") as the
  headline and also the native 480x640 numbers ("@full"), which are strictly
  harder and never reported -- reporting both makes the comparison to published
  tables valid without hiding what resizing buys.

* "RMSE".  The number printed as RMSE in the shadow-removal literature is a
  *mean absolute error* in CIE L*a*b*.  The authors of the evaluation code say
  so themselves (cvlab-stonybrook/SID README: "the RMSE evaluation code that
  many papers have been using is actually calculating Mean Absolute Error").
  We therefore report `mae_lab` (= the literature's "RMSE", comparable to
  published tables) AND the true `rmse_lab` / `rmse_rgb`.

* Regions.  S = shadow region (mask>127), NS = non-shadow region, ALL = whole
  image.  Region PSNR uses only the pixels of that region (all 3 channels).
  Region SSIM averages the SSIM map inside the region.

* L*a*b*.  skimage convention, L in [0,100] -- the same as MATLAB rgb2lab used
  by the original evaluation scripts.  (cv2's LAB with L in [0,255] would make
  every number ~2.55x larger.)
"""
from __future__ import annotations

import numpy as np
import cv2
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab

EVAL_SIZE = (256, 256)  # (w, h)

_LPIPS_NET = None


def _get_lpips():
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import torch, lpips  # noqa
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad_(False)
    return _LPIPS_NET


def _resize(img, size, nearest=False):
    if (img.shape[1], img.shape[0]) == size:
        return img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def _region_psnr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """a,b float64 in [0,255]; m bool HxW.  MAX=255."""
    if m.sum() == 0:
        return float("nan")
    d = (a - b)[m]  # (N,3)
    mse = float(np.mean(d * d))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel SSIM map (Wang et al. defaults), averaged over channels."""
    _, smap = structural_similarity(
        a, b, data_range=255.0, channel_axis=2, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return smap.mean(axis=2)


def _masked_mean(x: np.ndarray, m: np.ndarray) -> float:
    return float(x[m].mean()) if m.sum() else float("nan")


def compute_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray, mask_u8: np.ndarray,
                    at_size=EVAL_SIZE, with_lpips=True) -> dict:
    """pred/gt: HxWx3 uint8 RGB.  mask: HxW uint8.  Returns flat dict."""
    if at_size is not None:
        pred_u8 = _resize(pred_u8, at_size)
        gt_u8 = _resize(gt_u8, at_size)
        mask_u8 = _resize(mask_u8, at_size, nearest=True)

    p = pred_u8.astype(np.float64)
    g = gt_u8.astype(np.float64)
    ms = mask_u8 > 127
    mn = ~ms
    ma = np.ones_like(ms)

    smap = _ssim_map(pred_u8.astype(np.float64), gt_u8.astype(np.float64))

    p_lab = rgb2lab(pred_u8.astype(np.float32) / 255.0)
    g_lab = rgb2lab(gt_u8.astype(np.float32) / 255.0)
    dlab = np.abs(p_lab - g_lab)              # per-channel |.|
    dlab2 = (p_lab - g_lab) ** 2
    drgb2 = (p - g) ** 2

    out = {}
    for tag, m in (("S", ms), ("NS", mn), ("ALL", ma)):
        out[f"psnr_{tag}"] = _region_psnr(p, g, m)
        out[f"ssim_{tag}"] = _masked_mean(smap, m)
        # literature "RMSE" == MAE in Lab, averaged over the 3 Lab channels
        out[f"mae_lab_{tag}"] = _masked_mean(dlab.mean(axis=2), m)
        out[f"rmse_lab_{tag}"] = float(np.sqrt(np.mean(dlab2[m]))) if m.sum() else float("nan")
        out[f"rmse_rgb_{tag}"] = float(np.sqrt(np.mean(drgb2[m]))) if m.sum() else float("nan")
        out[f"pixels_{tag}"] = int(m.sum())

    if with_lpips:
        import torch
        net = _get_lpips()
        def t(x):
            z = torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None]) / 127.5 - 1.0
            return z
        with torch.no_grad():
            out["lpips_ALL"] = float(net(t(pred_u8), t(gt_u8)).item())
    return out


def aggregate(per_image: list[dict]) -> dict:
    """Mean over images of each scalar metric (the protocol used in the
    literature: per-image metric, then unweighted average over the test set)."""
    keys = [k for k in per_image[0]
            if not k.startswith("pixels_") and isinstance(per_image[0][k], (int, float))]
    agg = {}
    for k in keys:
        v = np.array([d[k] for d in per_image], dtype=np.float64)
        v = v[np.isfinite(v)]
        agg[k] = float(v.mean()) if v.size else float("nan")
    return agg
