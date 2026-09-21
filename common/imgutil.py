"""Shared image / mask helpers used by both the baselines and experiment e1."""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------- color space
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """x in [0,1] float -> linear radiance in [0,1]."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


def to_float(img_u8: np.ndarray, linear: bool) -> np.ndarray:
    f = img_u8.astype(np.float32) / 255.0
    return srgb_to_linear(f) if linear else f


def to_u8(f: np.ndarray, linear: bool) -> np.ndarray:
    f = linear_to_srgb(f) if linear else f
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------- masks
def mask_bool(mask_u8: np.ndarray) -> np.ndarray:
    return mask_u8 > 127


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def core_and_outside(m: np.ndarray, erode: int = 7, dilate: int = 7):
    """Umbra core (safely inside) and lit region (safely outside the penumbra)."""
    mu = m.astype(np.uint8)
    core = cv2.erode(mu, disk(erode)) > 0
    outside = cv2.dilate(mu, disk(dilate)) == 0
    return core, outside


def feather(m: np.ndarray, inner: int = 3, width: int = 12) -> np.ndarray:
    """Soft alpha: ~1 in the umbra interior, falling to 0 just outside the mask.

    Built from signed distance to the mask boundary so the ramp straddles the
    penumbra instead of the binary edge (a hard edge is what produces the
    residual shadow contour that learned methods add a refinement stage for).
    """
    mu = (m > 0).astype(np.uint8)
    if mu.sum() == 0:
        return np.zeros(m.shape, np.float32)
    din = cv2.distanceTransform(mu, cv2.DIST_L2, 5)            # >0 inside
    dout = cv2.distanceTransform(1 - mu, cv2.DIST_L2, 5)       # >0 outside
    sd = din - dout                                            # signed distance
    t = np.clip((sd + width * 0.5 - inner) / max(width, 1e-6), 0.0, 1.0)
    # smoothstep: C1 at both ends, so the gain field has no visible kink at the
    # edge of the ramp and needs no post-hoc blur (a blur would leak gain>1 into
    # the lit region and measurably damage non-shadow PSNR)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def instances(m: np.ndarray, min_area: int = 200):
    """Connected shadow instances -> list of boolean masks (largest first)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append(lab == i)
    out.sort(key=lambda x: -x.sum())
    if not out and m.any():
        out = [m.astype(bool)]
    return out


def ring(m: np.ndarray, r_in: int, r_out: int, inside: bool) -> np.ndarray:
    """Band of width (r_out-r_in) just inside or just outside the mask edge."""
    mu = (m > 0).astype(np.uint8)
    if inside:
        return (cv2.erode(mu, disk(r_in)) > 0) & (cv2.erode(mu, disk(r_out)) == 0)
    return (cv2.dilate(mu, disk(r_out)) > 0) & (cv2.dilate(mu, disk(r_in)) == 0)


def apply_gain(img_f: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """gain: HxW or HxWx3 multiplicative field."""
    if gain.ndim == 2:
        gain = gain[:, :, None]
    return img_f * gain
