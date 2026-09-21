"""Training-free shadow-removal baselines for ISTD+.

Every method here is weightless and mask-guided, i.e. it sits in exactly the
same operating regime as the proposed method, so the comparison isolates the
contribution rather than the supervision budget.  Two ground-truth *oracles*
are included and clearly labelled -- they are not baselines, they are ceilings
that say whether the multiplicative-gain family can work at all (guideline 8.4).

Signature for all of them:  fn(img_rgb_u8, mask_u8) -> (pred_rgb_u8, aux|None)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from scipy.fft import dctn, idctn

from common.imgutil import (EPS, core_and_outside, disk, feather, instances,
                            mask_bool, ring, to_float, to_u8)

GAIN_MAX = 8.0          # a cast shadow is a direct/ambient ratio, not a black hole
GAIN_MIN = 1.0          # a shadow can only be brightened


# ------------------------------------------------------------------ 0. no-op
def method_input(img, mask):
    return img.copy(), None


# --------------------------------------------- 1. global per-channel mean gain
def _region_gain(f, src, dst, robust=True):
    """Per-channel multiplicative gain taking `src` statistics to `dst`."""
    g = np.ones(3, np.float64)
    for c in range(3):
        a = f[:, :, c][src]
        b = f[:, :, c][dst]
        if a.size < 32 or b.size < 32:
            continue
        num, den = (np.median(b), np.median(a)) if robust else (b.mean(), a.mean())
        g[c] = float(num / max(den, EPS))
    return np.clip(g, GAIN_MIN, GAIN_MAX)


def make_global_gain(linear=True, robust=True):
    def fn(img, mask):
        m = mask_bool(mask)
        f = to_float(img, linear)
        if not m.any():
            return img.copy(), None
        core, outside = core_and_outside(m)
        if core.sum() < 64:
            core = m
        g = _region_gain(f, core, outside, robust)
        a = feather(m)[:, :, None]
        out = f * (1.0 + a * (g[None, None, :] - 1.0))
        return to_u8(out, linear), {"gain_rgb": g.tolist()}
    return fn


# ----------------------------------------- 2. per-instance boundary-ring gain
def make_boundary_gain(linear=True, band=10, gap=4, f_inner=3, f_width=12):
    """Estimate k from a thin band on either side of each shadow boundary.

    The same-material assumption that any region-pairing method needs is far
    more likely to hold across a few pixels of boundary than across the whole
    image, so ring statistics are a much better illumination estimate than
    global statistics.
    """
    def fn(img, mask):
        m = mask_bool(mask)
        f = to_float(img, linear)
        if not m.any():
            return img.copy(), None
        gain = np.ones(f.shape, np.float32)
        gains = []
        for inst in instances(m):
            r_in = ring(inst, gap, gap + band, inside=True)
            r_out = ring(inst, gap, gap + band, inside=False)
            if r_in.sum() < 64 or r_out.sum() < 64:
                r_in, r_out = core_and_outside(inst)
            g = _region_gain(f, r_in, r_out)
            gains.append(g.tolist())
            a = feather(inst, f_inner, f_width)[:, :, None]
            gain = gain * (1.0 + a * (g[None, None, :] - 1.0))
        return to_u8(f * gain, linear), {"gains_rgb": gains}
    return fn


# ------------------------------- 3. Guo et al. style paired-region relighting
def make_guo_paired(linear=True, n_segments=220, topk=3, smooth=21,
                    w_chroma=1.0, w_tex=0.6, w_dist=0.25):
    """Re-implementation in the spirit of Guo, Dai and Hoiem (paired regions).

    Over-segment -> label segments shadow/lit -> pair each shadow segment with
    the most similar lit segments (illumination-invariant chromaticity + a small
    gradient-texture descriptor + spatial proximity) -> relight each shadow
    segment by the per-channel ratio of its paired lit segments.

    This is the critical baseline: it also pairs shadow and lit regions and it
    also does not train the relighting.  It differs from the proposed method by
    working in pixel space, with no spectral decomposition and no self-check.
    """
    from skimage.segmentation import slic

    fallback = make_boundary_gain(linear)

    def descriptors(f, lab_map, ids, m):
        chroma, tex, cen, area, sfrac = {}, {}, {}, {}, {}
        gray = f.mean(axis=2).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx) % np.pi
        bins = np.minimum((ang / np.pi * 8).astype(np.int32), 7)
        ys, xs = np.mgrid[0:f.shape[0], 0:f.shape[1]].astype(np.float32)
        for i in ids:
            sel = lab_map == i
            n = int(sel.sum())
            if n == 0:
                continue
            mean_rgb = np.array([f[:, :, c][sel].mean() for c in range(3)])
            chroma[i] = mean_rgb / max(mean_rgb.sum(), EPS)   # invariant to scalar k
            h = np.bincount(bins[sel], weights=gm[sel], minlength=8)
            tex[i] = h / max(h.sum(), EPS)
            cen[i] = np.array([ys[sel].mean(), xs[sel].mean()])
            area[i] = n
            sfrac[i] = float(m[sel].mean())
        return chroma, tex, cen, area, sfrac

    def fn(img, mask):
        m = mask_bool(mask)
        f = to_float(img, linear)
        if not m.any():
            return img.copy(), None
        seg = slic(np.clip(f, 0, 1), n_segments=n_segments, compactness=12,
                   start_label=0, channel_axis=2)
        ids = np.unique(seg)
        chroma, tex, cen, area, sfrac = descriptors(f, seg, ids, m)
        sh = [i for i in ids if sfrac.get(i, 0.0) > 0.85 and area.get(i, 0) > 40]
        lit = [i for i in ids if sfrac.get(i, 1.0) < 0.02 and area.get(i, 0) > 40]
        if not sh or not lit:
            return fallback(img, mask)

        diag = float(np.hypot(*f.shape[:2]))
        acc = np.zeros(f.shape, np.float32)
        wsum = np.zeros(f.shape[:2], np.float32)
        pairs = 0
        for i in sh:
            d = []
            for j in lit:
                dc = np.abs(chroma[i] - chroma[j]).sum()
                dt = np.abs(tex[i] - tex[j]).sum()
                dd = float(np.linalg.norm(cen[i] - cen[j])) / diag
                d.append((w_chroma * dc + w_tex * dt + w_dist * dd, j))
            d.sort()
            chosen = [j for _, j in d[:topk]]
            gs = np.ones(3, np.float64)
            sel = seg == i
            for c in range(3):
                src = f[:, :, c][sel].mean()
                dst = float(np.mean([f[:, :, c][seg == j].mean() for j in chosen]))
                gs[c] = dst / max(src, EPS)
            gs = np.clip(gs, GAIN_MIN, GAIN_MAX)
            acc[sel] = gs[None, :]
            wsum[sel] = 1.0
            pairs += 1

        # diffuse segment gains over the whole shadow: segments that were not
        # confidently shadow, plus the penumbra, get an interpolated value
        acc_b = cv2.GaussianBlur(acc, (61, 61), 0)
        w_b = cv2.GaussianBlur(wsum, (61, 61), 0)
        gfield = np.where(w_b[:, :, None] > 1e-3,
                          acc_b / np.maximum(w_b[:, :, None], 1e-3), 1.0)
        gfield = np.clip(gfield, GAIN_MIN, GAIN_MAX).astype(np.float32)
        if smooth:
            s = smooth | 1
            gfield = cv2.GaussianBlur(gfield, (s, s), 0)
        a = feather(m)[:, :, None]
        gain = 1.0 + a * (gfield - 1.0)
        return to_u8(f * gain, linear), {"n_shadow_seg": len(sh),
                                         "n_lit_seg": len(lit), "pairs": pairs}
    return fn


# ------------------------------------- 4. Finlayson-style gradient-domain fill
def _poisson_neumann(div: np.ndarray) -> np.ndarray:
    h, w = div.shape
    dct = dctn(div, type=2, norm="ortho")
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    denom = (2 * np.cos(np.pi * i / h) - 2) + (2 * np.cos(np.pi * j / w) - 2)
    denom[0, 0] = 1.0
    u = dct / denom
    u[0, 0] = 0.0
    return idctn(u, type=2, norm="ortho")


def make_gradient_domain(linear=True, edge_width=7, smooth=0):
    """Log-domain gradient nulling across the shadow edge + Poisson
    reintegration (Finlayson, Hordley and Drew, in spirit).
    """
    def fn(img, mask):
        m = mask_bool(mask)
        f = to_float(img, linear)
        if not m.any():
            return img.copy(), None
        mu = m.astype(np.uint8)
        edge = (cv2.dilate(mu, disk(edge_width)) - cv2.erode(mu, disk(edge_width))) > 0
        L = np.log(np.clip(f, 1e-3, None))
        out = np.empty_like(L)
        for c in range(3):
            u = L[:, :, c]
            gx = np.zeros_like(u)
            gy = np.zeros_like(u)
            gx[:, :-1] = u[:, 1:] - u[:, :-1]
            gy[:-1, :] = u[1:, :] - u[:-1, :]
            gx[edge] = 0.0
            gy[edge] = 0.0
            dxx = np.zeros_like(u)
            dyy = np.zeros_like(u)
            dxx[:, 1:] = gx[:, 1:] - gx[:, :-1]
            dxx[:, 0] = gx[:, 0]
            dyy[1:, :] = gy[1:, :] - gy[:-1, :]
            dyy[0, :] = gy[0, :]
            v = _poisson_neumann(dxx + dyy)
            # Neumann BC fixes the solution only up to a constant: anchor it on
            # the lit region, which the method must leave unchanged.
            v = v + (u[~m].mean() - v[~m].mean())
            out[:, :, c] = v
        res = np.exp(out)
        g = np.clip(res / np.maximum(f, EPS), GAIN_MIN, GAIN_MAX)
        a = feather(m)[:, :, None]
        g = (1.0 + a * (g - 1.0)).astype(np.float32)
        if smooth:
            s = smooth | 1
            g = cv2.GaussianBlur(g, (s, s), 0)
        return to_u8(f * g, linear), None
    return fn


# ----------------------------------------------------------------- 5. oracles
def oracle_region_gain(img, mask, gt, linear=True):
    """Ceiling for any per-instance constant-k method: k taken from the GT."""
    m = mask_bool(mask)
    f = to_float(img, linear)
    t = to_float(gt, linear)
    if not m.any():
        return img.copy(), None
    gain = np.ones(f.shape, np.float32)
    for inst in instances(m):
        core = cv2.erode(inst.astype(np.uint8), disk(5)) > 0
        if core.sum() < 64:
            core = inst
        g = np.array([float(np.median(t[:, :, c][core]) /
                            max(np.median(f[:, :, c][core]), EPS)) for c in range(3)])
        g = np.clip(g, GAIN_MIN, GAIN_MAX)
        a = feather(inst)[:, :, None]
        gain = gain * (1.0 + a * (g[None, None, :] - 1.0))
    return to_u8(f * gain, linear), None


def oracle_smooth_gain(img, mask, gt, linear=True, sigma=15):
    """Ceiling for any *smooth gain field* method: the true per-pixel ratio,
    low-pass filtered.  If this is far from perfect then the gain-field
    formulation itself -- not the estimator -- is the limit.
    """
    m = mask_bool(mask)
    f = to_float(img, linear)
    t = to_float(gt, linear)
    g = np.clip(t / np.maximum(f, 1e-3), GAIN_MIN, GAIN_MAX).astype(np.float32)
    k = int(sigma * 4) | 1
    g = cv2.GaussianBlur(g, (k, k), sigma)
    a = feather(m)[:, :, None]
    g = 1.0 + a * (g - 1.0)
    return to_u8(f * g, linear), None


REGISTRY = {
    "input":                     lambda: method_input,
    "global_gain_srgb":          lambda: make_global_gain(linear=False),
    "global_gain_linear":        lambda: make_global_gain(linear=True),
    "boundary_gain_srgb":        lambda: make_boundary_gain(linear=False),
    "boundary_gain_linear":      lambda: make_boundary_gain(linear=True),
    "guo_paired_regions_srgb":   lambda: make_guo_paired(linear=False),
    "guo_paired_regions_linear": lambda: make_guo_paired(linear=True),
    "gradient_domain_srgb":      lambda: make_gradient_domain(linear=False),
    "gradient_domain_linear":    lambda: make_gradient_domain(linear=True),
}

ORACLES = {
    "oracle_region_gain_linear": lambda img, mask, gt: oracle_region_gain(img, mask, gt, True),
    "oracle_smooth_gain_linear": lambda img, mask, gt: oracle_smooth_gain(img, mask, gt, True),
}
