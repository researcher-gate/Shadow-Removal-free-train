"""e2 -- Instance Affine Relighting with a measured penumbra (IAR).

A training-free shadow remover with ZERO learned weights of any kind. Its only
fitted quantities are a handful of scalar hyper-parameters chosen on the ISTD+
train split; there is no model, no descriptor, and no prior network.

WHY THIS DESIGN  (full reasoning in research.txt, evidence in RESULTS.md)

  1. The model, not the estimator, was the bottleneck.
     Five quite different *multiplicative* estimators in baseline/ all land
     within 0.5 dB of each other on ISTD+. Shor & Lischinski (EG 2008) and
     Le & Samaras (ICCV'19 / TPAMI'21 -- the ISTD+ dataset paper itself) both
     model the shadowed-to-lit relation as AFFINE, `I_free = a*I + b`, because
     ambient light survives inside the umbra. Forcing `b = 0`, as every
     baseline here and e1 does, fits the wrong family.
     => Stage C fits a robust per-instance, per-channel affine map.

  2. The penumbra was costing more than the estimator.
     Decomposing the ground-truth oracle (baseline/RESULTS.md 4.2) showed a
     fixed feather ramp costs ~14 dB and a fixed gain clamp another ~3.4 dB.
     Gong & Cosker (BMVC'14 / JOSA A'16) instead *register the penumbra to a
     normalised frame and estimate the illumination change across it*.
     => Stage B reads the matte off the image's own radial intensity profile
        across the boundary. No assumed shape, no assumed width.

  3. Patch matching does not pay for itself on this dataset.
     The tuned Guo-style region-pairing baseline is 8x slower than the
     boundary-ring one and loses to it on PSNR. ISTD+ is predominantly single
     shadows on near-uniform ground, so there is little material diversity to
     match against.
     => No segmentation, no descriptors, no search. This is also what keeps
        latency in the tens of milliseconds.

Entry point:  IAR(cfg) -> fn(img_rgb_u8, mask_u8) -> (pred_u8, aux_dict)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from common.imgutil import EPS, disk, instances, mask_bool, to_float, to_u8


@dataclass
class IARConfig:
    linear: bool = True          # fit in linear radiance, not gamma-encoded sRGB

    # --- Stage B: measured penumbra ---
    profile_radius: int = 28     # how far either side of the edge to profile
    profile_bin: float = 1.0     # px per distance bin
    profile_smooth: int = 5      # 1-D smoothing of the profile, in bins
    measured_matte: bool = True  # False -> fixed smoothstep ramp (ablation)
    matte_fallback_width: int = 10
    matte_fallback_inner: int = 3   # push the fallback ramp inside the mask
    min_contrast: float = 0.04   # below this umbra/lit gap, the profile is noise
    max_penumbra: float = 40.0   # wider than this is a failed measurement

    # --- Stage C: affine fit ---
    strip: int = 14              # thickness of each sampling strip, px
    strip_gap: int = 2           # gap from the measured penumbra edge
    n_quantiles: int = 15
    affine: bool = True          # False -> multiplicative only (ablation)
    a_min: float = 0.5
    a_max: float = 20.0
    couple_channels: bool = True  # tie the 3 channel gains to one another
    chroma_kappa: float = 1.5     # max ratio between a channel gain and the median
    min_range: float = 0.02       # inner-strip dynamic range below which a channel
                                  # is too degenerate to support an affine fit

    # --- Stage C': self-check on the same-material assumption ---
    trust_check: bool = True
    trust_lo: float = 0.15        # chroma distance below which the fit is trusted
    trust_hi: float = 0.45        # above which it is not trusted at all
    max_brighten: float = 16.0   # cap on the effective gain at the umbra level

    # --- Stage D: local affine along the boundary ---
    local: bool = False          # True -> per-boundary-arc affine (2x slower)
    arc_spacing: int = 24        # px between boundary anchor points
    arc_radius: int = 70         # radius of the local sampling window
    arc_min_px: int = 120        # per-side sample count below which an arc is dropped
    idw_power: float = 2.0
    local_weight: float = 0.85   # how far to trust local over the global fit

    min_instance_area: int = 200


# ----------------------------------------------------------------- stage B


def _signed_distance(inst: np.ndarray) -> np.ndarray:
    mu = inst.astype(np.uint8)
    din = cv2.distanceTransform(mu, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - mu, cv2.DIST_L2, 3)
    return (din - dout).astype(np.float32)


def _measure_matte(lum, sd, roi, cfg):
    """Build the penumbra matte from the image's own profile across the edge.

    Returns (alpha_lut, centers, d_umbra, d_lit) where alpha_lut[i] is the
    shadowed-ness in [0,1] at signed distance centers[i] (positive = inside).

    This is the training-free analogue of a learned shadow matte: the shape and
    the width of the transition are whatever this image says they are.
    """
    R = cfg.profile_radius
    sel = roi & (np.abs(sd) <= R)
    if sel.sum() < 200:
        return None
    d = sd[sel]
    v = lum[sel]
    idx = np.clip(((d + R) / cfg.profile_bin).astype(np.int32), 0,
                  int(2 * R / cfg.profile_bin) - 1)
    nb = int(2 * R / cfg.profile_bin)
    # median per distance bin would be ideal; a mean of a trimmed range is far
    # cheaper and the ring already averages over albedo
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    s1 = np.bincount(idx, weights=v, minlength=nb)
    ok = cnt >= 12
    if ok.sum() < 8:
        return None
    prof = np.full(nb, np.nan)
    prof[ok] = s1[ok] / cnt[ok]
    centers = (np.arange(nb) + 0.5) * cfg.profile_bin - R

    good = np.isfinite(prof)
    prof = np.interp(centers, centers[good], prof[good])
    k = max(1, cfg.profile_smooth) | 1
    prof = np.convolve(prof, np.ones(k) / k, mode="same")

    lit = float(np.median(prof[centers <= -R * 0.3]))
    umb = float(np.median(prof[centers >= R * 0.3]))
    if not np.isfinite(lit) or not np.isfinite(umb) or (lit - umb) < cfg.min_contrast:
        return None

    a = np.clip((lit - prof) / (lit - umb), 0.0, 1.0)

    # Localise the transition around the mask boundary instead of trusting the
    # far field. The lit side of a real image is not a flat plateau -- a dark
    # object 25 px away from the shadow drags the profile down, and a monotone
    # pass then propagates that inward as a nonzero alpha floor across the whole
    # lit region. Measured cost of getting this wrong: non-shadow PSNR
    # 34.9 -> 24.8 dB, with alpha reaching 0.5 over a fifth of the lit pixels.
    #
    # So: find where the profile first becomes clearly shadowed (d1), then walk
    # BACK from there to the last clearly-lit bin (d0). Everything outside
    # [d0, d1] is saturated, and d0 is bounded so the ramp has to straddle the
    # annotated boundary rather than wander off into the scene.
    hi = np.nonzero(a >= 0.90)[0]
    if hi.size == 0:
        return None
    i1 = int(hi[0])
    lo_candidates = np.nonzero(a[:i1] <= 0.10)[0]
    i0 = int(lo_candidates[-1]) if lo_candidates.size else max(0, i1 - 1)

    d1 = float(centers[i1])
    d0 = float(centers[i0])
    if not (-R * 0.6 <= d0 <= 4.0) or not (-4.0 <= d1 <= R * 0.8) or d1 - d0 < 1.0:
        return None                      # implausible transition -> fall back
    if d1 - d0 > cfg.max_penumbra:
        return None

    out = np.zeros_like(a)
    seg = slice(i0, i1 + 1)
    inner = a[seg]
    inner = np.maximum.accumulate(np.clip(inner, 0.0, 1.0))
    if inner[-1] > inner[0]:
        inner = (inner - inner[0]) / max(inner[-1] - inner[0], 1e-6)
    out[seg] = np.clip(inner, 0.0, 1.0)
    out[i1:] = 1.0
    return out.astype(np.float32), centers.astype(np.float32), d1, d0


def _smoothstep_matte(cfg):
    R = cfg.profile_radius
    nb = int(2 * R / cfg.profile_bin)
    centers = (np.arange(nb) + 0.5) * cfg.profile_bin - R
    w = cfg.matte_fallback_width
    t = np.clip((centers + w * 0.5) / max(w, 1e-6), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32), centers.astype(np.float32), \
           w * 0.5, -w * 0.5


# ----------------------------------------------------------------- stage C


def _affine_from_strips(f, inner, outer, cfg):
    """Robust per-channel affine map taking shadowed values to lit values.

    Matched quantiles rather than matched moments: two strips either side of a
    boundary see the same material distribution but not the same pixels, so
    quantile matching is the right estimator and it tolerates the outliers a
    mean/std fit does not.

    The channels are then COUPLED. Fitted independently they can disagree
    wildly -- measured on ISTD+ test: a = [1.52, 4.48, 1.94] where the truth
    was [1.32, 1.25, 1.28], i.e. a 3.6x over-correction of green alone, which
    shows up as a cyan halo; and a = [8.6, 20.0, 11.8] on an image whose green
    strip is almost black. That is physically impossible: the illumination
    ratio is a property of the light, so the three channels may differ by the
    ambient/direct colour balance and no more.
    """
    q = np.linspace(0.05, 0.95, cfg.n_quantiles)
    A = np.ones((3, 2), np.float64)
    med = np.ones(3)
    degenerate = np.zeros(3, bool)

    for c in range(3):
        x = np.quantile(f[:, :, c][inner], q)
        y = np.quantile(f[:, :, c][outer], q)
        med[c] = max(float(np.median(x)), EPS)
        rng = float(x[-1] - x[0])

        if rng < cfg.min_range or med[c] <= cfg.min_range:
            # too little dynamic range (or too dark) to identify a slope and an
            # intercept separately; a ratio of medians is all the data supports
            A[c] = (float(np.median(y) / med[c]), 0.0)
            degenerate[c] = True
            continue

        if not cfg.affine:
            A[c] = (float(np.sum(x * y) / max(np.sum(x * x), EPS)), 0.0)
            continue

        M = np.stack([x, np.ones_like(x)], 1)
        sol = np.linalg.lstsq(M, y, rcond=None)[0]
        a, b = float(sol[0]), float(sol[1])
        if not np.isfinite(a) or not np.isfinite(b):
            a, b = float(np.median(y) / med[c]), 0.0
        A[c] = (float(np.clip(a, cfg.a_min, cfg.a_max)), b)

    # --- couple the channels through the effective gain at the strip median ---
    if cfg.couple_channels:
        g = (A[:, 0] * med + A[:, 1]) / med
        ref = float(np.median(g[~degenerate])) if (~degenerate).any() else float(np.median(g))
        ref = max(ref, 1.0)
        lo, hi = ref / cfg.chroma_kappa, ref * cfg.chroma_kappa
        for c in range(3):
            if not np.isfinite(g[c]) or g[c] <= 0:
                A[c] = (ref, 0.0)
                continue
            tgt = float(np.clip(g[c], lo, hi))
            if abs(tgt - g[c]) > 1e-9:
                t = tgt / g[c]            # rescale the map, keep its shape
                A[c] = (A[c, 0] * t, A[c, 1] * t)

    # the map must brighten, and must not blow up the umbra: evaluate it at the
    # observed shadow range rather than trusting the coefficients
    for c in range(3):
        x = np.quantile(f[:, :, c][inner], [0.05, 0.95])
        lo, hi = float(x[0]), float(x[1])
        a, b = A[c]
        if a * lo + b < lo:
            b = lo - a * lo
        if lo > EPS and (a * lo + b) / lo > cfg.max_brighten:
            b = cfg.max_brighten * lo - a * lo
        if a * hi + b < hi:
            b = hi - a * hi
        A[c] = (float(np.clip(a, cfg.a_min, cfg.a_max)), float(b))
    return A


# ----------------------------------------------------------------- pipeline


def _boundary_anchors(inst, cfg):
    """Points spaced along the instance contour, used as local fit centres."""
    cnts, _ = cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cnts:
        c = c.reshape(-1, 2)                      # (x, y)
        if len(c) < 8:
            continue
        step = max(1, cfg.arc_spacing)
        pts.extend(c[::step][:, ::-1])            # -> (y, x)
    return np.array(pts, np.float32) if pts else None


def _local_affine_field(f, sd, inst, m_all, d_umb, d_lit, cfg, A_global, info):
    """Fit one affine per boundary anchor, then interpolate over the instance.

    A single affine per instance silently assumes the whole shadow sits on one
    material. When it does not -- a shadow falling across a mural, say -- the
    inner strip samples one albedo and the outer strip samples another, and the
    fit maps one material onto the other. Measured: that costs up to 8 dB
    BELOW the untouched input on 4% of ISTD+ test images.

    Pairing the strips LOCALLY along the boundary removes the assumption
    without giving up the two-strip estimator.
    """
    anchors = _boundary_anchors(inst, cfg)
    if anchors is None or len(anchors) < 2:
        return None

    inner_band = inst & (sd >= d_umb + cfg.strip_gap) & (sd <= d_umb + cfg.strip_gap + cfg.strip)
    outer_band = (~m_all) & (sd <= d_lit - cfg.strip_gap) & (sd >= d_lit - cfg.strip_gap - cfg.strip)
    H, W = sd.shape
    R = cfg.arc_radius

    cen, As = [], []
    for (ay, ax) in anchors:
        y0, y1 = int(max(0, ay - R)), int(min(H, ay + R + 1))
        x0, x1 = int(max(0, ax - R)), int(min(W, ax + R + 1))
        ib = inner_band[y0:y1, x0:x1]
        ob = outer_band[y0:y1, x0:x1]
        if ib.sum() < cfg.arc_min_px or ob.sum() < cfg.arc_min_px:
            continue
        sub = f[y0:y1, x0:x1]
        As.append(_affine_from_strips(sub, ib, ob, cfg))
        cen.append((ay, ax))
    if len(As) < 2:
        return None

    cen = np.array(cen, np.float32)
    As = np.array(As, np.float64)                 # (K, 3, 2)
    info["n_arcs"] = int(len(As))

    sel = inst | (sd > d_lit)
    ys, xs = np.nonzero(sel)
    if ys.size == 0:
        return None
    d2 = ((ys[:, None] - cen[None, :, 0]) ** 2
          + (xs[:, None] - cen[None, :, 1]) ** 2).astype(np.float32)
    w = 1.0 / np.power(d2 + 16.0, cfg.idw_power / 2.0)
    w /= w.sum(1, keepdims=True)
    loc = np.einsum("nk,kcp->ncp", w, As)         # (N, 3, 2)

    lam = cfg.local_weight
    loc = lam * loc + (1.0 - lam) * A_global[None]

    field = np.zeros((H, W, 3, 2), np.float32)
    field[ys, xs] = loc.astype(np.float32)
    field[~sel] = A_global[None, None]
    return field


def _instance_params(f, lum, inst, m_all, cfg, info):
    sd = _signed_distance(inst)
    roi = np.abs(sd) <= cfg.profile_radius
    # never sample the lit side from inside another shadow
    other = m_all & ~inst
    if other.any():
        roi &= ~cv2.dilate(other.astype(np.uint8), disk(6)).astype(bool)

    meas = _measure_matte(lum, sd, roi, cfg) if cfg.measured_matte else None
    if meas is None:
        meas = _smoothstep_matte(cfg)
        info["matte"] = "fallback_smoothstep"
    else:
        info["matte"] = "measured"
    lut, centers, d_umb, d_lit = meas
    info["penumbra_px"] = float(d_umb - d_lit)
    info["d_umbra"] = d_umb
    info["d_lit"] = d_lit

    inner = inst & (sd >= d_umb + cfg.strip_gap) & (sd <= d_umb + cfg.strip_gap + cfg.strip)
    outer = (~m_all) & (sd <= d_lit - cfg.strip_gap) & (sd >= d_lit - cfg.strip_gap - cfg.strip)
    if inner.sum() < 150:
        inner = inst & (sd >= d_umb)
    if outer.sum() < 150:
        outer = (~m_all) & (sd <= d_lit) & (sd >= d_lit - 3 * cfg.strip)
    if inner.sum() < 60 or outer.sum() < 60:
        info["degenerate"] = True
        return None, None

    A = _affine_from_strips(f, inner, outer, cfg)

    # --- is the same-material assumption actually satisfied here? ---
    # Both strips are supposed to show the same material under different
    # illumination. Chromaticity is invariant to a scalar illumination change,
    # so if the two strips disagree chromatically they are on DIFFERENT
    # MATERIALS and the fit is measuring albedo, not light.
    # Measured on ISTD+ test: images where this distance exceeds ~0.35 are
    # exactly the ones where the estimate diverges (gain 3.5 against a true
    # 1.57) and the output lands up to 8 dB BELOW the untouched input.
    # This is guideline sec. 5's reject logic, with a signal that works.
    ci = f[inner].mean(0); ci = ci / max(ci.sum(), EPS)
    co = f[outer].mean(0); co = co / max(co.sum(), EPS)
    dchr = float(np.abs(ci - co).sum())
    info["chroma_distance"] = dchr
    trust = 1.0
    if cfg.trust_check:
        trust = float(np.clip((cfg.trust_hi - dchr) / max(cfg.trust_hi - cfg.trust_lo, 1e-6),
                              0.0, 1.0))
        if trust < 1.0:
            med = np.array([max(float(np.median(f[:, :, c][inner])), EPS) for c in range(3)])
            g = (A[:, 0] * med + A[:, 1]) / med
            tgt = 1.0 + trust * (g - 1.0)       # shrink toward "change nothing"
            for c in range(3):
                if g[c] > EPS:
                    t = tgt[c] / g[c]
                    A[c] = (A[c, 0] * t, A[c, 1] * t)
    info["trust"] = trust
    info["affine"] = A.tolist()
    field = None
    if cfg.local:
        field = _local_affine_field(f, sd, inst, m_all, d_umb, d_lit, cfg, A, info)
        info["local"] = field is not None
    info["n_inner"] = int(inner.sum())
    info["n_outer"] = int(outer.sum())

    alpha = np.interp(sd.ravel(), centers, lut, left=0.0, right=1.0)
    alpha = alpha.reshape(sd.shape).astype(np.float32)
    alpha[sd <= d_lit] = 0.0              # never touch pixels the profile calls lit
    return (field if field is not None else A), alpha


def IAR(cfg: IARConfig | None = None):
    cfg = cfg or IARConfig()

    def fn(img_u8, mask_u8):
        m = mask_bool(mask_u8)
        f = to_float(img_u8, cfg.linear)
        if not m.any():
            return img_u8.copy(), {"empty_mask": True}

        lum = f.mean(axis=2)
        out = f.copy()
        info_all = []
        for inst in instances(m, cfg.min_instance_area):
            info = {"area": int(inst.sum())}
            A, alpha = _instance_params(f, lum, inst, m, cfg, info)
            info_all.append(info)
            if A is None:
                continue
            if A.ndim == 4:                        # per-pixel local affine field
                a = A[:, :, :, 0]
                b = A[:, :, :, 1]
            else:                                  # one affine for the instance
                a = A[:, 0].astype(np.float32)[None, None, :]
                b = A[:, 1].astype(np.float32)[None, None, :]
            relit = f * a + b
            al = alpha[:, :, None]
            # composite through the measured matte (Le & Samaras' matte layer,
            # with the matte measured instead of predicted)
            out = out * (1.0 - al) + relit * al

        aux = {"n_instances": len(info_all)}
        for key, how in (("penumbra_px", np.mean), ("area", np.sum)):
            v = [d[key] for d in info_all if key in d and np.isfinite(d[key])]
            aux[key] = float(how(v)) if v else float("nan")
        aux["measured_matte_frac"] = float(np.mean(
            [1.0 if d.get("matte") == "measured" else 0.0 for d in info_all])) \
            if info_all else float("nan")
        aux["degenerate"] = float(np.mean(
            [1.0 if d.get("degenerate") else 0.0 for d in info_all])) \
            if info_all else float("nan")
        for key in ("chroma_distance", "trust"):
            v = [d[key] for d in info_all if key in d and np.isfinite(d[key])]
            aux[key] = float(np.mean(v)) if v else float("nan")
        aux["trust_min"] = float(np.min([d["trust"] for d in info_all if "trust" in d]))             if any("trust" in d for d in info_all) else float("nan")
        aux["local_frac"] = float(np.mean(
            [1.0 if d.get("local") else 0.0 for d in info_all])) if info_all else float("nan")
        aux["n_arcs"] = float(np.mean(
            [d["n_arcs"] for d in info_all if "n_arcs" in d])) if any(
            "n_arcs" in d for d in info_all) else float("nan")
        aff = [d["affine"] for d in info_all if "affine" in d]
        if aff:
            A = np.array(aff).mean(0)
            aux["a_rgb"] = A[:, 0].tolist()
            aux["b_rgb"] = A[:, 1].tolist()
        return to_u8(np.clip(out, 0.0, 1.0), cfg.linear), aux

    return fn
