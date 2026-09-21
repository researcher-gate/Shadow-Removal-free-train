"""e1 -- Intra-image Amplitude Transfer (IAT) with self-verification.

Implements the pipeline of the root guideline.md with the corrections listed in
experiments/e1/CORRECTIONS.md.  The ones that change code rather than prose:

  C2/C3  matching uses the *phase-correlation peak* (shift invariant, Fourier
         shift theorem) scored against its own candidate distribution, plus a
         chromaticity term -- phase alone is invariant to albedo *level*, so it
         cannot do the job the guideline assigns it;
  C1     patch size and band cutoff are independent parameters in absolute
         cycles/pixel; they band-limit opposite ends of the spectrum;
  C5     the gain field is estimated in the log domain from robust per-patch
         ratios and low-passed, never by per-pixel division (which explodes in
         the deep umbra);
  C12    the per-patch transfer estimate is blended with the boundary-ring
         prior under the *match confidence*, so the self-check is wired into
         the estimator instead of only into a post-hoc reject.  `use_prior=False`
         recovers the literal guideline pipeline for the ablation.

Entry point:  IAT(cfg) -> fn(img_rgb_u8, mask_u8) -> (pred_u8, aux_dict)
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from scipy.fft import fft2, ifft2

from common.imgutil import (EPS, core_and_outside, disk, feather, instances,
                            mask_bool, ring, to_float, to_u8)

# ----------------------------------------------------------------- configuration


@dataclass
class IATConfig:
    # --- Stage 0: the one calibrated scalar (fitted on ISTD+ train masks) ---
    C: float = 1.00            # f_c = C / r  (cycles/pixel), r in pixels
    a_s: float = 1.00          # patch size      s = a_s / f_c
    a_b: float = 1.00          # band cutoff  f_beta = a_b * f_c

    # --- guideline sec. 8.1 ablation switches (re-specified, see C1) ---
    geometry_patch: bool = True
    geometry_band: bool = True
    fixed_s: int = 64          # patch size when geometry_patch is False
    fixed_beta: float = 0.02   # cycles/pixel when geometry_band is False
    full_band: bool = False    # literal "beta = none": transfer all amplitude

    # --- tiling / matching ---
    s_min: int = 24
    s_max: int = 128
    overlap: float = 0.5
    topk: int = 4
    cand_max: int = 40
    w_chroma: float = 1.5
    w_dist: float = 0.15
    conf_peak_ratio: float = 2.0   # peak/median(peak) at which phase conf -> 1
    conf_chroma: float = 0.06      # chroma distance scale for confidence

    # --- estimator ---
    linear: bool = True
    use_prior: bool = True     # blend patch estimate with the boundary-ring prior
    gain_max: float = 6.0      # plausible direct-to-ambient ratio (guideline 5.2)
    gain_smooth_frac: float = 0.5
    feather_inner: int = 3
    feather_width: int = 12

    # --- self-verification (guideline sec. 5) ---
    verify: bool = True
    rt_trials: int = 1

    # --- reject / fallback (guideline sec. 5.4) ---
    reject: bool = False
    reject_threshold: float = -1e9
    reject_signal: str = "rt_psnr"
    fallback: str = "boundary"      # boundary | dc | identity


# ------------------------------------------------------------------- utilities


def _radius(inst) -> float:
    return float(np.sqrt(inst.sum() / np.pi))


def _tile_positions(h, w, s, stride):
    ys = list(range(0, max(h - s, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - s, 0) + 1, stride)) or [0]
    if h - s > 0 and ys[-1] != h - s:
        ys.append(h - s)
    if w - s > 0 and xs[-1] != w - s:
        xs.append(w - s)
    return np.array(ys), np.array(xs)


def _hann2d(s):
    w = np.hanning(s + 1)[:s].astype(np.float32)
    w = np.maximum(w, 1e-3)
    return np.outer(w, w)


def _band_mask(s, f_cut):
    fr = np.fft.fftfreq(s).astype(np.float32)
    r = np.sqrt(fr[:, None] ** 2 + fr[None, :] ** 2)
    return (r <= f_cut).astype(np.float32)


def _gather(f, ys, xs, s):
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(f, (s, s), axis=(0, 1))
    return np.ascontiguousarray(v[np.ix_(ys, xs)].reshape(-1, 3, s, s))


def _coverage(mask, ys, xs, s):
    integ = cv2.integral(mask.astype(np.float64))
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    Y, X = Y.ravel(), X.ravel()
    return (integ[Y + s, X + s] - integ[Y, X + s]
            - integ[Y + s, X] + integ[Y, X]) / float(s * s)


def _ring_prior(f, inst, gain_max, band=10, gap=4):
    """Per-channel gain from thin bands either side of the instance boundary."""
    r_in = ring(inst, gap, gap + band, inside=True)
    r_out = ring(inst, gap, gap + band, inside=False)
    if r_in.sum() < 64 or r_out.sum() < 64:
        r_in, r_out = core_and_outside(inst)
    g = np.ones(3, np.float64)
    for c in range(3):
        a, b = f[:, :, c][r_in], f[:, :, c][r_out]
        if a.size >= 32 and b.size >= 32:
            g[c] = float(np.median(b) / max(np.median(a), EPS))
    return np.clip(g, 1.0, gain_max)


# --------------------------------------------------------------- core pipeline


def _instance_gain(f, inst, m_all, cfg, out_info):
    h, w = inst.shape
    r = _radius(inst)
    f_c = cfg.C / max(r, 1.0)

    if cfg.geometry_patch:
        s0 = int(round(cfg.a_s / max(f_c, 1e-6)))
    else:
        s0 = cfg.fixed_s
    # a patch must actually fit inside the umbra it is sampling
    s0 = min(s0, max(cfg.s_min, int(r)))
    s = int(np.clip(s0, cfg.s_min, min(cfg.s_max, h, w)))

    prior = _ring_prior(f, inst, cfg.gain_max)
    info = {"r": r, "f_c": f_c, "prior_rgb": prior.tolist()}

    lit_ok = cv2.dilate(m_all.astype(np.uint8), disk(12)) == 0
    sh_idx = lit_idx = None
    for _ in range(3):
        core = cv2.erode(inst.astype(np.uint8), disk(max(2, s // 8))) > 0
        stride = max(2, int(round(s * (1.0 - cfg.overlap))))
        ys, xs = _tile_positions(h, w, s, stride)
        cov_core = _coverage(core, ys, xs, s)
        cov_lit = _coverage(lit_ok, ys, xs, s)
        sh_idx = np.where(cov_core >= 0.70)[0]
        lit_idx = np.where(cov_lit >= 0.995)[0]
        if sh_idx.size and lit_idx.size:
            break
        if s <= cfg.s_min:
            break
        s = max(cfg.s_min, s // 2)          # shrink until the shadow admits a patch

    info["patch"] = s
    if cfg.full_band:
        f_beta = 0.5
    elif cfg.geometry_band:
        f_beta = cfg.a_b * f_c
    else:
        f_beta = cfg.fixed_beta
    f_beta = float(np.clip(f_beta, 1.0 / s, 0.5))
    info["f_beta"] = f_beta
    info["n_shadow_patches"] = int(sh_idx.size)
    info["n_lit_patches"] = int(lit_idx.size)

    a_feather = feather(inst, cfg.feather_inner, cfg.feather_width)[:, :, None]

    if sh_idx.size == 0 or lit_idx.size == 0:
        # no admissible patch pair: the transfer is undefined, fall back to the
        # ring prior and say so (this is a reject condition, not a silent path)
        info["degenerate"] = True
        out_info.append(info)
        g = 1.0 + a_feather * (prior[None, None, :].astype(np.float32) - 1.0)
        return g.astype(np.float32)

    if lit_idx.size > 300:
        lit_idx = lit_idx[np.linspace(0, lit_idx.size - 1, 300).astype(int)]

    win = _hann2d(s)
    bmask = _band_mask(s, f_beta)
    P = _gather(f, ys, xs, s)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    pos = np.stack([Y.ravel(), X.ravel()], 1).astype(np.float32)

    Ps, Pl = P[sh_idx], P[lit_idx]
    Fs = fft2((Ps.mean(1) * win).astype(np.float32), workers=-1)
    Fl = fft2((Pl.mean(1) * win).astype(np.float32), workers=-1)
    Al = np.abs(fft2(Pl * win[None, None], workers=-1))

    mean_s = Ps.mean(axis=(2, 3))
    mean_l = Pl.mean(axis=(2, 3))
    chr_s = mean_s / np.maximum(mean_s.sum(1, keepdims=True), EPS)
    chr_l = mean_l / np.maximum(mean_l.sum(1, keepdims=True), EPS)

    diag = float(np.hypot(h, w))
    acc_log = np.zeros((h, w, 3), np.float32)
    accw = np.zeros((h, w), np.float32)
    peaks_rel, confs, raw_g = [], [], []

    log_prior = np.log(prior).astype(np.float32)

    for a, gi in enumerate(sh_idx):
        dc = np.abs(chr_l - chr_s[a][None]).sum(1)
        dd = np.linalg.norm(pos[lit_idx] - pos[gi][None], axis=1) / diag
        pre = cfg.w_chroma * dc + cfg.w_dist * dd
        cand = np.argsort(pre)[:cfg.cand_max]

        R = Fs[a][None] * np.conj(Fl[cand])
        R /= np.abs(R) + 1e-8
        pk = np.real(ifft2(R, workers=-1)).reshape(cand.size, -1).max(1)

        score = (1.0 - pk / max(pk.max(), 1e-8)) + pre[cand]
        order = np.argsort(score)[:cfg.topk]
        sel = cand[order]

        # confidence: how far the chosen peak stands above the candidate pool's
        # own noise floor, times how well the reflectance descriptor agrees
        med_pk = float(np.median(pk)) + 1e-8
        rel = float(pk[order].max() / med_pk)
        peaks_rel.append(rel)
        c_phase = np.clip((rel - 1.0) / max(cfg.conf_peak_ratio - 1.0, 1e-6), 0.0, 1.0)
        c_chroma = float(np.exp(-dc[sel].mean() / cfg.conf_chroma))
        conf = float(c_phase * c_chroma) if cfg.use_prior else 1.0
        confs.append(conf)

        A_ref = Al[sel].mean(0)
        Fsh = fft2(Ps[a] * win[None], workers=-1)
        A_new = bmask[None] * A_ref + (1.0 - bmask[None]) * np.abs(Fsh)
        rec = np.real(ifft2(A_new * (Fsh / (np.abs(Fsh) + 1e-12)), workers=-1))

        # Stage 5 as a log-domain ratio of the corrected estimate to the input.
        # Not a per-pixel division of raw pixels (C6): the ratio is taken in
        # logs on clipped values and immediately low-passed, so the deep umbra
        # cannot blow it up -- but intra-patch variation survives, which is what
        # lets the band cutoff have any effect at all.
        rec_un = rec / np.maximum(win, 0.05)[None]
        dlog = (np.log(np.clip(rec_un, 1e-3, None))
                - np.log(np.clip(Ps[a], 1e-3, None)))
        dlog = np.clip(dlog, -np.log(cfg.gain_max), np.log(cfg.gain_max * 2))
        sp = max(1.0, 0.15 * s)
        dlog = np.stack([cv2.GaussianBlur(dlog[c], (0, 0), sp) for c in range(3)])
        raw_g.append(np.exp(np.median(dlog.reshape(3, -1), axis=1)))

        lg = conf * dlog + (1.0 - conf) * log_prior[:, None, None]
        y0, x0 = int(pos[gi][0]), int(pos[gi][1])
        acc_log[y0:y0 + s, x0:x0 + s, :] += win[:, :, None] * np.moveaxis(lg, 0, -1)
        accw[y0:y0 + s, x0:x0 + s] += win

    ok = accw > 1e-3
    log_g = np.broadcast_to(log_prior[None, None, :], (h, w, 3)).copy()
    log_g[ok] = acc_log[ok] / accw[ok][:, None]

    sig = max(2.0, cfg.gain_smooth_frac * s)
    k = int(sig * 4) | 1
    log_g = cv2.GaussianBlur(log_g, (k, k), sig)

    raw_g = np.array(raw_g)
    info.update({
        "match_peak_rel_mean": float(np.mean(peaks_rel)),
        "match_peak_rel_min": float(np.min(peaks_rel)),
        "conf_mean": float(np.mean(confs)),
        "conf_min": float(np.min(confs)),
        # pre-clamp violation mass: the physical checks of guideline 5.2 can
        # only be informative if measured BEFORE the clamp enforces them
        "frac_patch_gain_below_1": float((raw_g < 1.0).mean()),
        "frac_patch_gain_above_max": float((raw_g > cfg.gain_max).mean()),
        "patch_gain_mean": float(raw_g.mean()),
        "patch_gain_p95": float(np.percentile(raw_g, 95)),
    })

    g_field = np.exp(np.clip(log_g, 0.0, np.log(cfg.gain_max))).astype(np.float32)
    med = np.array([float(np.median(g_field[:, :, c][inst])) for c in range(3)])
    info["gain_rgb_median"] = med.tolist()
    info["chroma_order_ok"] = bool(med[0] >= med[1] >= med[2])
    info["chroma_order_margin"] = float(min(med[0] - med[1], med[1] - med[2]))

    out_info.append(info)
    return (1.0 + a_feather * (g_field - 1.0)).astype(np.float32)


def _core_run(f, m, cfg, out_info):
    G = np.ones(f.shape, np.float32)
    for inst in instances(m):
        G = G * _instance_gain(f, inst, m, cfg, out_info)   # composes multiplicatively
    return G


# --------------------------------------------------------- self-verification


def _grad_ncc(a, b, sel):
    if sel.sum() < 64:
        return float("nan")
    out = []
    for img in (a, b):
        g = np.log(np.clip(img.mean(2), 1e-3, None)).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        out.append(np.stack([gx[sel], gy[sel]]).ravel())
    u, v = out
    u = u - u.mean(); v = v - v.mean()
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else float("nan")


def _boundary_contour(out_f, m, width=6):
    mu = m.astype(np.uint8)
    ring_p = (cv2.dilate(mu, disk(width)) - cv2.erode(mu, disk(width))) > 0
    ctrl = (cv2.dilate(mu, disk(3 * width)) - cv2.dilate(mu, disk(2 * width))) > 0
    if ring_p.sum() < 64 or ctrl.sum() < 64:
        return float("nan")
    g = np.log(np.clip(out_f.mean(2), 1e-3, None)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return float(np.median(e[ring_p]) / max(np.median(e[ctrl]), 1e-6))


def _place_synthetic(m, rng):
    h, w = m.shape
    mu = m.astype(np.uint8)
    lit = cv2.dilate(mu, disk(20)) == 0
    if m.sum() == 0:
        return None
    for _ in range(24):
        dy = int(rng.integers(-h // 2, h // 2))
        dx = int(rng.integers(-w // 2, w // 2))
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        sm = cv2.warpAffine(mu, T, (w, h), flags=cv2.INTER_NEAREST) > 0
        if sm.sum() < 1500:
            continue
        if (sm & ~lit).sum() / sm.sum() < 0.02:
            return sm
    rad = int(np.sqrt(int(m.sum()) / np.pi))
    cand = np.argwhere(lit)
    if cand.size == 0 or rad < 10:
        return None
    py, px = cand[int(rng.integers(0, len(cand)))]
    sm = np.zeros((h, w), np.uint8)
    cv2.circle(sm, (int(px), int(py)), rad, 1, -1)
    sm = (sm > 0) & lit
    return sm if sm.sum() > 1500 else None


def _round_trip(f, m, cfg, k_rgb, rng):
    """Synthetic-shadow round trip (C8).

    The guideline's round trip runs the pipeline on an *untouched* lit patch,
    which is near-identity by construction: it matches lit content to equally
    lit references, so it never exercises the illumination estimate -- the one
    component that actually fails.  Here a lit region is darkened by a per-
    channel factor taken from this image's own estimated shadow strength, given
    a penumbra, and pushed through the full pipeline.  Ground truth is still
    free and the check now probes the failing part.
    """
    sm = _place_synthetic(m, rng)
    if sm is None:
        return None
    alpha = cv2.GaussianBlur(sm.astype(np.float32), (25, 25), 6.0)[:, :, None]
    k = np.clip(1.0 / np.clip(k_rgb, 1.05, cfg.gain_max), 0.05, 0.95).astype(np.float32)
    f_syn = np.clip(f * (1.0 - alpha * (1.0 - k[None, None, :])), 1e-4, 1.0)

    sub = IATConfig(**{**asdict(cfg), "verify": False, "reject": False})
    G = _core_run(f_syn, sm, sub, [])
    rec = np.clip(f_syn * G, 0.0, 1.0)

    sel = cv2.erode(sm.astype(np.uint8), disk(5)) > 0
    if sel.sum() < 256:
        sel = sm

    def psnr(x):
        d = (x[sel] - f[sel]) * 255.0
        mse = float(np.mean(d * d))
        return 100.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))

    return {"rt_psnr": psnr(rec), "rt_psnr_degraded": psnr(f_syn),
            "rt_gain_db": psnr(rec) - psnr(f_syn),
            "rt_area_frac": float(sm.mean())}


# ------------------------------------------------------------------ front end


def _prior_only_gain(f, m, cfg):
    G = np.ones(f.shape, np.float32)
    for inst in instances(m):
        g = _ring_prior(f, inst, cfg.gain_max)
        a = feather(inst, cfg.feather_inner, cfg.feather_width)[:, :, None]
        G = G * (1.0 + a * (g[None, None, :].astype(np.float32) - 1.0))
    return G


def IAT(cfg: IATConfig | None = None, seed: int = 0):
    cfg = cfg or IATConfig()

    def fn(img_u8, mask_u8):
        rng = np.random.default_rng(seed)
        m = mask_bool(mask_u8)
        f = to_float(img_u8, cfg.linear)
        if not m.any():
            return img_u8.copy(), {"empty_mask": True}

        inst_info: list[dict] = []
        G = _core_run(f, m, cfg, inst_info)
        out_f = np.clip(f * G, 0.0, 1.0)

        def agg(key, how=np.mean, default=float("nan")):
            v = [d[key] for d in inst_info
                 if key in d and isinstance(d[key], (int, float))
                 and np.isfinite(d[key])]
            return float(how(v)) if v else default

        aux = {
            "n_instances": len(inst_info),
            "r_mean": agg("r"),
            "patch_mean": agg("patch"),
            "f_beta_mean": agg("f_beta"),
            "shadow_area_frac": float(m.mean()),
            "match_peak_rel_mean": agg("match_peak_rel_mean"),
            "match_peak_rel_min": agg("match_peak_rel_min", np.min),
            "conf_mean": agg("conf_mean"),
            "conf_min": agg("conf_min", np.min),
            "frac_patch_gain_below_1": agg("frac_patch_gain_below_1", np.max),
            "frac_patch_gain_above_max": agg("frac_patch_gain_above_max", np.max),
            "patch_gain_mean": agg("patch_gain_mean"),
            "patch_gain_p95": agg("patch_gain_p95", np.max),
            "chroma_order_margin": agg("chroma_order_margin", np.min),
            "chroma_order_ok": float(np.mean(
                [1.0 if d.get("chroma_order_ok") else 0.0 for d in inst_info])),
            "degenerate": float(np.mean(
                [1.0 if d.get("degenerate") else 0.0 for d in inst_info])),
            "gain_mean": float(G[m].mean()),
            "gain_p95": float(np.percentile(G[m], 95)),
        }

        if cfg.verify:
            core = cv2.erode(m.astype(np.uint8), disk(7)) > 0
            aux["struct_ncc"] = _grad_ncc(f, out_f, core if core.sum() > 64 else m)
            aux["boundary_contour"] = _boundary_contour(out_f, m)
            med = [d["gain_rgb_median"] for d in inst_info if "gain_rgb_median" in d]
            k_rgb = np.array(med).mean(0) if med else np.array([2.0, 2.0, 2.0])
            rts = [r for r in (_round_trip(f, m, cfg, k_rgb, rng)
                               for _ in range(cfg.rt_trials)) if r]
            if rts:
                for key in rts[0]:
                    aux[key] = float(np.mean([r[key] for r in rts]))

        if cfg.reject:
            sc = aux.get(cfg.reject_signal, float("nan"))
            aux["reject_score"] = float(sc)
            if np.isfinite(sc) and sc < cfg.reject_threshold:
                aux["rejected"] = 1.0
                if cfg.fallback == "identity":
                    return img_u8.copy(), aux
                out_f = np.clip(f * _prior_only_gain(f, m, cfg), 0.0, 1.0)
            else:
                aux["rejected"] = 0.0

        return to_u8(out_f, cfg.linear), aux

    return fn
