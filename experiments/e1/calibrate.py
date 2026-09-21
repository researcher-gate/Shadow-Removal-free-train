"""Stage 0 -- the single calibrated scalar, and gating experiment 8.2.

Measures, per shadow instance on the ISTD+ TRAIN split:

    r    = sqrt(area / pi)                      from the mask only
    f_c  = band centroid of the log residual    from the paired images

and regresses f_c on 1/r.  The slope is C, the one number the whole pipeline
learns.  The scatter of f_c against 1/r *is* gating experiment 8.2.

Deviation from the root guideline, deliberate and documented in CORRECTIONS.md
(C7): calibration runs on ISTD+ train rather than DESOBA, so that calibration
and evaluation share a domain.  The cost is dynamic range -- ISTD+ shadows are
single cast shadows of fairly uniform size -- and that limitation is reported
with the fit rather than hidden.

    python experiments/e1/calibrate.py --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from common.imgutil import mask_bool, srgb_to_linear
from common.istd import load_split

OUT = os.path.join(ROOT, "experiments", "e1", "results")


def soft_band_profile(power, fr, n_bands=24, f_max=0.5):
    """Overlapping Gaussian bands with linearly increasing centre and width --
    the guideline's prescription, which avoids the ringing that hard annuli
    put into the profile.

    Returns (centers, summed energy, mean power per band).  The distinction
    matters enormously and the guideline does not draw it: in 2D the number of
    DFT bins in a band grows like f, so for any residual with a noise floor the
    *summed* energy profile rises with f no matter what the signal does, and
    its centroid measures the noise floor rather than the shadow.
    """
    centers = np.linspace(0.0, f_max, n_bands + 1)[1:]
    widths = centers * 0.35 + f_max / (2.0 * n_bands)
    e = np.empty(n_bands)
    mp = np.empty(n_bands)
    for j in range(n_bands):
        wgt = np.exp(-0.5 * ((fr - centers[j]) / widths[j]) ** 2)
        wsum = float(wgt.sum())
        e[j] = float((power * wgt).sum())
        mp[j] = e[j] / max(wsum, 1e-12)
    return centers, e, mp


def _centroid(centers, prof):
    t = prof.sum()
    return float((centers * prof).sum() / t) if t > 0 else float("nan")


def _quantile_freq(centers, prof, q=0.5):
    t = prof.sum()
    if t <= 0:
        return float("nan")
    c = np.cumsum(prof) / t
    return float(np.interp(q, c, centers))


def instance_centroid(R, inst, n_bands=24):
    ys, xs = np.nonzero(inst)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    py = int(0.25 * (y1 - y0)); px = int(0.25 * (x1 - x0))
    y0 = max(0, y0 - py); y1 = min(R.shape[0], y1 + py)
    x0 = max(0, x0 - px); x1 = min(R.shape[1], x1 + px)
    crop = R[y0:y1, x0:x1]
    h, w = crop.shape
    if h < 16 or w < 16:
        return None
    win = np.outer(np.hanning(h), np.hanning(w))

    # the degradation is spatially confined to the mask (guideline 3.1), so the
    # residual outside it is pair misalignment and sensor noise, not signal
    conf = cv2.GaussianBlur(inst[y0:y1, x0:x1].astype(np.float32), (0, 0), 3.0)
    out = {}
    for tag, sig in (("raw", crop), ("masked", crop * conf)):
        F = np.fft.fft2(sig * win)
        P = F.real ** 2 + F.imag ** 2
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        fr = np.sqrt(fy ** 2 + fx ** 2)
        P[0, 0] = 0.0                   # DC carries the mean, not the scale
        centers, e, mp = soft_band_profile(P, fr, n_bands)
        floor = float(np.median(mp[centers > 0.35]))
        mps = np.clip(mp - floor, 0.0, None)
        out[f"fc_sum_{tag}"] = _centroid(centers, e)          # guideline, literal
        out[f"fc_mean_{tag}"] = _centroid(centers, mp)        # bin-count corrected
        out[f"fc_floorsub_{tag}"] = _centroid(centers, mps)   # + noise floor removed
        out[f"f50_floorsub_{tag}"] = _quantile_freq(centers, mps, 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=800)
    args = ap.parse_args()

    samples = load_split("train")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)
    samples = [samples[i] for i in sorted(idx)]

    rows = []
    for i, s in enumerate(samples):
        img, mask, free = s.load()
        m = mask_bool(mask)
        if not m.any():
            continue
        a = srgb_to_linear(img.astype(np.float32) / 255.0)
        b = srgb_to_linear(free.astype(np.float32) / 255.0)
        R = (np.log(np.clip(a, 1e-3, None)) - np.log(np.clip(b, 1e-3, None))).mean(2)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < args.min_area:
                continue
            inst = lab == k
            est = instance_centroid(R, inst)
            if est is None or not np.isfinite(est["fc_sum_raw"]):
                continue
            per = cv2.arcLength(cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
            rows.append({
                "image": s.name, "area": area,
                "r": float(np.sqrt(area / np.pi)),
                "f_c": est["fc_sum_raw"],
                **est,
                "perimeter": float(per),
                "p_over_a": float(per / max(area, 1)),
                "mean_log_drop": float(np.abs(R[inst]).mean()),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} images, {len(rows)} instances", flush=True)

    from scipy.stats import spearmanr

    r = np.array([x["r"] for x in rows])
    x = 1.0 / r
    ESTIMATORS = ["fc_sum_raw", "fc_sum_masked", "fc_mean_raw", "fc_mean_masked",
                  "fc_floorsub_raw", "fc_floorsub_masked",
                  "f50_floorsub_raw", "f50_floorsub_masked"]

    def fit(vals):
        v = np.asarray(vals, float)
        ok = np.isfinite(v)
        xx, vv = x[ok], v[ok]
        if vv.size < 10 or vv.std() == 0:
            return None
        C0 = float((xx * vv).sum() / (xx * xx).sum())
        A = np.stack([xx, np.ones_like(xx)], 1)
        slope, intercept = np.linalg.lstsq(A, vv, rcond=None)[0]
        ss_tot = float(((vv - vv.mean()) ** 2).sum())
        r2 = 1.0 - float(((vv - (A @ [slope, intercept])) ** 2).sum()) / ss_tot
        r2_0 = 1.0 - float(((vv - C0 * xx) ** 2).sum()) / ss_tot
        return {
            "n": int(vv.size),
            "C_through_origin": C0, "r2_through_origin": r2_0,
            "slope": float(slope), "intercept": float(intercept), "r2": r2,
            "pearson_vs_inv_r": float(np.corrcoef(xx, vv)[0, 1]),
            "spearman_vs_inv_r": float(spearmanr(xx, vv).statistic),
            "value_range": [float(vv.min()), float(vv.max())],
            "value_median": float(np.median(vv)),
        }

    fits = {k: fit([row.get(k, np.nan) for row in rows]) for k in ESTIMATORS}
    best = max((k for k in fits if fits[k]),
               key=lambda k: abs(fits[k]["spearman_vs_inv_r"]))

    out = {
        "experiment": "Stage 0 calibration / gating experiment 8.2",
        "split": "train", "n_images": len(samples), "n_instances": len(rows),
        "definitions": {
            "r": "sqrt(area/pi) of the connected shadow instance, mask only",
            "fc_sum_*": "guideline 3.1 as literally written: centroid of the "
                        "SUMMED soft-band energy of log(I_shadow)-log(I_free)",
            "fc_mean_*": "same, but mean power per band (divides out the fact "
                         "that a 2D annulus at frequency f contains ~f times "
                         "more DFT bins)",
            "fc_floorsub_*": "mean power with the high-frequency noise floor "
                             "(median of bands above 0.35 cyc/px) subtracted",
            "f50_floorsub_*": "frequency below which half the floor-subtracted "
                              "energy lies",
            "*_raw": "residual over the instance bounding box",
            "*_masked": "residual multiplied by a soft instance mask first, so "
                        "pair misalignment outside the shadow cannot contribute",
        },
        "fits": fits,
        "best_estimator_by_abs_spearman": best,
        "r_range": [float(r.min()), float(r.max())],
        "r_percentiles": {p: float(np.percentile(r, p)) for p in (5, 25, 50, 75, 95)},
        "dynamic_range_note":
            "r spans %.2fx on ISTD+ (5th-95th pct)." % (
                np.percentile(r, 95) / max(np.percentile(r, 5), 1e-6)),
        "rows": rows,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stage0_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in ESTIMATORS if fits[k]]
        n = len(keys)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.0 * ((n + 1) // 2), 7.2))
        for ax, k in zip(np.ravel(axes), keys):
            v = np.array([row.get(k, np.nan) for row in rows], float)
            ok = np.isfinite(v)
            ax.scatter(x[ok], v[ok], s=7, alpha=0.35)
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, fits[k]["C_through_origin"] * xs, "r-", lw=1)
            ax.set_title("%s\nrho=%+.3f  C=%.3f"
                         % (k, fits[k]["spearman_vs_inv_r"],
                            fits[k]["C_through_origin"]), fontsize=9)
            ax.set_xlabel("1/r"); ax.set_ylabel("cyc/px")
        for ax in np.ravel(axes)[n:]:
            ax.axis("off")
        fig.suptitle("Gating 8.2 -- spectral scale of the log residual vs 1/r "
                     "(ISTD+ train, %d instances)" % len(rows))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "gating_8_2_scatter.png"), dpi=125)
        print("plot ->", os.path.join(OUT, "gating_8_2_scatter.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
