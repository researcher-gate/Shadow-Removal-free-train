"""Gating experiment 8.3 / guideline sec. 5.5 -- validating the validator.

Question: without any ground truth, can the method tell which of its own
outputs are good?

Protocol (the split discipline matters; without it this experiment proves
nothing):
  * signals are read from the e1 run's per-image `aux`, which uses no labels;
  * single-signal correlations against true per-image PSNR are reported on the
    540-image TEST split;
  * the *combined* reject score is a linear model fitted on a TRAIN-split run
    and then applied unchanged to test, so the combiner cannot have seen the
    numbers it is scored on;
  * the accuracy/coverage trade-off curve is reported against two references:
    an oracle ordering (sort by true PSNR) and the random ordering.

    python experiments/e1/analyze_selfcheck.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import pearsonr, spearmanr

RES = os.path.join(HERE, "results")

SIGNALS = [
    "rt_psnr", "rt_gain_db", "rt_psnr_degraded",
    "conf_mean", "conf_min", "match_peak_rel_mean", "match_peak_rel_min",
    "frac_patch_gain_below_1", "frac_patch_gain_above_max",
    "patch_gain_mean", "patch_gain_p95", "gain_mean", "gain_p95",
    "chroma_order_ok", "chroma_order_margin",
    "struct_ncc", "boundary_contour",
    "degenerate", "shadow_area_frac", "r_mean", "patch_mean", "n_instances",
]
TARGET = "psnr_S"


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    aux = d.get("aux") or {}
    per = {r["name"]: r for r in d["per_image"]}
    names = [n for n in per if n in aux]
    X, y = [], []
    for n in names:
        X.append([float(aux[n].get(k, np.nan)) for k in SIGNALS])
        y.append(float(per[n][TARGET]))
    return names, np.array(X, float), np.array(y, float)


def clean(X):
    """Median-impute then z-score; returns (Z, median, mean, std)."""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med[None, :])
    mu, sd = Xi.mean(0), Xi.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (Xi - mu) / sd, med, mu, sd


def apply_clean(X, med, mu, sd):
    Xi = np.where(np.isfinite(X), X, med[None, :])
    return (Xi - mu) / sd


def coverage_curve(order, y, steps=20):
    """Mean true quality of the images kept, as the reject threshold sweeps."""
    out = []
    n = len(y)
    for i in range(1, steps + 1):
        cov = i / steps
        k = max(1, int(round(cov * n)))
        out.append({"coverage": cov, "n_kept": k,
                    "mean_psnr_S_kept": float(y[order[:k]].mean())})
    return out


def main():
    test_p = os.path.join(RES, "e1_main.json")
    train_p = os.path.join(RES, "e1_train_verify.json")
    if not os.path.exists(test_p):
        print("run run_e1.py --stages main first"); return 1

    _, Xte, yte = load(test_p)
    print(f"test: {len(yte)} images")

    single = {}
    for j, k in enumerate(SIGNALS):
        v = Xte[:, j]
        ok = np.isfinite(v) & np.isfinite(yte)
        if ok.sum() < 20 or np.nanstd(v[ok]) < 1e-12:
            single[k] = None
            continue
        single[k] = {
            "n": int(ok.sum()),
            "pearson": float(pearsonr(v[ok], yte[ok]).statistic),
            "spearman": float(spearmanr(v[ok], yte[ok]).statistic),
            "pearson_p": float(pearsonr(v[ok], yte[ok]).pvalue),
        }

    out = {
        "experiment": "gating 8.3 / guideline 5.5 -- reject signal vs true PSNR",
        "target": "per-image PSNR over the shadow region (256x256 protocol)",
        "n_test": int(len(yte)),
        "single_signal_correlations_TEST": single,
    }

    if os.path.exists(train_p):
        _, Xtr, ytr = load(train_p)
        Ztr, med, mu, sd = clean(Xtr)
        A = np.hstack([Ztr, np.ones((len(ytr), 1))])
        # ridge, so 22 correlated signals on a few hundred images stay stable
        lam = 1.0
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
        pred_tr = A @ W
        Zte = apply_clean(Xte, med, mu, sd)
        pred_te = np.hstack([Zte, np.ones((len(yte), 1))]) @ W

        out["combiner"] = {
            "fitted_on": "train split run (%d images)" % len(ytr),
            "model": "ridge linear regression on z-scored signals, lambda=1.0",
            "weights": {k: float(w) for k, w in zip(SIGNALS, W[:-1])},
            "intercept": float(W[-1]),
            "train_pearson": float(pearsonr(pred_tr, ytr).statistic),
            "train_spearman": float(spearmanr(pred_tr, ytr).statistic),
            "TEST_pearson": float(pearsonr(pred_te, yte).statistic),
            "TEST_spearman": float(spearmanr(pred_te, yte).statistic),
            "TEST_mae_of_predicted_psnr_dB": float(np.mean(np.abs(pred_te - yte))),
        }
        score = pred_te
    else:
        print("no train run; falling back to the single best signal as the score")
        best = max((k for k in single if single[k]),
                   key=lambda k: abs(single[k]["spearman"]))
        score = Xte[:, SIGNALS.index(best)]
        score = np.where(np.isfinite(score), score, np.nanmedian(score))
        if single[best]["spearman"] < 0:
            score = -score
        out["combiner"] = {"fallback_single_signal": best}

    order = np.argsort(-score)
    out["coverage_curve_selfcheck"] = coverage_curve(order, yte)
    out["coverage_curve_oracle"] = coverage_curve(np.argsort(-yte), yte)
    rng = np.random.default_rng(0)
    rnd = np.mean([[c["mean_psnr_S_kept"]
                    for c in coverage_curve(rng.permutation(len(yte)), yte)]
                   for _ in range(50)], axis=0)
    out["coverage_curve_random"] = [
        {"coverage": (i + 1) / 20, "mean_psnr_S_kept": float(v)}
        for i, v in enumerate(rnd)]

    p = os.path.join(RES, "selfcheck_8_3.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)

    print("\nsingle-signal |Spearman| vs true PSNR_S (test):")
    for k in sorted((k for k in single if single[k]),
                    key=lambda k: -abs(single[k]["spearman"]))[:12]:
        print(f"  {k:28s} r={single[k]['pearson']:+.3f}  rho={single[k]['spearman']:+.3f}")
    if "TEST_spearman" in out.get("combiner", {}):
        print(f"\ncombined (fitted on train): TEST r={out['combiner']['TEST_pearson']:+.3f} "
              f"rho={out['combiner']['TEST_spearman']:+.3f}")
    print("\ncoverage curve (self-check / oracle / random):")
    for a, b, c in zip(out["coverage_curve_selfcheck"], out["coverage_curve_oracle"],
                       out["coverage_curve_random"]):
        print(f"  {a['coverage']:.2f}  {a['mean_psnr_S_kept']:6.2f}  "
              f"{b['mean_psnr_S_kept']:6.2f}  {c['mean_psnr_S_kept']:6.2f}")
    print("->", p)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].scatter(score, yte, s=9, alpha=0.45)
        ax[0].set_xlabel("reject score (no ground truth)")
        ax[0].set_ylabel("true PSNR over shadow region (dB)")
        t = out.get("combiner", {}).get("TEST_spearman")
        ax[0].set_title("8.3: self-check vs truth" +
                        (f"   rho={t:+.3f}" if t is not None else ""))
        for key, lab, st in (("coverage_curve_oracle", "oracle ordering", "k--"),
                             ("coverage_curve_selfcheck", "self-check", "b-"),
                             ("coverage_curve_random", "random", "r:")):
            cc = out[key]
            ax[1].plot([c["coverage"] for c in cc],
                       [c["mean_psnr_S_kept"] for c in cc], st, label=lab)
        ax[1].set_xlabel("coverage (fraction of images accepted)")
        ax[1].set_ylabel("mean PSNR_S of accepted images (dB)")
        ax[1].set_title("accuracy / coverage trade-off")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "gating_8_3_selfcheck.png"), dpi=130)
        print("plot ->", os.path.join(RES, "gating_8_3_selfcheck.png"))
    except Exception as e:
        print("plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
