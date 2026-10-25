# Results — every method measured on ISTD+ test

All numbers below were produced by **this repository, on one machine, through
one code path** ([`common/runner.py`](common/runner.py)), on the **same 540
ISTD+ test images**. Nothing is quoted from a paper. That is deliberate: see
§6.

Per-method detail: [`baseline/RESULTS.md`](baseline/RESULTS.md) ·
[`experiments/e2/RESULTS.md`](experiments/e2/RESULTS.md)
Raw per-image data: `baseline/results/*.json`, `experiments/e2/results/*.json`

---

## 1. What was run

| # | Method | Kind | Weights | Evaluated |
|---|---|---|---|---|
| 1 | `input` | no-op lower bound | none | ✅ 540 |
| 2 | `global_gain` | training-free, mask-guided | none | ✅ 540 |
| 3 | `gradient_domain` | training-free, mask-guided | none | ✅ 540 |
| 4 | `guo_paired_regions` | training-free, mask-guided | none | ✅ 540 |
| 5 | `boundary_gain` | training-free, mask-guided | none | ✅ 540 |
| 6 | **`e2_iar`** | **training-free, mask-guided** | **none** | ✅ **540** |
| 7 | `oracle_region_gain` | **uses ground truth** | none | ✅ 540 |
| 8 | `oracle_smooth_gain` | **uses ground truth** | none | ✅ 540 |
| — | `bmnet_supervised` | supervised (CVPR'22) | 432 K | ⬜ integrated, not evaluated |
| — | `shadowformer_supervised` | supervised (AAAI'23) | 11.4 M | ⬜ integrated, not evaluated |
| — | `e1_iat` (amplitude transfer) | training-free | none | ⬜ train only |

Rows 7–8 are **ceilings, not baselines** — they read the ground-truth
shadow-free image. They are shown because §4 is the most important result in
this table.

The three unevaluated rows are code-complete and verified to load/run; they
just have not been run over the full test split. Both supervised checkpoints
load with zero missing or unexpected state-dict keys. Running them:
`python baseline/run_baseline.py --supervised bmnet shadowformer` (~2 h on CPU).

---

## 2. Headline comparison — 256×256 (literature protocol)

Sorted by shadow-region PSNR. **S** = shadow region, **NS** = non-shadow,
**all** = whole image.

| Method | PSNR-S ↑ | PSNR-NS ↑ | PSNR-all ↑ | SSIM-S ↑ | SSIM-all ↑ | "RMSE"-S ↓ | "RMSE"-all ↓ | true RMSE-Lab ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input (no-op) | 11.86 | 33.87 | 20.37 | 0.6148 | 0.8963 | 12.66 | 2.92 | 7.06 | 0.1431 |
| gradient_domain | 21.11 | 34.24 | 28.38 | 0.7345 | 0.9200 | 4.82 | 1.58 | 3.05 | 0.0926 |
| global_gain | 21.55 | **34.67** | 28.66 | 0.7754 | 0.9286 | 5.16 | 1.63 | 3.07 | 0.0869 |
| guo_paired_regions | 21.76 | 34.55 | 28.88 | 0.7369 | 0.9210 | 4.35 | 1.49 | 2.86 | 0.0917 |
| boundary_gain | 22.04 | 34.49 | 28.94 | 0.7838 | 0.9293 | 5.23 | 1.65 | 3.13 | 0.0861 |
| *oracle_smooth_gain* | *22.67* | *34.54* | *29.13* | *0.7742* | *0.9262* | *4.36* | *1.51* | *2.89* | *0.0944* |
| *oracle_region_gain* | *23.22* | *34.66* | *29.54* | *0.7871* | *0.9300* | *4.51* | *1.56* | *2.91* | *0.0851* |
| **e2_iar** | **25.02** | 34.82 | **31.09** | **0.8309** | **0.9388** | **3.63** | **1.35** | **2.40** | **0.0593** |

`"RMSE"` is **mean absolute error in CIE L\*a\*b\*** — what the field prints
under that name. `true RMSE-Lab` is an actual RMSE and is ~2× larger; mixing
them would move a method two places in a published table.

### Native resolution (480×640)

Strictly harder, essentially never reported. Ranking is unchanged, so the
256×256 convention costs nothing in comparison terms — it just inflates every
absolute number by ~0.3–0.9 dB.

| Method | PSNR-S ↑ | PSNR-NS ↑ | PSNR-all ↑ | "RMSE"-all ↓ |
|---|---:|---:|---:|---:|
| input (no-op) | 11.81 | 33.10 | 20.22 | 3.05 |
| gradient_domain | 20.75 | 33.16 | 27.74 | 1.73 |
| global_gain | 21.23 | 33.34 | 27.99 | 1.78 |
| guo_paired_regions | 21.34 | 33.29 | 28.15 | 1.65 |
| boundary_gain | 21.72 | 33.10 | 28.25 | 1.80 |
| *oracle_smooth_gain* | *22.29* | *33.30* | *28.44* | *1.66* |
| *oracle_region_gain* | *22.79* | *33.34* | *28.76* | *1.71* |
| **e2_iar** | **24.48** | **33.42** | **30.13** | **1.50** |

---

## 3. Latency

Per image, inference only, native 480×640, after warm-up, excluding disk I/O
and metric computation. Single machine, CPU only, OpenCV pinned to one thread.

| Method | mean ms | median ms | p95 ms | p99 ms | max ms | img/s | 540 images |
|---|---:|---:|---:|---:|---:|---:|---:|
| input (no-op) | 0.2 | 0.2 | 0.5 | 0.7 | 1.0 | 4888 | 0.1 s |
| **e2_iar** | **93.6** | **78.6** | **196.9** | **269.8** | **429.8** | **10.7** | **51 s** |
| global_gain | 137.4 | 126.4 | 240.7 | 275.8 | 320.1 | 7.3 | 74 s |
| *oracle_smooth_gain* | *148.6* | *134.9* | *271.7* | *331.3* | *357.3* | *6.7* | *80 s* |
| *oracle_region_gain* | *167.9* | *151.3* | *294.2* | *350.5* | *397.7* | *6.0* | *91 s* |
| gradient_domain | 168.9 | 151.5 | 300.2 | 349.3 | 387.6 | 5.9 | 91 s |
| boundary_gain | 196.8 | 171.8 | 337.4 | 517.7 | 668.2 | 5.1 | 106 s |
| guo_paired_regions | 1589.3 | 1519.3 | 2288.6 | 2706.7 | 3199.6 | 0.6 | 858 s |

**e2 is the most accurate *and* the fastest non-trivial method.** It is 2.1×
faster than the best baseline and 17× faster than the region-pairing one. That
is a consequence of the design, not a trade-off against it: e2 has no
segmentation, no descriptor search, no FFT and no iterative solve.

Spreads matter more than means here — cost scales with the number and size of
shadow instances, not with image size. For a method sold as "runs in
milliseconds", p95 is the honest number.

---

## 4. The oracle result

**e2 beats both ground-truth oracles**: 25.02 against 23.22 and 22.67.

A method with no access to ground truth should not beat an oracle that has it.
The reason it does is the single most important finding in this project: those
oracles take the *true* gain, but are confined to the **multiplicative,
clamped, feathered** gain-field family. e2 uses an affine map with a measured
matte, so it leaves that family entirely.

Decomposing the smooth-gain oracle on 40 train images shows how much the
parameterisation was costing:

| oracle variant | PSNR-S |
|---|---:|
| σ=15, clamp 8×, feathered — *as originally specified* | 22.01 |
| σ=8 | 22.88 |
| σ=3 | 24.38 |
| no smoothing | 26.78 |
| no smoothing, clamp 50× | 30.18 |
| no smoothing, clamp 50×, hard mask | **44.37** |

A per-pixel multiplicative gain field reaches **44 dB**; the specified Stage 5
discards ~22 dB of it (smoothing −18, clamp −3.4, feather −14).

**Taken at face value, this oracle would have killed the project** — it says
the whole family caps near 23 dB while supervised methods reach the mid-30s.
It was bounding the formulation it was given, not the problem.

---

## 5. Distribution, not just the mean

Per-image PSNR-S across the 540 test images. The mean hides what matters.

| Method | mean | p05 | p25 | median | p75 | p95 | min | worse than doing nothing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_gain | 21.55 | 14.60 | 17.71 | 22.33 | 24.96 | 27.84 | 11.32 | 11 (2.0 %) |
| gradient_domain | 21.11 | 14.89 | 18.36 | 21.30 | 23.74 | 26.71 | 10.95 | 19 (3.5 %) |
| guo_paired_regions | 21.76 | 14.91 | 18.20 | 22.70 | 24.89 | 27.71 | 12.52 | 10 (1.9 %) |
| boundary_gain | 22.04 | 14.85 | 18.06 | 22.76 | 25.55 | 28.77 | 12.42 | **8 (1.5 %)** |
| **e2_iar** | **25.02** | **15.32** | **22.22** | **25.51** | **28.94** | **32.21** | 8.44 | 20 (3.7 %) |

Head-to-head, per image:

| | e2 wins | mean Δ | best Δ | worst Δ |
|---|---:|---:|---:|---:|
| e2 vs global_gain | 426/540 (78.9 %) | +3.46 dB | +16.81 | −8.49 |
| e2 vs gradient_domain | 475/540 (88.0 %) | +3.91 dB | +11.27 | −6.57 |
| e2 vs guo_paired_regions | 445/540 (82.4 %) | +3.26 dB | +11.25 | −11.00 |
| e2 vs boundary_gain | 437/540 (80.9 %) | +2.97 dB | +16.80 | −7.85 |

**Read the last column of the first table.** e2 has the best mean, the best
median, the best p05 and the best p95 — and the *worst* minimum and a higher
failure rate than `boundary_gain`. It is a higher-variance method: it wins
bigger and it loses harder.

The losses are not random. They are shadows falling across **material
boundaries** — a mural, a painted court line — where the two strips e2 fits
land on different materials and the fit measures albedo instead of
illumination. `boundary_gain` fails on the same images for the same reason, so
this limits the whole two-strip family, not e2 specifically.

e2 detects this condition (chroma distance between its strips) and shrinks the
correction toward "change nothing", which cuts the worst case from −8.12 dB to
−5.01 dB. It mitigates the failure; it does not remove it.

---

## 6. Why nothing here is quoted from a paper

Published ISTD+ numbers are **not mutually comparable**. ShadowFormer scores
**26.55** whole-image PSNR in one widely-used unified re-benchmark and **~32.9**
in its own paper. Two incompatible conventions are in circulation, differing in
resize, in region definition, and in whether "RMSE" means MAE-in-Lab.

So every method in this repository — including the two supervised networks — is
**run inside this harness, on this machine**, rather than quoted. For a
comparison to be fair, that is what it has to mean.

For external context only, never mixed into the tables above: the strongest
classical methods reported on ISTD+ in the recent literature are Guo et al.
(TPAMI'13) at 25.52 PSNR / 0.925 SSIM / 6.09 MAE and Yang et al. (TIP'12) at
20.26 / 0.706 / 15.90. Full survey: [`experiments/e2/research.txt`](experiments/e2/research.txt).

---

## 7. Protocol

| | |
|---|---|
| Dataset | ISTD+ (Adjusted ISTD) — original ISTD shadow images + masks, Le & Samaras' colour-corrected shadow-free frames |
| Split | `test`, 540 triplets, native 480×640 |
| Headline resolution | 256×256 (`cv2.INTER_LINEAR`, mask nearest) |
| Aggregation | per-image metric, then unweighted mean |
| Masks | ground-truth ISTD+ annotations, for **every** method |
| Tuning | every knob selected on the **train** split, under a guard rejecting any config that loses >0.3 dB non-shadow PSNR |
| Hardware | Windows 11, Intel Raptor Lake mobile, 8 physical / 12 logical cores, **CPU only** |

**Caveat on cross-run latency.** Baseline latencies were measured under
python 3.12.14 / numpy 2.5.3 / torch 2.13.0; e2's under a later interpreter on
the same machine. Library minor versions shift numpy/OpenCV throughput by a few
percent. Every result JSON carries its own `machine` block — check it matches
before treating a small latency difference as real. The 2× gap between e2 and
`boundary_gain` is far outside that noise; a 5 % gap would not be.

**All methods use ground-truth masks**, so every number assumes a perfect mask.
e2's mask sensitivity: ±6 px of error costs ~2.8 dB, ±12 px costs ~11 dB.

---

## 8. Summary

- **e2 is the best method measured here on every quality metric, and the
  fastest non-trivial one.** +2.97 dB PSNR-S and +2.15 dB PSNR-all over the
  best baseline, LPIPS down 31 %, at 2.1× lower latency.
- **It beats both ground-truth oracles**, which proves those oracles were
  bounding a parameterisation rather than the problem.
- **It is higher-variance than the baselines** — better at every percentile
  from p05 up, but with a worse floor and a 3.7 % rate of being worse than
  doing nothing.
- **The gap to supervised methods is not measured here yet.** Both checkpoints
  are integrated and load cleanly; running them is one command and ~2 h.
