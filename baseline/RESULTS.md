# Baseline results — ISTD+ test (540 images)

Reference numbers for [`../guideline.md`](../guideline.md) and
[`../experiments/e1/`](../experiments/e1/). Produced by
[`run_baseline.py`](run_baseline.py); raw per-image data in
[`results/`](results/).

**Reproduce**

```bash
bash data/download_data.sh
python baseline/tune.py --n 60          # choose every knob on TRAIN
python baseline/run_baseline.py         # evaluate once on TEST
python baseline/run_baseline.py --list  # what is runnable, and is it ready
```

---

## 1. Protocol

Conventions are stated because the shadow-removal literature is inconsistent
about all three, and each one moves the numbers by more than the gaps between
methods.

| | |
|---|---|
| Dataset | ISTD+ (Adjusted ISTD): original ISTD shadow images + masks, Le & Samaras' colour-corrected shadow-free frames |
| Split | `test`, 540 triplets, native 480×640 |
| Headline resolution | **256×256** (`cv2.INTER_LINEAR`, mask nearest) — the literature protocol. Native-resolution tables also given |
| Regions | **S** = shadow (mask>127), **NS** = non-shadow, **all** = whole image. Region PSNR uses only that region's pixels; region SSIM averages the SSIM map inside it |
| Aggregation | per-image metric, then unweighted mean over the 540 images |
| **"RMSE"** | **is a mean absolute error in CIE L\*a\*b\*** (L∈[0,100]), which is what the literature prints under that name — the authors of the evaluation code say so themselves. True RMSE is reported separately |
| LPIPS | AlexNet, whole image, at 256×256 |
| Latency | around the inference call only, native resolution, after 2 warm-up images, excluding disk I/O and metric computation |

Masks are the ISTD+ annotations for every method, including the supervised
ones. Not a contribution — a component.

**Hyper-parameter discipline.** Every knob (ring widths, feather width, segment
count, edge width, sRGB vs linear radiance) was selected on a fixed random
subset of the 1330 **training** triplets by `tune.py`, under a guard that
rejects any configuration losing more than 0.3 dB of non-shadow PSNR relative
to the untouched input. Selected values are in
[`results/tuning_train.json`](results/tuning_train.json) and in each result
JSON's `meta.config`. A baseline whose knobs were never fitted is a straw man;
one whose knobs were fitted on test is a leak.

---

## 2. Methods

Four tiers. The tier matters more than the numbers: a training-free method and
a supervised network are not competing at the same thing.

### Training-free, mask-guided — e1's actual peer group

| Name | Mechanism | Config chosen on train |
|---|---|---|
| `input` | No-op. Establishes what "do nothing" scores | — |
| `global_gain` | One per-channel multiplicative gain from whole-region statistics | `linear=True, robust=False` |
| `gradient_domain` | Log-domain gradient nulling across the shadow edge, Poisson reintegration by DCT (Finlayson/Hordley/Drew in spirit) | `linear=False, edge_width=7` |
| `guo_paired_regions` | SLIC over-segmentation → pair each shadow segment with lit segments by illumination-invariant chromaticity + gradient-orientation texture + spatial proximity → relight by the pair ratio (**Guo/Dai/Hoiem in spirit**) | `linear=False, n_segments=400, topk=5` |
| `boundary_gain` | Per-instance gain from thin rings either side of the shadow boundary | `linear=True, band=24, gap=4, f_width=8` |

`guo_paired_regions` is the critical baseline: it also pairs shadow and lit
regions and it also does not train the relighting. It differs from the proposed
method by working in pixel space, with no spectral decomposition and no
self-verification.

### Supervised — pretrained by the original authors on ISTD+

| Name | Model | Params | Checkpoint |
|---|---|---|---|
| `bmnet_supervised` | BMNet (Zhu et al., CVPR 2022) | 431,559 | `AISTD.pth`, ships in the upstream repo |
| `shadowformer_supervised` | ShadowFormer (Guo et al., AAAI 2023) | 11,372,617 | `ShadowFormer_ISTDplus.pth`, epoch 485 |

Opt in with `--supervised bmnet shadowformer`. Both load with zero missing or
unexpected state-dict keys. **Both run on CPU**, so their latency sits in the
same column as everything else — honest for cross-method comparison, and badly
pessimistic as a deployment number for either network.

### Oracles — use the ground truth; ceilings, never results

| Name | What it bounds |
|---|---|
| `oracle_region_gain` | The best achievable by one constant per-channel `k` per shadow instance |
| `oracle_smooth_gain` | The best achievable by the *smooth, clamped, feathered* gain field that guideline Stage 5 specifies |

These answer gating experiment §8.4 for every method in the study at once,
which is why they live here rather than in `experiments/e1/`.

---

## 3. Results

### 3.1 Quality — 256×256 (headline protocol)

| Method | PSNR S ↑ | PSNR NS ↑ | PSNR all ↑ | SSIM S ↑ | SSIM NS ↑ | SSIM all ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input (no-op) | 11.86 | 33.87 | 20.37 | 0.6148 | 0.9443 | 0.8963 |
| global_gain | 21.55 | **34.67** | 28.66 | 0.7754 | **0.9519** | 0.9286 |
| gradient_domain | 21.11 | 34.24 | 28.38 | 0.7345 | 0.9490 | 0.9200 |
| guo_paired_regions | 21.76 | 34.55 | 28.88 | 0.7369 | 0.9510 | 0.9210 |
| **boundary_gain** | **22.04** | 34.49 | **28.94** | **0.7838** | 0.9515 | **0.9293** |
| *ORACLE region gain* | *23.22* | *34.66* | *29.54* | *0.7871* | *0.9525* | *0.9300* |
| *ORACLE smooth gain* | *22.67* | *34.54* | *29.13* | *0.7742* | *0.9497* | *0.9262* |

| Method | "RMSE" S ↓ | "RMSE" NS ↓ | "RMSE" all ↓ | true RMSE-Lab all ↓ | RMSE-RGB all ↓ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input (no-op) | 12.66 | 0.99 | 2.92 | 7.06 | 27.76 | 0.1431 |
| global_gain | 5.16 | 0.98 | 1.63 | 3.07 | 10.87 | 0.0869 |
| gradient_domain | 4.82 | 0.98 | 1.58 | 3.05 | 11.01 | 0.0926 |
| **guo_paired_regions** | **4.35** | 0.98 | **1.49** | **2.86** | 10.45 | 0.0917 |
| boundary_gain | 5.23 | 0.98 | 1.65 | 3.13 | 10.56 | **0.0861** |
| *ORACLE region gain* | *4.51* | *0.98* | *1.56* | *2.91* | *9.89* | *0.0851* |
| *ORACLE smooth gain* | *4.36* | *0.98* | *1.51* | *2.89* | *10.24* | *0.0944* |

Note the `true RMSE-Lab` column against the `"RMSE"` column: 2.86 vs 1.49 for
the same method and the same images. Reporting one under the other's name
would move a method roughly two places in a published table.

### 3.2 Quality — native 480×640

Strictly harder, and essentially never reported. Included so the 256×256
figures above can be read as the convention they are rather than as the truth.

| Method | PSNR S ↑ | PSNR NS ↑ | PSNR all ↑ | "RMSE" S ↓ | "RMSE" all ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| input (no-op) | 11.81 | 33.10 | 20.22 | 12.75 | 3.05 |
| global_gain | 21.23 | 33.34 | 27.99 | 5.34 | 1.78 |
| gradient_domain | 20.75 | 33.16 | 27.74 | 5.04 | 1.73 |
| guo_paired_regions | 21.34 | 33.29 | 28.15 | 4.62 | 1.65 |
| boundary_gain | 21.72 | 33.10 | 28.25 | 5.40 | 1.80 |
| *ORACLE region gain* | *22.79* | *33.34* | *28.76* | *4.71* | *1.71* |
| *ORACLE smooth gain* | *22.29* | *33.30* | *28.44* | *4.55* | *1.66* |

Downsizing to 256×256 is worth about +0.3 dB PSNR-S and −0.15 "RMSE" across
the board. The ranking is unchanged, so the convention costs nothing in
comparison terms — it just inflates every absolute number equally.

### 3.3 Latency

Per image, inference only, native 480×640, after warm-up.

| Method | mean ms | median ms | std ms | min ms | max ms | p90 ms | p95 ms | p99 ms | fps | 540 imgs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| input (no-op) | 0.2 | 0.2 | 0.1 | 0.1 | 1.0 | 0.4 | 0.5 | 0.7 | 4888 | 0.1 s |
| global_gain | 137.4 | 126.4 | 42.4 | 76.1 | 320.1 | 197.1 | 240.7 | 275.8 | 7.3 | 74 s |
| gradient_domain | 168.9 | 151.5 | 63.5 | 79.9 | 387.6 | 264.5 | 300.2 | 349.3 | 5.9 | 91 s |
| guo_paired_regions | 1589.3 | 1519.3 | 403.1 | 649.8 | 3199.6 | 2144.0 | 2288.6 | 2706.7 | 0.6 | 858 s |
| boundary_gain | 196.8 | 171.8 | 77.0 | 102.7 | 668.2 | 284.6 | 337.4 | 517.7 | 5.1 | 106 s |
| *ORACLE region gain* | *167.9* | *151.3* | *58.2* | *90.3* | *397.7* | *252.6* | *294.2* | *350.5* | *6.0* | *91 s* |
| *ORACLE smooth gain* | *148.6* | *134.9* | *55.5* | *78.2* | *357.3* | *230.3* | *271.7* | *331.3* | *6.7* | *80 s* |

Spreads are wide (`boundary_gain` p99 is 2.6× its median) because cost scales
with the number and size of shadow instances, not with image size. For a
method whose selling point is "runs in seconds", p95 is the number that
matters, not the mean.

`guo_paired_regions` is **8× slower than `boundary_gain` and loses to it on
PSNR**. It wins only on Lab error. Its cost is the SLIC segmentation plus an
all-pairs descriptor comparison over ~400 segments.

---

## 4. What the baselines say about the project

### 4.1 The spread between training-free methods is 0.5 dB

`global_gain` — one per-channel scalar for the entire shadow — is within
**0.5 dB** of every more sophisticated training-free method, and beats
`gradient_domain` outright. On ISTD+ there is almost no headroom for a
spatially adaptive estimator, because the dataset is predominantly single cast
shadows on near-uniform ground. A method whose whole thesis is intra-image
*reference matching* has very little to match against here.

This is a property of the dataset, not of the methods, and it is the strongest
argument in these results for moving to SRD or WSRD+ before drawing
conclusions about the amplitude-transfer idea.

### 4.2 Gating §8.4 — the oracle result, and why it nearly misled us

Read at face value, §3.1 kills the project: `oracle_smooth_gain` scores 22.67
dB, only **0.6 dB** above `boundary_gain`, and `oracle_region_gain` only 1.2 dB
above. That would mean the whole multiplicative-gain family is capped near 23
dB while ShadowFormer reaches the mid-30s, and no estimator improvement could
matter.

That conclusion is wrong, and the reason is instructive: those oracles are
ceilings for **guideline Stage 5's particular parameterisation**, not for the
multiplicative model. Decomposing the smooth-gain oracle on 40 train images:

| Oracle variant | PSNR-S |
|---|---:|
| σ=15, clamp 8×, feathered — **Stage 5 as written** | 22.01 |
| σ=8 | 22.88 |
| σ=3 | 24.38 |
| no smoothing | 26.78 |
| no smoothing, clamp 50× | 30.18 |
| no smoothing, clamp 50×, hard mask (no feather) | **44.37** |

A per-pixel multiplicative gain field reaches **44 dB**. Stage 5 discards ~22 dB
of that in three independent places:

1. **Gaussian smoothing of the gain field** — the single largest cost, ~18 dB.
2. **The `G ≤ 8` magnitude bound** — ~3.4 dB, so true deep-umbra gains genuinely
   exceed 8×, and guideline §5.2's "plausible 1–6×" range is too tight to be
   used as a hard clamp.
3. **The penumbra feather** — ~14 dB, because PSNR-S is measured over the
   *whole* mask including the boundary band that the feather deliberately
   under-corrects.

So **§8.4 passes**: the formulation is adequate and the limit is the estimator
plus Stage 5's own conservatism. But two caveats keep this from being a licence
to delete the smoothing:

- The 44 dB variant uses a *hard* mask. It would produce a visible shadow
  contour on any imperfect mask — exactly the artefact the feather exists to
  prevent, and which PSNR does not penalise but a human immediately sees.
- Smoothing suppresses *estimator noise*, not just signal. Tested directly on
  e1 ([`../experiments/e1/results/tuning_train_e1.json`](../experiments/e1/results/tuning_train_e1.json),
  `refine_pass`): dropping e1's smoothing from 0.35 to 0.05 of the patch size
  *lowers* its PSNR-S and damages the lit region, because e1's gain estimate
  is not the ground truth. The oracle can afford sharpness; the estimator
  cannot.

The actionable version: Stage 5's clamp is too tight and should be raised, the
smoothing is a bias/variance trade-off that must be tuned per estimator rather
than fixed, and the feather should be evaluated with a metric that sees
contours (not PSNR).

### 4.3 The non-shadow region is nearly free, and nearly meaningless

Every method scores 34.2–34.7 dB on the non-shadow region against 33.87 for
doing nothing. Because ISTD+'s adjusted ground truth is colour-matched to the
shadow frame *in the lit region*, NS-PSNR mostly measures "did you leave the
lit pixels alone". It is a useful guard — it is what the tuning guard rejects
configurations on, and it catches gain leaking outside the mask — but
whole-image PSNR inherits it and is correspondingly inflated. **PSNR-S is the
only column in which these methods actually differ.**

One concrete instance: an early `boundary_gain` variant post-blurred the gain
field, which leaked `G>1` into the lit region and cost ~1.5 dB NS-PSNR for no
shadow-region benefit. Replacing the blur with a C¹ smoothstep feather ramp
fixed it. That class of bug is invisible in PSNR-all and obvious in PSNR-NS.

---

## 5. Honest limitations of these baselines

1. `guo_paired_regions` and `gradient_domain` are **re-implementations in the
   spirit of** the cited work, not the authors' code. They are calibrated on
   train and are respectable, but a published number for Guo et al. should not
   be read off this table.
2. Tuning used 60 train images, not all 1330. Enough to pick between
   configurations whose gaps are ~0.5 dB; not enough to resolve ties.
3. The two supervised baselines are timed on **CPU**. Comparable within this
   table, meaningless as deployment latency.
4. All methods use ground-truth masks. Mask sensitivity is measured for e1
   (§8.5) but not for the baselines, so the baselines are flattered by a
   perfect mask they would not have in deployment.
5. The oracles bound *this* gain parameterisation, as §4.2 spells out. They are
   not bounds on shadow removal.

---

## 6. Environment

Recorded per run in each result JSON under `machine`.

- Windows 11 (10.0.26200), Intel64 Family 6 Model 186 (Raptor Lake mobile),
  8 physical / 12 logical cores
- OpenCV 5.0.0, pinned to **one** thread (`common/imgutil.py`) — the FFT
  matching in e1 already parallelises via `scipy.fft(workers=-1)`, and letting
  both libraries spawn pools over the same cores both added latency variance
  and triggered an intermittent untyped C++ exception out of
  `connectedComponentsWithStats`
- LPIPS on CPU in all cases

> **Caveat on cross-run latency comparison.** The tables in §3.3 were produced
> under python 3.12.14 / numpy 2.5.3 / torch 2.13.0; later runs in this project
> may report a different interpreter (same machine, same CPU). Library minor
> versions can shift numpy/OpenCV throughput by a few percent. Every result
> JSON carries its own `machine` block — **check it matches before comparing
> latency across two result files**, and re-measure both in one session if the
> comparison is load-bearing.
