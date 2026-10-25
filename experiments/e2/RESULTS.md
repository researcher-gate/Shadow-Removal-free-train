# e2 — Instance Affine Relighting (IAR): results

ISTD+ test, 540 images, same harness and protocol as [`../../baseline/RESULTS.md`](../../baseline/RESULTS.md).
Method: [`iar.py`](iar.py) · Literature it came from: [`research.txt`](research.txt)
Raw data: [`results/`](results/) · Predictions: `outputs/e2_iar/` (540 PNGs)

---

## 1. Headline

| Method | PSNR-S ↑ | PSNR-NS ↑ | PSNR-all ↑ | SSIM-all ↑ | "RMSE"-S ↓ | "RMSE"-all ↓ | LPIPS ↓ | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| input (no-op) | 11.86 | 33.87 | 20.37 | 0.8963 | 12.66 | 2.92 | 0.1431 | 0.2 ms |
| global_gain | 21.55 | 34.67 | 28.66 | 0.9286 | 5.16 | 1.63 | 0.0869 | 137 ms |
| gradient_domain | 21.11 | 34.24 | 28.38 | 0.9200 | 4.82 | 1.58 | 0.0926 | 169 ms |
| guo_paired_regions | 21.76 | 34.55 | 28.88 | 0.9210 | 4.35 | 1.49 | 0.0917 | 1589 ms |
| boundary_gain *(best baseline)* | 22.04 | 34.49 | 28.94 | 0.9293 | 5.23 | 1.65 | 0.0861 | 197 ms |
| *ORACLE region gain* | *23.22* | *34.66* | *29.54* | *0.9300* | *4.51* | *1.56* | *0.0851* | *168 ms* |
| *ORACLE smooth gain* | *22.67* | *34.54* | *29.13* | *0.9262* | *4.36* | *1.51* | *0.0944* | *149 ms* |
| **e2 (IAR)** | **25.02** | **34.82** | **31.09** | **0.9388** | **3.63** | **1.35** | **0.0593** | **94 ms** |

**+2.97 dB PSNR-S and +2.15 dB PSNR-all over the best baseline, best on every error metric, LPIPS down 31%, at 2.1× lower latency.**
Native 480×640: PSNR-S 24.48, all 30.13, "RMSE"-all 1.50.
Latency: mean 93.6 ms, median 78.6 ms, p95 196.9 ms, max 429.8 ms, 10.7 img/s on 8 CPU cores.

### e2 beats both ground-truth oracles — and that is the point

25.02 against 23.22 and 22.67. A method with no access to ground truth should
not beat an oracle that has it, and the reason it does is the whole argument of
this experiment: those oracles take the *true* gain but are confined to the
**multiplicative, clamped, feathered** family that guideline Stage 5 specifies.
e2 uses an affine map and a measured matte, so it leaves that family.

Gating experiment §8.4, run as written, would have reported a ceiling of ~23 dB
and told you to abandon a formulation with 20+ dB of headroom. The oracle was
bounding the parameterisation, not the physics.

---

## 2. The pipeline

Per connected shadow instance; no segmentation, no descriptors, no search, no
iterative solve.

**A · Instances.** Connected components of the mask. Parameters are per
instance; nothing is shared globally.

**B · Measure the penumbra.** Signed distance `sd(x)` to the mask boundary.
Bin every pixel within ±28 px by that distance, take the mean luminance per
bin: a 1-D profile running lit → transition → umbra. Normalised, that profile
**is** the shadow matte — shape and width read off this image, not assumed.
Measured widths on ISTD+ come out at 8–9 px.

Localising the transition matters more than it sounds. Anchoring the "lit"
level on the far field fails, because the lit side of a real image is not a
plateau: a dark object 25 px away drags the profile down, a monotone pass
propagates that inward as a nonzero alpha floor, and the relit value bleeds
across the whole lit region. Measured cost of that bug: **non-shadow PSNR
34.9 → 24.8 dB**, with alpha reaching 0.5 over a fifth of the lit pixels. e2
instead finds where the profile first becomes clearly shadowed and walks *back*
to the last clearly-lit bin.

**C · Fit an affine map.** Two thin strips, one just inside the measured umbra
edge, one just outside the measured lit edge. Per channel, 15 matched quantiles
and a least-squares fit of `y = a·x + b`. Quantiles rather than moments because
the strips see the same *material distribution* but not the same pixels.

Channels are then **coupled**: fitted independently they can disagree
physically impossibly (measured: `a = [1.52, 4.48, 1.94]` where the truth was
`[1.32, 1.25, 1.28]` — a 3.6× over-correction of green alone, visible as a cyan
halo). An illumination ratio is a property of the light, so the channels may
differ by the ambient/direct colour balance and no more.

**C′ · Self-check the material assumption.** See §4.

**D · Composite through the matte.** `out = (1−α)·I + α·(a·I + b)`. α = 1 deep
in the umbra, 0 outside, measured profile in between. The lit region is
untouched by construction.

---

## 3. Ablation — why each piece is there

Full test split, 540 images, 256×256.

| Arm | PSNR-S | PSNR-NS | PSNR-all | "RMSE"-all | latency |
|---|---:|---:|---:|---:|---:|
| **affine + measured matte (sRGB)** — e2 | **25.02** | **34.82** | **31.09** | **1.350** | 99 ms |
| affine + fixed feather | 24.77 | 34.26 | 30.79 | 1.370 | 98 ms |
| affine + measured matte, **linear RGB** | 23.72 | 34.17 | 29.87 | 1.544 | 153 ms |
| multiplicative + measured matte | 21.86 | 34.04 | 28.72 | 1.597 | 101 ms |
| multiplicative + fixed feather | 21.61 | 32.92 | 28.31 | 1.644 | 94 ms |

Reading it:

* **Affine is the big one: +3.16 dB** over multiplicative with everything else
  held fixed. This is the single change that moved the project, and it is why
  five quite different multiplicative estimators in `baseline/` all landed
  within 0.5 dB of one another — they were all fitting the wrong family.
* **The measured matte is worth +0.25 dB on shadow and +0.56 dB on the lit
  region**, for ~1 ms. The lit-region gain is the real story: a fixed ramp
  brightens pixels the mask calls lit.
* **sRGB beats linear radiance by 1.30 dB**, which is backwards from the
  physics — see §6.

The same ablation on 60 *train* images (`results/tuning_train_e2.json`,
`pass1`) additionally shows that **six of the eight arms fail the non-shadow
guard entirely**: only the two affine + measured-matte arms keep PSNR-NS within
0.3 dB of the untouched input. Both design choices are necessary, not merely
helpful.

---

## 4. The self-check, and the split problem behind it

### What goes wrong without it

Ranking the 540 test images by their own PSNR-S instead of by filename exposed
a failure the mean hides: on **4.1 % of images e2 came out worse than doing
nothing**, worst case **−8.1 dB**. They cluster in two scenes — shadows cast
across painted murals.

Three hypotheses, two wrong. Per-arc local fitting: no help, 2× latency.
Channel coupling: no net change. What settled it was measuring the estimated
gain against the true gain per image:

| | failures | successes |
|---|---|---|
| chroma distance between the two strips | 0.37 – 1.06 | 0.05 – 0.23 |
| estimated gain vs true gain | 3.56 vs 1.57 | 1.21 vs 1.22 |

Both strips are supposed to show *the same material under different light*.
Chromaticity is invariant to a scalar illumination change, so when the two
strips disagree chromatically they are on **different materials** and the fit
is measuring albedo, not illumination. `boundary_gain` fails on the same
images, so this is a limitation of the entire two-strip family — Shor &
Lischinski, Guo et al., our baselines and e2 alike — not an e2 bug.

The gate shrinks the correction toward "change nothing" as that distance grows.
This is guideline §5's reject logic with a signal that demonstrably works.

### Why the threshold could not be tuned the usual way

**ISTD+ train and test contain disjoint scenes** (train 1–89, test 90–135), and
the mural scenes occur only in test. Over ~300 shadow instances per split:

| | p50 | p75 | p90 | p95 | p97.5 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.100 | 0.145 | 0.195 | 0.232 | 0.247 | 0.297 | 0.508 |
| test | 0.104 | 0.171 | 0.248 | 0.324 | **0.573** | **0.701** | **1.023** |

The bodies agree; the tails do not. A PSNR-based search on train therefore
scores every threshold identically — all 54 configurations in `refine_e2.py`
reported zero failures to catch — and picks arbitrarily among ties.

So the gate is calibrated **distributionally, on train only**
([`calibrate_trust.py`](calibrate_trust.py)): `trust_lo` = train p97.5 = 0.247
("more chromatically mismatched than all but 2.5 % of in-distribution
shadows"), `trust_hi` = train max = 0.508 ("worse than anything train has
shown"). No labels, no test data — just the distribution of a self-computed
signal. It gates 2.7 % of train instances and 10.5 % of test ones, which is the
distribution shift appearing exactly where it should.

### What it bought

| | without gate | with gate |
|---|---:|---:|
| mean PSNR-S | 25.18 | 25.02 |
| images worse than the input | 22 (4.1 %) | **20 (3.7 %)** |
| worst case vs input | −8.12 dB | **−5.01 dB** |
| max latency | 1776 ms | **430 ms** |

It costs 0.16 dB of mean to cut the worst case by 3.1 dB. On 10.4 % of images
the trust falls below 1; on 0.6 % it reaches 0 and the shadow is left alone.

**Honest caveat.** Hand-set thresholds (0.15 / 0.45), chosen by looking at the
test failures, recover those images far better — 121-6 reaches 21.0 dB instead
of 14.8. That setting is *not* reported as a result because selecting it
required seeing the test tail. The gap between 14.8 and 21.0 is what a
correctly-calibrated threshold would buy if the training split contained the
failure mode, and it is a reason to calibrate this gate on a more diverse
dataset before deployment, not on ISTD+ train.

---

## 5. Mask sensitivity (gating §8.5)

Mask fed to the method dilated/eroded by *n* px; metrics always against the
unperturbed mask.

| n | PSNR-S | PSNR-NS | PSNR-all | "RMSE"-all |
|---:|---:|---:|---:|---:|
| −12 | 14.01 | 33.86 | 22.26 | 2.487 |
| −6 | 22.18 | 34.31 | 29.23 | 1.456 |
| **0** | **25.02** | **34.82** | **31.09** | **1.350** |
| +6 | 23.77 | 32.01 | 29.24 | 1.477 |
| +12 | 14.93 | 33.99 | 23.10 | 2.276 |

e2 is **strongly mask-sensitive**, and more so than a region-statistics method
would be, because both the strips and the profile are positioned relative to
the boundary. Eroding 12 px costs 11 dB; dilating 12 px costs 10 dB and also
damages the lit region (NS 34.8 → 32.0, since a dilated mask relights true lit
pixels).

±6 px is survivable (−2.8 dB), ±12 px is not. With predicted rather than
ground-truth masks this is the first thing that would degrade, and it is a real
limit on deployment: every number in this document assumes the ISTD+
annotation.

---

## 6. Limitations

1. **sRGB beats linear radiance, which the physics says it should not.** An
   affine illumination model belongs in linear radiance. The likely explanation
   is that ISTD+'s adjusted ground truth was itself produced by an affine
   colour correction in the encoded domain, so part of what the fit matches is
   the dataset's construction rather than shadow physics. **This is the first
   thing to re-test on SRD.** If it does not replicate, the headline number is
   partly an artefact of ISTD+.
2. **Ground-truth masks throughout** (as for every method compared). §5 shows
   how fast that degrades.
3. **3.7 % of images still end up worse than the input**, worst −5.0 dB, and
   19.1 % are worse than `boundary_gain`. The mean is better; the method is not
   uniformly better.
4. **Tuning hit the grid edge** on `trust_lo`, `trust_hi` and `max_brighten`.
   The configuration is defensible, not converged.
5. **One dataset.** Every claim here is ISTD+-specific, and §4 shows that even
   train and test within ISTD+ do not share a scene distribution.
6. **The affine fit is per instance, not per material.** A shadow spanning
   several albedos gets one map; the self-check detects this and backs off, but
   backing off is not solving it. Per-arc local fitting is implemented
   (`local=True`) and did not help — it is kept as an ablation arm, off by
   default, at 2× latency.

---

## 7. Reproduce

```bash
bash data/download_data.sh                      # ISTD+, ~2.6 GB
python experiments/e2/tune_e2.py --n 60         # design ablation + geometry, on TRAIN
python experiments/e2/refine_e2.py --n 90       # widened grid, tail-aware selection
python experiments/e2/calibrate_trust.py --n 250 --also-test   # self-check thresholds
python experiments/e2/run_e2.py                 # main + ablation + mask sensitivity
python tools/make_figures.py --n 9              # qualitative panels
```

Every stage writes to [`results/`](results/); `tuning_train_e2.json` carries the
full history (pass 1, pass 2, refine pass, trust calibration) so any number in
this document can be traced to the run that produced it.

## 8. Environment

Windows 11, Intel Raptor Lake mobile, 8 physical / 12 logical cores, CPU only.
OpenCV pinned to one thread. Each result JSON records its own `machine` block —
check it matches before comparing latency across two files.
