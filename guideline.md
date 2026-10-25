# Training-Free Shadow Removal: Research Plan (v2)

**This document replaces the original plan.** The original argued for a
self-verifying training-free shadow remover built on intra-image Fourier
amplitude transfer. That plan has now been implemented and measured, and the
measurements changed it substantially. What follows is the plan as it stands
*after* the evidence, with every claim either backed by a number in this repo
or explicitly flagged as untested.

The audit of the original document is preserved in
[experiments/e1/CORRECTIONS.md](experiments/e1/CORRECTIONS.md) — 13 numbered
findings, three of which would have produced a silently broken implementation.
It is worth keeping: several of those errors are the kind that survive review.

---

## 1. Motivation

### 1.1 The asymmetry (unchanged, and it held up)

Zero-shot low-light enhancement needs external priors because a uniformly dark
image contains no correct reference anywhere in the frame. A shadow image is
different: the lit region is the same scene, the same camera, the same instant,
the same materials. **The reference is already in the image.**

This was the founding intuition and it survives. What changed is *how* to use
it. The original plan read "reference" as *a matching patch elsewhere in the
image*, and built Fourier amplitude transfer to find and exploit one. The
evidence says the useful reference is much closer and much simpler: **the lit
pixels immediately across the shadow boundary**, which are the same material by
spatial continuity rather than by a learned or spectral similarity measure.

### 1.2 Why training-free is worth pursuing

Unchanged, and still the strongest part of the case. Paired shadow/shadow-free
data is impractical to collect outside controlled scenes, masks are hard to
annotate, and supervised methods degrade under cross-dataset transfer. A method
with no learned weights cannot overfit a training distribution and runs in
tens of milliseconds on a CPU.

One correction of terminology. Recent work calls itself "training-free" while
using a frozen 800M-parameter diffusion prior (FreeShadow). That is a different
claim. **Ours is: zero learned weights of any kind.** The only fitted
quantities are a handful of scalar hyper-parameters selected on the ISTD+ train
split, each of which is a knob, not a model.

### 1.3 The gap

No published shadow removal method reports its own confidence. Deployed shadow
removal has no ground truth, and a method that silently hallucinates on hard
inputs is worse than one that declines them.

**This gap is now demonstrated rather than asserted** — see §4.

---

## 2. Contributions, as the evidence now ranks them

The original ranked these: (1) self-verification, (2) intra-image reference
framing, (3) geometry-derived spectral scale, (4) the pipeline. The ranking was
right about (1) and wrong about (3).

| # | Contribution | Status |
|---|---|---|
| 1 | **The model is affine, not multiplicative** | **Established.** +3.16 dB on ISTD+ test, and it beats the ground-truth oracle of the multiplicative family. §3 |
| 2 | **A measured penumbra matte** | **Established.** +0.25 dB shadow, +0.56 dB lit region, ~1 ms. §3 |
| 3 | **Self-verification** | **Partly established.** The material-mismatch check works and cuts the worst case by 3.1 dB. The broader claim — predicting per-image quality without labels — is still untested. §4 |
| 4 | Geometry-derived spectral scale | **Abandoned.** The relationship is weak (ρ = +0.48, R² = 0.30) and the original's own estimator reported it with the wrong sign. §6 |
| 5 | Intra-image amplitude transfer | **Superseded.** Implemented as e1; worse than a boundary-ring baseline on ISTD+. §6 |

---

## 3. Method

Full specification and results: [experiments/e2/RESULTS.md](experiments/e2/RESULTS.md).
Implementation: [experiments/e2/iar.py](experiments/e2/iar.py).

### 3.1 Physical model

Under a Lambertian assumption with direct and ambient illumination, a lit pixel
sees `(L_direct + L_ambient)·R` and a shadowed pixel sees
`(k·L_direct + L_ambient)·R`. Shor & Lischinski (EG 2008) and Le & Samaras
(ICCV'19 / TPAMI'21 — the ISTD+ dataset paper) both conclude the same thing:
the shadowed→lit relation is **affine**,

```
I_free = a ⊙ I_shadow + b
```

with the offset carrying the ambient term. Forcing `b = 0` — which the original
plan, and every baseline in this repo, does — fits the wrong family.

**This is the single most consequential correction in the project.** It is also
the explanation for a result that was otherwise puzzling: five quite different
multiplicative estimators (global statistics, boundary rings, region pairing,
gradient-domain reintegration) all land within 0.5 dB of each other on ISTD+.
They were not converging on the problem's difficulty. They were converging on a
shared modelling error.

### 3.2 Pipeline

Per connected shadow instance:

**A — Instances.** Connected components of the mask. Masks come from the
dataset; not a contribution, a component.

**B — Measure the penumbra.** Signed distance to the boundary; bin pixels by
it; mean luminance per bin. That 1-D profile, normalised, **is** the matte —
its shape and its width read off this image rather than assumed. Measured
widths on ISTD+: 8–9 px.

**C — Affine fit from two strips.** One strip just inside the measured umbra
edge, one just outside the measured lit edge; per channel, matched quantiles
and a least-squares affine fit. Quantiles rather than moments because the
strips see the same material *distribution*, not the same pixels. Channels are
coupled: an illumination ratio is a property of the light, so they may differ
by the ambient/direct colour balance and no more.

**C′ — Self-check** (§4).

**D — Composite.** `out = (1−α)·I + α·(a·I + b)`.

No segmentation, no descriptors, no search, no iterative solve — which is why
it is *faster* than the baselines, not slower: 94 ms/image against 197 ms for
the best training-free baseline and 1589 ms for the region-pairing one.

### 3.3 Where the model breaks

- **Chromatic shift.** Ambient skylight is bluer than direct sun, so `a` and
  `b` are per-channel. Handled, with the channels coupled.
- **Penumbra.** Handled by the measured matte.
- **Deep umbra.** Where attenuation drives signal below the noise floor,
  texture is destroyed and no affine correction recovers it.
- **Material discontinuity across the boundary.** The two-strip assumption
  fails outright. **This is the dominant real failure mode** (§4) and it breaks
  every method in this family, not just ours.
- **Indirect illumination.** Violates the single-source assumption; untested
  here, ISTD+ is outdoor.

---

## 4. Self-verification

### 4.1 What was wrong with the original §5

Three of its proposed checks could not fail:

- "Phase identical before/after" is trivially true per patch by construction,
  and false for the output, since multiplying by a spatial gain field convolves
  the spectrum.
- "`G ≥ 1` everywhere" and the magnitude bound are *enforced* by clamping, so
  they always pass and carry zero information. Only the **pre-clamp violation
  mass** is informative.
- The round trip on an untouched lit patch is near-identity by construction: it
  matches lit content to equally lit references and never loads the
  illumination estimate, which is the component that actually fails.

Details in [CORRECTIONS.md](experiments/e1/CORRECTIONS.md) C4, C5, C8.

### 4.2 A check that works

Both strips are supposed to show *the same material under different light*.
Chromaticity is invariant to a scalar illumination change, so if the two strips
disagree chromatically they are on **different materials** and the fit is
measuring albedo, not illumination.

Measured on ISTD+ test:

| | failures | successes |
|---|---|---|
| chroma distance between strips | 0.37 – 1.06 | 0.05 – 0.23 |
| estimated gain vs true gain | 3.56 vs 1.57 | 1.21 vs 1.22 |

Shrinking the correction toward "change nothing" as that distance grows cuts
the worst case from −8.12 dB to −5.01 dB relative to the untouched input, for
0.16 dB of mean. This is the original §5.4 reject logic, with a signal that
demonstrably discriminates.

### 4.3 The validation problem the original plan did not anticipate

The original §5.5 proposed correlating the reject signal against true per-image
PSNR on a paired benchmark, and called that "the paper". On ISTD+ that
experiment has a structural problem:

**train and test contain disjoint scenes** (train 1–89, test 90–135), and the
failure mode the check exists to catch — shadows across painted murals —
occurs **only in test**. Over ~300 shadow instances per split:

| | p50 | p75 | p90 | p95 | p97.5 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.100 | 0.145 | 0.195 | 0.232 | 0.247 | 0.297 | 0.508 |
| test | 0.104 | 0.171 | 0.248 | 0.324 | **0.573** | **0.701** | **1.023** |

The bodies agree; the tails do not. A threshold search on train scores every
value identically — all 54 configurations reported zero failures to catch.

So the threshold is calibrated **distributionally on train**: `trust_lo` = train
p97.5, `trust_hi` = train max. No labels, no test data, just the distribution of
a self-computed signal. This is a *better* story than "we tuned the threshold",
and it generalises: a safety threshold for a failure your training data does
not contain can still be set from what your training data does contain.

### 4.4 What is still untested

The headline claim of the original plan — that a combination of invariants
predicts per-image output quality well enough to drive an accept/reject
decision — **has not been run**. The machinery exists
([experiments/e1/analyze_selfcheck.py](experiments/e1/analyze_selfcheck.py)):
~22 label-free signals, a ridge combiner fitted on train and evaluated on test,
and an accuracy/coverage curve against oracle and random orderings.

Given §4.3, it should be run on a dataset whose train and test share a scene
distribution, or across datasets. **This is the main open item.**

---

## 5. Experimental protocol

### 5.1 Dataset

ISTD+ (Adjusted ISTD): original ISTD shadow images and masks + Le & Samaras'
colour-corrected shadow-free frames. 1330 train / 540 test.
Assembly: [data/download_data.sh](data/download_data.sh).

Original ISTD is unusable here: its shadow-free frame carries a global colour
shift against the shadow frame, so a method would be scored partly on guessing
that shift.

Next, in priority order: **SRD** (does the sRGB-over-linear result replicate, or
is it an ISTD+ artefact? §7.1), then DESOBA, then WSRD+/INS.

### 5.2 Baselines

All training-free baselines tuned on train, run once on test, through the same
harness as the method so metrics *and* latency compare:
`global_gain`, `gradient_domain`, `guo_paired_regions` (the critical classical
baseline), `boundary_gain`.

Two supervised references — BMNet and ShadowFormer, authors' ISTD+ checkpoints
— are **measured inside this harness on the same CPU**, not quoted. Published
ISTD+ tables are not mutually comparable: ShadowFormer scores 26.55 whole-image
PSNR in one widely-used re-benchmark and ~32.9 in its own paper.

Two ground-truth **oracles** bound the gain-field family. See §6.2 for why they
nearly produced a false negative.

### 5.3 Metrics

PSNR, SSIM, MAE-Lab (**this is what the literature prints as "RMSE"**), true
RMSE, LPIPS — on shadow / non-shadow / whole image, at 256×256 and native
resolution. Latency reported as mean/median/p95/p99, measured around inference
only, after warm-up.

Reliability metrics (ours): correlation between reject signal and true PSNR;
accuracy/coverage curve; fraction of images made worse than the input.

**Report the tail, not just the mean.** The failure that mattered most in this
project was invisible in every mean and obvious the moment images were ranked
by their own score.

---

## 6. What was tested and abandoned

Keeping this section is the point of the document. Two ideas were ranked highly
in the original plan and did not survive contact with measurement.

### 6.1 Geometry-derived spectral scale — abandoned

The claim: shadow spatial extent determines the spectral extent of its
degradation, so the operating scale can be *computed* from the mask.

Two findings killed it:

1. **The original's own estimator reports the relationship with the wrong
   sign.** It specifies the centroid of *summed* radial band energy. In 2D the
   number of DFT bins in an annulus grows like `f`, so for any residual with a
   noise floor the summed profile rises with `f` regardless of signal. Measured
   on 473 train instances: Pearson **−0.028**, Spearman −0.168. A bin-count-
   and floor-corrected estimator recovers ρ = **+0.48**, R² = 0.30.
2. Even corrected, the relationship is **weak**, over a dataset whose instance
   radii span only ~9×.

Demoting this also removes the original §2's claim that patch size and band
cutoff are redundant — they band-limit *opposite* ends of the spectrum
(CORRECTIONS C1), so the argument for demotion in the original §11 was itself
wrong, even though its conclusion was right.

### 6.2 The oracle that nearly killed the project

Read naively, the gain-field oracles say the whole family is capped near 23 dB
and should be abandoned. That is wrong, and instructively so: they bound the
**parameterisation**, not the physics. Decomposed:

| oracle variant | PSNR-S |
|---|---:|
| σ=15, clamp 8×, feathered — *as originally specified* | 22.01 |
| no smoothing | 26.78 |
| no smoothing, clamp 50× | 30.18 |
| no smoothing, clamp 50×, hard mask | **44.37** |

A per-pixel gain field reaches 44 dB; the specified Stage 5 discards ~22 dB of
it. **e2 then beats both oracles outright** (25.02 vs 23.22 and 22.67), which
is only possible because it leaves that family.

**Lesson for any future gating experiment: an oracle bounds the formulation you
gave it, not the problem.** State which.

### 6.3 Intra-image amplitude transfer — superseded

Implemented in full as [experiments/e1](experiments/e1/), including
shift-invariant phase-correlation matching (the original's raw phase distance
is dominated by translation — CORRECTIONS C3) and a chromaticity term (phase is
invariant to albedo level, so it cannot do the job the original assigned it —
C2). Result: **below a boundary-ring baseline** on ISTD+ train, at ~10×
the latency.

The diagnosis is in the data: on ISTD+ the match confidences are low because
the dataset is predominantly single shadows on near-uniform ground, so there is
little material diversity for intra-image matching to exploit. It is possible
this idea pays on SRD or WSRD+. It does not pay here, and the honest statement
is that the boundary is a better reference than a distant patch.

---

## 7. Limitations

1. **sRGB beats linear radiance (+1.30 dB), which the physics says it should
   not.** The likely cause is that ISTD+'s adjusted ground truth was itself
   produced by an affine colour correction in the encoded domain, so the fit is
   partly matching the dataset's construction. **The highest-priority open
   question.** If it does not replicate on SRD, the headline number is partly
   an ISTD+ artefact.
2. **Ground-truth masks throughout.** ±6 px of mask error costs ~2.8 dB; ±12 px
   costs ~11 dB. This method is *more* mask-sensitive than a region-statistics
   method because its strips and profile are positioned relative to the
   boundary.
3. **3.7 % of test images end up worse than the input**, worst −5.0 dB, and
   19.1 % are worse than the best baseline. The mean is better; the method is
   not uniformly better.
4. **Material discontinuity across the shadow boundary** breaks the whole
   two-strip family. Detected and backed off from, not solved.
5. **One dataset**, and §4.3 shows that even its train and test do not share a
   scene distribution.
6. **Deep umbra is unrecoverable** where texture fell below the noise floor.
7. Tuning hit the grid edge on three parameters; the configuration is
   defensible, not converged.

---

## 8. Next actions

In priority order:

1. **SRD.** Settles whether the sRGB result and the affine advantage are real
   or ISTD+ artefacts (§7.1). Nothing else should be claimed until this runs.
2. **Run the self-verification validation** (§4.4) on a split whose train and
   test share a scene distribution, or cross-dataset.
3. **Predicted masks instead of ground-truth**, to put a real number on §7.2.
4. **Material-aware fitting** for the failure in §7.4 — per-arc fitting was
   tried and did not help, so this needs a different idea, not a bigger version
   of the same one.
5. Converge the three grid-edge parameters.

---

## 9. Honest assessment

The original plan's instinct was right about which contribution would last —
self-verification — and wrong about which mechanism would get there. Fourier
amplitude transfer was the elaborate answer; the affine model and the boundary
strips were the correct one, and they were already in the 2008 literature.

The project's actual result is smaller and more solid than the original pitch:
**a training-free shadow remover that beats every training-free baseline we
could build, beats the ground-truth oracle of the formulation it replaces, runs
in 94 ms on a CPU, and knows when to decline.** That is a workshop paper with a
reliability angle, not the reliability result the original document imagined —
and it is supported by measurements rather than by argument.

The most transferable findings are the ones about *method*, not about shadows:
an oracle bounds the formulation you give it (§6.2); a metric's mean hides the
failures that matter (§5.3); and a safety threshold for a failure your training
split does not contain can still be calibrated from that split's distribution
(§4.3).
