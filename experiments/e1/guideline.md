# e1 — Self-Verifying Training-Free Shadow Removal on ISTD+

**Scope of this document.** This is the root `guideline.md` narrowed to one
dataset (ISTD+), one experiment round (e1), and one executable pipeline. It
keeps the parent's structure so the two read side by side. Where it departs
from the parent it says so and points at the entry in
[CORRECTIONS.md](CORRECTIONS.md) that justifies the departure.

**One-line pitch.** A shadow image contains its own reference. We exploit that
to remove shadows with no learned image-domain weights, and — because the
method is algorithmic rather than learned — it can check its own output and
report when it failed.

---

## 1. Motivation

### 1.1 The asymmetry (unchanged from the parent)

Zero-shot low-light enhancement needs external priors because a uniformly dark
image contains no correct reference anywhere in the frame. A shadow image is
different: the lit region is the same scene, the same camera, the same instant,
the same materials. The reference is already in the image. This is why
reference-free correction is tractable for shadows and not for low light.

### 1.2 What e1 is for

e1 is not an attempt at a paper. It is the cheapest set of measurements that
can tell us which of the parent's four contributions is real. Concretely it
must answer:

1. Does the geometry → spectral-scale law exist at all, and is the parent's own
   measurement of it sound? (§8.2)
2. Can the method predict its own per-image quality without labels? (§8.3)
3. Is the multiplicative-gain formulation good enough to be worth estimating
   well? (§8.4)
4. How much of the result is the amplitude transfer, and how much is the
   boundary prior underneath it?

Everything else — SRD, DESOBA, cross-dataset generalisation, a comparison
against supervised networks — waits until those four have answers.

### 1.3 The gap (unchanged)

No published shadow-removal method reports its own confidence. Deployed shadow
removal has no ground truth, and a method that silently hallucinates on hard
inputs is worse than one that declines them.

---

## 2. Contributions, as e1 tests them

| # | Contribution | e1's test | Status going in |
|---|---|---|---|
| 1 | Ground-truth-free self-verification | §8.3: correlate the reject signal against true per-image PSNR on 540 held-out images, with the combiner fitted on train | **Headline.** Three of the parent's signals were replaced because they could not fail (C4, C5, C8) |
| 2 | Intra-image reference framing | Implicit in every run; the matching criterion is rebuilt (C2, C3) | Framing sound, mechanism was wrong |
| 3 | Geometry-derived spectral scale | §8.2 scatter, then §8.1 four-arm ablation | **Conditional.** The parent's own estimator cannot see the effect (C13) |
| 4 | Training-free amplitude transfer pipeline | The vehicle; measured against training-free baselines in `baseline/` | Lowest novelty, as the parent says |

---

## 3. Physical model and notation

### 3.1 The log-multiplicative model (unchanged)

```
I(x)        = L(x) · R(x)
I_shadow(x) = k(x) · L(x) · R(x)
log I_shadow(x) − log I_free(x) = log k(x)
```

With `k` approximately constant over the umbra interior the residual is
`log k · M(x)`: the degradation is spatially confined to the mask and spectrally
shaped by it. This is homomorphic filtering restated, and it is why amplitude
carries illumination while phase carries structure.

**Two consequences the parent draws only one of.** Phase being invariant to a
scalar `k` is what makes it illumination invariant — and equally what makes it
invariant to albedo level. See C2.

### 3.2 Where the model breaks (unchanged, all four apply on ISTD+)

Chromatic shift (ambient skylight is bluer than direct sun, so `k` is
per-channel); penumbra (`k` varies smoothly across the boundary); deep umbra
(attenuation below the noise floor destroys texture, and no multiplicative
correction recovers it); indirect illumination (ISTD+ is outdoor, so this one
mostly does not bite here — it is the reason INS is the eventual stress test).

### 3.3 Notation

| Symbol | Meaning |
|---|---|
| `I`, `M` | input shadow image, binary shadow mask |
| `M_i`, `r_i` | i-th connected instance; `r_i = √(area/π)` |
| `A(p), φ(p)` | amplitude and phase spectra of patch `p` |
| `f_c` | spectral scale of the degradation, cycles/pixel |
| `C` | the calibrated scalar: `f_c = C / r` |
| `s_i` | patch size for instance i, pixels |
| `f_β` | band cutoff for the transfer, **cycles/pixel** (C1) |
| `G` | multiplicative gain field; `I_out = I ⊙ G` |

> **Unit change from the parent.** `β` was a dimensionless "band fraction".
> Here the cutoff is absolute (cycles/pixel). This is not cosmetic: it is what
> makes patch size and band cutoff independently variable, which is what §8.1
> needs in order to answer its own question (C1).

---

## 4. Method

Implementation: [`iat.py`](iat.py).

### Stage 0 — Offline calibration (run once, on ISTD+ **train**)

Fits the single scalar relating shadow geometry to spectral scale.

1. `R = log I_shadow − log I_free` in **linear** RGB, averaged over channels.
2. Multiply `R` by a soft instance mask. §3.1 says the degradation lives inside
   the mask; everything outside it is pair misalignment and sensor noise.
3. Radial soft-Gaussian band profile, then **mean power per band, not summed
   band energy** — in 2D the bin count in an annulus grows like `f`, so a
   summed profile rises with `f` whatever the signal does.
4. Subtract the high-frequency plateau (median mean-power above 0.35 cyc/px).
5. Centroid of what is left ⇒ `f_c`. Measure `r = √(area/π)` from the mask.
6. Regress `f_c` on `1/r` through the origin. The slope is `C`.

Steps 2–4 are the correction in C13. The literal version of step 3 reports the
relationship as absent with the wrong sign; the corrected version finds it, and
finds it weak.

**Output:** one scalar. Two further global multipliers (`a_s`, `a_b`) are
chosen on train by grid search, giving three fitted numbers in total, none of
which ever sees a pixel — only mask geometry and aggregate quality.

**Escalation path**, capped at step 2 (C10): linear fit (1 parameter) → ridge
on geometry features `√(area/π)`, perimeter/area, eccentricity, mean intensity
drop (~5–20 parameters). A tiny MLP is *not* on the path; if ridge is not
enough, the honest conclusion is that the geometry link does not hold.

**Hard constraint (unchanged and important):** the predictor's input is mask
geometry only, never pixels.

### Stage 1 — Mask and instance decomposition

Masks come from the ISTD+ annotations. Not trained by us — a component, not a
contribution. Mask sensitivity is measured separately in §8.5 precisely because
a geometry-derived scale makes us more mask-dependent than a typical mask-based
method.

Connected components ⇒ instances. Per instance: erode for the umbra core,
dilate-minus-erode for the penumbra ring, `r_i` from the area.

### Stage 2 — Scale-adaptive tiling

Per instance, overlapping Hann-windowed patches, ≥50 % overlap.

* `s_i = a_s / f_c,i = a_s · r_i / C`, clipped to `[s_min, s_max]`, and
  additionally capped at `r_i` — **a patch must fit inside the umbra it is
  sampling**. Without that cap the pipeline finds no admissible shadow patch on
  small shadows and silently degenerates; this was the first thing the
  implementation exposed.
* `f_β,i = a_b · f_c,i` cycles/pixel, clipped to `[1/s_i, 0.5]`.
* If no admissible shadow patch or no admissible lit patch exists, halve `s`
  and retry (three times). Still nothing ⇒ mark the instance `degenerate` and
  fall back to the ring prior. This is the cheap version of §6.2's
  applicability test (C11).

Patches are classified as shadow (≥70 % inside the eroded umbra) or lit
reference (≥99.5 % outside the dilated mask).

### Stage 3 — Reference matching

For each shadow patch, rank lit patches by

```
score = (1 − peak/peak_max) + w_chroma · ‖c_s − c_l‖₁ + w_dist · ‖x_s − x_l‖/diag
```

* **peak** = maximum of the normalised cross-power spectrum,
  `max ℱ⁻¹[F_s·conj(F_l) / |F_s·conj(F_l)|]` — shift-invariant phase
  correlation, replacing the parent's raw phase distance, which is dominated by
  translation (C3).
* **c** = normalised chromaticity `mean_RGB / Σ mean_RGB`: invariant to a
  scalar `k`, but *not* invariant to albedo. This is the term that does the job
  the parent assigned to phase (C2).

Take the best `k`. **Confidence** for the patch is

```
conf = clip((peak_best / median(peak_candidates) − 1) / (ρ − 1), 0, 1) · exp(−chroma_dist / σ_c)
```

i.e. how far the chosen match stands above this image's own match noise floor,
gated by reflectance agreement.

**Fallback when no adequate match exists:** `conf → 0`, which by Stage 5 means
the patch defers to the ring prior instead of transferring from a wrong-albedo
reference.

### Stage 4 — Low-frequency amplitude transfer

Per shadow patch, with `M_β` the low-band mask at `f_β` cycles/pixel:

```
A_new = M_β ⊙ mean(A_lit_matched) + (1 − M_β) ⊙ A_shadow
φ_new = φ_shadow                                     [unchanged]
p_corrected = ℱ⁻¹(A_new · e^{jφ_new})
```

Phase is untouched **at the patch level**. Note that this does *not* mean the
final output preserves phase: `I ⊙ G` convolves the spectrum and perturbs phase
(C4). The correct claim is that a smooth positive gain field perturbs structure
little, and `struct_ncc` in §5 measures how little.

### Stage 5 — Gain field and output

1. Per patch and channel, `g_patch = median(p_corrected) / median(p_input)` —
   a ratio of robust statistics, never a per-pixel division (C6).
2. Blend with the instance's boundary-ring prior under the match confidence
   (C12):
   `log g = conf · log g_patch + (1 − conf) · log g_ring`.
3. Accumulate Hann-weighted into a log-gain field; Gaussian-smooth at
   `0.5 · s_i`; exponentiate; clamp to `[1, G_max]`.
4. Feather across the penumbra with a smoothstep ramp so the gain reaches
   exactly 1 outside the instance. (A post-hoc blur of the gain field leaks
   `G>1` into the lit region and measurably costs non-shadow PSNR — this is
   why the ramp is C¹ rather than the field being blurred.)
5. Compose instances multiplicatively: `G = ∏_i G_i`. Illumination composes
   multiplicatively, and 1 is the natural no-op outside an instance.
6. `I_out = I ⊙ G`.

All of this runs in **linearised sRGB**; the multiplicative model is a statement
about radiance, not about gamma-encoded code values. The `linear` switch exists
and both settings are reported for the baselines.

---

## 5. Self-verification (primary contribution)

No loss function at inference. Instead, invariants that follow from the
method's own construction, none of which need ground truth. The parent's list
is kept where it can fail and replaced where it cannot.

### 5.1 Construction checks

| Check | e1 signal | Why the parent's version was replaced |
|---|---|---|
| Structure preservation | `struct_ncc` — NCC of `∇log I` vs `∇log I_out` in the umbra | "φ identical before/after" is trivially true per patch and false for the output (C4) |
| Boundary continuity | `boundary_contour` — gradient energy on the penumbra ring ÷ a control ring in the lit region | Kept, but made a ratio against an in-image control so it compares across scenes |
| Reconstruction admissibility | `degenerate` — fraction of instances with no admissible patch pair at any scale | New; also serves as §6.2's applicability test (C11) |

### 5.2 Physical checks — measured **before** the clamp

| Check | e1 signal |
|---|---|
| Direction (`G ≥ 1`) | `frac_patch_gain_below_1` |
| Magnitude bound | `frac_patch_gain_above_max`, `patch_gain_p95` |
| Chromatic consistency (`g_R ≥ g_G ≥ g_B`, ambient being bluer than direct sun) | `chroma_order_ok`, `chroma_order_margin` |

A clamped gain always satisfies its own clamp. Only the violation mass before
clamping carries information (C5).

### 5.3 Synthetic-shadow round trip (strongest signal)

Take a **lit** region, darken it by a per-channel factor drawn from this image's
own estimated shadow strength, give it a Gaussian penumbra, and run the full
pipeline on it. Ground truth is free — the untouched original.

This replaces the parent's "treat an untouched lit patch as if shadowed", which
is near-identity by construction and never loads the illumination estimate —
the one component that fails (C8).

Signals: `rt_psnr`, `rt_gain_db` (dB recovered over the degraded input).

### 5.4 Reject and fallback

Signals → one score (§5.5). On rejection: defer to the ring prior, then to a
DC-only correction, then output the input unchanged. A method that declines an
input is more useful in deployment than one that silently hallucinates.

Note that in e1 the confidence is *also* used inside the estimator (C12), so
rejection is the last resort rather than the only response.

### 5.5 Validating the validator — **this experiment is the point**

Correlate the reject signal against true per-image PSNR over the shadow region
on the 540-image ISTD+ test split.

Split discipline, without which the result would mean nothing:

* per-signal correlations are reported on test;
* the **combiner** (ridge regression over the ~22 signals) is fitted on a
  **train**-split run and applied unchanged to test;
* the accuracy/coverage curve is reported against an oracle ordering (sort by
  true PSNR) and against random, because a curve with no reference is
  unreadable.

Implementation: [`analyze_selfcheck.py`](analyze_selfcheck.py).

---

## 6. Scope

### 6.1 Out of scope (unchanged)

Ambient lighting variation — a room dimmed by an overhang, a gradual falloff
with no boundary — is a different problem. There is no characteristic radius
(`r → ∞`, giving DC-only correction) and often no lit reference in frame, so
the intra-image asymmetry of §1.1 disappears. The field already separates these
tasks. ISTD+ contains none of it, so this is not exercised in e1.

### 6.2 Applicability test

Replaced by the geometric degeneracy test in Stage 2 (C11): no admissible lit
patch, or no admissible shadow patch at any scale, means the intra-image
reference does not exist for that instance. It folds into the reject logic as
the `degenerate` signal and costs nothing.

---

## 7. Experimental protocol

### 7.1 Dataset — ISTD+ only

| Purpose | Split | N |
|---|---|---|
| Stage 0 calibration, `a_s`/`a_b` selection, reject-combiner fitting | ISTD+ train | 1330 |
| Every reported number | ISTD+ test | 540 |

ISTD+ = original ISTD shadow images and masks + Le & Samaras' colour-adjusted
shadow-free images. Original ISTD is not usable here: its shadow-free frame
carries a global colour/exposure shift relative to the shadow frame, which
would corrupt any spectral measurement and would score a method partly on
guessing that shift. Assembly: [`tools/prepare_istd_plus.py`](../../tools/prepare_istd_plus.py).

Deferred to later rounds: DESOBA (the right dataset for *establishing* the
geometry law — C7, C13), SRD, WSRD+, INS.

### 7.2 Baselines — all training-free, all mask-guided

Implemented in [`baseline/methods.py`](../../baseline/methods.py), tuned on
ISTD+ train, run once on test through the same harness as e1:

* `input` — the untouched shadow image (lower bound)
* `global_gain` — one per-channel gain from whole-region statistics
* `boundary_gain` — per-instance gain from thin rings either side of the edge
* `guo_paired_regions` — **the critical baseline**; region pairing plus
  untrained relighting, in the spirit of Guo, Dai and Hoiem. It pairs shadow
  and lit regions and does not train the relighting, exactly as we do; it
  differs by working in pixel space with no spectral decomposition and no
  self-check
* `gradient_domain` — log-domain gradient nulling across the shadow edge plus
  Poisson reintegration, in the spirit of Finlayson, Hordley and Drew

Plus two **oracles**, clearly labelled, which are ceilings and not baselines:

* `oracle_region_gain` — the true per-instance constant `k` from the GT: the
  ceiling for any constant-`k`-per-instance method
* `oracle_smooth_gain` — the true per-pixel ratio, low-passed: the ceiling for
  the whole smooth-gain-field family (this is §8.4's question)

Supervised networks (ShadowFormer, BMNet, ShadowRefiner, HomoFormer, Refusion)
are **not** in e1. They answer a different question and would need weights and
a GPU; the number that matters for them is published and can be quoted in a
paper. e1's job is to establish whether the method works at all.

### 7.3 Metrics

Reported for **shadow region / non-shadow region / whole image**, per image then
averaged, at **256×256** (the literature protocol) and at native 480×640:

* `psnr`, `ssim`
* `mae_lab` — **this is the number the literature prints as "RMSE"** (MAE in
  CIE L\*a\*b\*, L∈[0,100]); see C9
* `rmse_lab`, `rmse_rgb` — actual RMSE, reported because the name should mean
  what it says
* `lpips` (AlexNet) on the whole image

**Latency** is reported for every method and every ablation arm: mean, median,
std, min, max, p90/p95/p99, and fps, measured around the inference call only at
native resolution, after warm-up, excluding I/O and metric computation. For e1
the run is repeated with verification off, so the cost of self-verification is
a separate line rather than a hidden tax.

Reliability metrics (ours, §8.3): Pearson and Spearman of the reject signal
against true PSNR; accuracy/coverage curve; mean absolute error of the
predicted PSNR in dB.

---

## 8. Gating experiments

Each can invalidate the project cheaply. All five run on ISTD+ test unless
stated.

### 8.1 Does geometry buy anything? (four arms, re-specified)

| Arm | Patch size | Band `f_β` |
|---|---|---|
| 1 | fixed | fixed (cyc/px) |
| 2 | fixed | geometry-derived |
| 3 | geometry-derived | fixed (cyc/px) |
| 4 | geometry-derived | geometry-derived |
| 3-lit | geometry-derived | none (`f_β` = Nyquist) |

The parent's arm 3 says "band: none". Taken literally that transfers the entire
amplitude spectrum, which imports the reference patch's texture wholesale — a
degenerate arm, not a "no band limiter" arm, because patch size limits the
*low* end and cannot substitute for a high cutoff (C1). It is run as **3-lit**
to show what it does, and the real arm 3 holds `f_β` fixed in absolute units.

Interpretation:
* arm 3 ≈ arm 4 ⇒ drop `f_β` as a geometry-derived quantity; the claim becomes
  geometry-derived *patch size*, which is cleaner because patch size must be
  chosen anyway
* arm 1 ≈ all ⇒ the geometry link does not pay for itself; demote to
  implementation detail and lead with §5
* arm 4 > all ⇒ both mechanisms contribute

### 8.2 The geometry scatter

`f_c` vs `1/r` over ISTD+ train instances, for the literal estimator and the
corrected one (C13). Already run: [`calibrate.py`](calibrate.py),
[results/stage0_calibration.json](results/stage0_calibration.json),
[results/gating_8_2_scatter.png](results/gating_8_2_scatter.png).

The parent expects "clean inverse relationship → contribution 3 holds; noisy →
escalate or abandon". The answer on ISTD+ is *in between and estimator
dependent*, which is itself the finding.

### 8.3 Does the self-check work?

Correlate the reject signal against true per-image PSNR (§5.5). **This is the
gate on the headline contribution.** [`analyze_selfcheck.py`](analyze_selfcheck.py).

### 8.4 Oracle upper bound

Covered by `oracle_region_gain` and `oracle_smooth_gain` in `baseline/`. If the
smooth-gain oracle is poor, the multiplicative-gain formulation itself is
inadequate and no amount of matching improvement will save it. Running this
in the baseline rather than in e1 is deliberate: it bounds *every* method in
this study at once, including the baselines.

### 8.5 Mask sensitivity

Dilate/erode the mask fed to the method by `n ∈ {−12, −6, 0, +6, +12}` pixels;
metrics are always computed against the unperturbed mask. Because our scale
derives from mask geometry we should be **more** mask-sensitive than a typical
mask-based method; collapse at small `n` is a fatal weakness and must be found
now, not in review.

---

## 9. Positioning (unchanged from the parent, with one addition)

The parent's table stands. FDA swaps low-frequency amplitude between two
*images* for domain adaptation; ours is intra-image and region-to-region. Guo
et al. pair regions in pixel space with no spectral decomposition and no
self-verification. CANet is the closest prior art — contextual patch matching
with learned feature transfer — and the defence is that we transfer spectral
statistics with no trained descriptor and that our formulation admits
verification a learned transfer cannot.

**Addition.** The verification claim now has to be stated as a *measured*
claim, not a structural one. "Learned methods structurally cannot self-verify"
is too strong: a network can be given an uncertainty head. The defensible
version is that our checks derive from the method's construction and need no
extra training, no extra parameters and no held-out calibration of the
uncertainty itself — and that §8.3 shows they work.

---

## 10. Limitations

Carried over from the parent, with e1's additions:

1. Will not beat supervised SOTA on in-domain PSNR. Supervised methods reach
   31–36 dB on ISTD+.
2. Mask-sensitive by construction (§8.5).
3. Deep umbra is unrecoverable where texture fell below the noise floor.
4. No adequate reference when an albedo appears only inside the shadow. Now
   detected by the chroma term and handled by confidence decay to the ring
   prior rather than by rejection alone.
5. Ambient lighting variation is out of scope (§6.1).
6. Indoor/indirect illumination violates the single-source model; not exercised
   on ISTD+.
7. Multi-scale single components (foliage) need morphological splitting; ISTD+
   has almost none, so e1 does not test it and must not claim it.
8. **New.** `C` is calibrated on a dataset whose instance radii span ~9×. Any
   `1/r` claim from e1 alone is weak evidence (C7, C13).
9. **New.** The boundary-ring prior inside the estimator (C12) means part of
   the reported quality is not attributable to amplitude transfer. The
   `use_prior=False` arm exists so the split is visible; reporting only the
   blended number would overstate the spectral contribution.

---

## 11. What e1 is trying to find out

The parent's §11 concludes that the geometry idea has weakened and
self-verification has strengthened. e1 was built to test both, and the Stage 0
measurement (C13) already shifted one of them: the geometry link is not absent,
but it is weak, and the parent's own measurement procedure reported it with the
wrong sign. That is a warning about the *measurements*, not only about the
idea — and it is a good argument for running §8.3 with the split discipline of
§5.5 rather than trusting a correlation computed the convenient way.

Results: [RESULTS.md](RESULTS.md).
