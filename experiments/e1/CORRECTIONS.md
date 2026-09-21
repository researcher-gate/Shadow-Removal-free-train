# Corrections to `guideline.md`

What follows is a review of the root `guideline.md` against the physics it
cites and against what the code actually does when you write it. Each entry
says what the guideline claims, why it does not hold, and what e1 does instead.
Entries marked **[measured]** are backed by a number in
[results/](results/) rather than by argument alone.

The framing of the guideline survives this review. Section 1's asymmetry
argument is right, and section 11's instinct — that self-verification is the
durable contribution and the geometry link is the fragile one — is right, and
is now supported by measurement rather than intuition. What does not survive is
a set of specific technical claims, three of which would have produced a
silently broken implementation.

---

## Severity 1 — would break the method

### C1. Patch size and `β` band-limit *opposite* ends of the spectrum

> §2 contribution 3, §Stage 2 "Note on redundancy", §8.1 "Known tension", §11

The guideline says patch size is "itself a band limiter — an `s × s` patch
cannot represent frequencies below `1/s`" and concludes that a separate `β` may
be redundant. The premise is correct and the conclusion does not follow from
it. An `s × s` DFT has bins at `k/s` for `k = 0 … s/2`: patch size sets the
**lowest** representable frequency, `β` caps the **highest** transferred one.
They are not two names for one knob; they clamp opposite ends.

The only place a genuine coupling exists is in the parameterisation. If `β` is
expressed as a count of DFT bins, its absolute cutoff `β/s` moves when `s`
moves, and the two axes get entangled by bookkeeping rather than by physics. If
`β` is expressed in absolute cycles/pixel, they are independent.

The consequence for §8.1 is direct: the four-arm ablation as written cannot
answer its own question, because arms that change `s` also silently change the
band. §11's conclusion that the geometry contribution "has been substantially
weakened" by this supposed redundancy rests on the same confusion, so that
conclusion is unsupported — the geometry contribution has to be judged on the
evidence in C13, not on this argument.

**e1:** `f_beta` is always in cycles/pixel (`IATConfig.a_b`, `fixed_beta`), and
patch size is a separate parameter (`a_s`, `fixed_s`). The ablation then
genuinely varies one at a time.

### C2. Phase similarity is invariant to albedo, so it cannot match albedo

> §2 contribution 2, §Stage 3 "Matching criterion", §4

> "Phase encodes structure and is approximately illumination-invariant, so
> phase distance approximates albedo/texture distance."

The first half is the reason the method works; the second half inverts it.
Under `I_shadow = k · L · R` with `k` constant over the umbra, multiplying an
image by a positive scalar scales every amplitude coefficient and leaves every
phase coefficient untouched. That is exactly why phase is illumination
invariant — and it is equally why phase is invariant to **albedo level**. A
patch of light-grey-on-white tiling and a patch of dark-grey-on-black tiling of
the same geometry have identical phase spectra.

So phase distance measures *structure*, not reflectance. The failure mode the
guideline names in §10.4 — "no adequate reference when an albedo appears only
inside the shadow" — is precisely the failure that phase matching is blind to,
by construction. A matching criterion made only of phase will happily pair a
dark shadowed road with a bright lit wall of similar texture and then transfer
the wall's brightness onto the road.

**e1:** matching keeps phase for structure and adds an explicitly
reflectance-sensitive, illumination-invariant term: normalised chromaticity
`c = mean_RGB / sum(mean_RGB)`, which is unchanged by a scalar `k` but does
separate materials. Ranking is `phase agreement + w_chroma · chroma distance +
w_dist · spatial distance`. §2's contribution 2 should be restated as
"phase similarity as an illumination-invariant **structure** proxy, paired with
a chromatic reflectance proxy".

### C3. A raw distance between phase spectra is dominated by translation

> §Stage 3: `match(p_shadow) = argmin_k d(φ(p_shadow), φ(p_lit))`

By the Fourier shift theorem, translating a patch by `(Δx, Δy)` multiplies its
spectrum by `exp(-2πi(f_x Δx + f_y Δy))`, i.e. adds a linear phase ramp across
the whole spectrum. Two patches cut from different places in the image are, by
definition, translated relative to each other. `d(φ_s, φ_l)` as written is
therefore dominated by the relative offset of the two crops and carries almost
no information about their content. Implemented literally, Stage 3 selects
references essentially at random.

**e1:** the similarity is the peak of the normalised cross-power spectrum,
`max ℱ⁻¹[ F_s · conj(F_l) / |F_s · conj(F_l)| ]` — classical phase correlation.
It uses phase only (so it stays illumination invariant and learning free) and
is shift invariant by construction. The peak is scored **relative to the median
peak over the candidate pool**, so "how confident is this match" means "how far
above this image's own noise floor", which is comparable across images in a way
a raw peak value is not.

---

## Severity 2 — invalidates a stated result or claim

### C4. The construction invariants of §5.1 are untestable or trivial

> §5.1, table row "Phase preservation: `φ` identical before/after per patch"

Two separate problems.

*At patch level it is trivially true.* Stage 4 sets `φ_new = φ_shadow` and then
inverse-transforms. The phase of the result equals `φ_shadow` to floating-point
precision, always, for every input, including inputs the method handles
terribly. A check that can only fail on a typo is a unit test, not a quality
signal.

*At output level it is false.* The output is `I_out = I ⊙ G` (§Stage 5).
Multiplication by a spatial field is convolution in the Fourier domain, which
changes amplitude **and** phase. So "phase carries structure and we never touch
it" is not a property of the delivered image — the gain field does perturb
phase, mildly at low frequency and less at high. The honest statement is that a
smooth positive gain field perturbs structure *little*, not that it preserves
phase exactly. The same applies to the "Out-of-band amplitude unchanged" row.

**e1:** §5.1's row is replaced by two checks that can actually fail on a real
image and vary with quality:

* `struct_ncc` — normalised correlation between `∇log I` and `∇log I_out`
  inside the umbra. A correct smooth gain field leaves it near 1; a gain field
  with structure of its own (bad matches, stitching seams, a `β` set too high)
  pushes it down.
* `boundary_contour` — median gradient energy on the penumbra ring divided by
  the same on a control ring further out in the lit region. ≈1 means the
  boundary is invisible; ≫1 means the residual shadow outline survived. This
  is the guideline's own "boundary continuity" row, but expressed as a ratio
  against an in-image control so it is comparable across scenes.

### C5. `G(x) ≥ 1` and the magnitude bound are enforced, not checked

> §5.2

Any implementation must clamp the gain — an unclamped `corrected / input` is
unbounded in the deep umbra. But once clamped, the check "is `G ≥ 1`
everywhere?" always passes, and reports nothing. As written, §5.2 contributes
zero bits to the reject signal.

**e1:** the informative quantity is the **pre-clamp violation mass**. The
per-patch raw estimates are recorded before clamping, and the exported signals
are `frac_patch_gain_below_1`, `frac_patch_gain_above_max`, `patch_gain_p95`.
These do vary across images and are what the combiner sees.

### C6. Stage 5's per-pixel `corrected / input` is numerically unusable

> §Stage 5.1: "divide the corrected estimate by the input to obtain a raw gain map"

In the deep umbra the input is a handful of DN above black. Dividing by it
amplifies sensor noise by the same factor the method is trying to apply, and
the result is a gain map whose high-frequency content is pure noise. Feathering
(§Stage 5.2) smooths the boundary, not the interior.

**e1:** the gain is a *ratio of robust statistics*, not a ratio of pixels. Per
patch and per channel the estimate is `median(corrected) / median(input)`; the
per-patch scalars are accumulated into a Hann-weighted **log**-gain field and
Gaussian-smoothed at a fraction of the patch size. The illumination field is
assumed smooth, so nothing is lost by refusing to represent it at pixel
resolution, and the division is never taken where the denominator is small.

### C7. Calibrating Stage 0 on DESOBA and testing on ISTD+ imports a domain gap

> §7.1

The caution against DESOBAv2 (inpainted shadow-free images carry the inpainter's
frequency signature) is correct and well argued. But for a study whose headline
numbers are on ISTD+, fitting `C` on DESOBA means any weakness in the geometry
link can be blamed on cross-dataset transfer rather than on the link itself,
which makes the gating experiment unable to gate.

**e1:** Stage 0 runs on the ISTD+ **train** split, evaluation on ISTD+ **test**.
The cost is honest and stated: ISTD+ instance radii span about 9× between the
5th and 95th percentile, and a `1/r` law demonstrated over that little leverage
is weak evidence. DESOBA remains the right dataset for *establishing* the law;
it is the wrong one for calibrating a model that will be tested somewhere else.

### C8. The reference round trip of §5.3 is biased optimistic

> §5.3 "Take a **lit** patch, treat it as if shadowed, run the full pipeline"

A lit patch run through the pipeline is matched against other lit patches under
the same illumination. The transferred amplitude is therefore already correct,
and the pipeline returns the patch nearly unchanged almost regardless of how
badly it would handle a real shadow. The check exercises tiling, windowing and
stitching — real things, but not the failure mode. The one component that
actually breaks, the illumination-magnitude estimate, is never put under load.
"Ground truth is free" is true; "the check is informative" does not follow.

**e1:** the round trip is run on a **synthetic shadow**. A lit region is
selected (the real shadow mask translated into lit territory, else a disk of
equal area), darkened by a per-channel factor taken from this image's own
estimated shadow strength, given a Gaussian penumbra, and pushed through the
full pipeline. Ground truth is still free — it is the untouched original — and
the check now loads the component that fails. Exported as `rt_psnr` and
`rt_gain_db` (dB recovered relative to the degraded input).

---

## Severity 3 — imprecision that would confuse a comparison

### C9. "RMSE" in this literature is a mean absolute error in Lab

> §7.3

Numbers printed as RMSE in shadow-removal papers are MAE in CIE L\*a\*b\*. The
authors of the evaluation code say so themselves — the `cvlab-stonybrook/SID`
README: *"the RMSE evaluation code that many papers have been using (including
ours) is actually calculating Mean Absolute Error."* A paper that computes a
true RMSE and puts it in the same column as published numbers will look ~30–40%
worse than it is.

Two further unstated conventions decide comparability: the literature
evaluates at **256×256**, not native 480×640; and L\*a\*b\* means L∈[0,100]
(MATLAB / skimage), not OpenCV's L∈[0,255], which differs by 2.55×.

**e1 and baseline:** `mae_lab_*` is reported as the literature's "RMSE" and
labelled as such, alongside true `rmse_lab_*` and `rmse_rgb_*`; every table is
given at both 256×256 and native resolution.

### C10. "Zero trained weights" versus a scalar fitted on paired data

> §1 one-line pitch, §2 contribution 4, §Stage 0

Stage 0 fits `C` by regression on a paired, labelled dataset. That is a
one-parameter supervised calibration. It is a perfectly defensible design — the
constraint that its input is mask geometry and never pixels is what keeps it
from becoming a learned frequency selector — but "zero trained weights" is not
the accurate phrase for it, and a reviewer will say so on page 1.

The escalation path compounds this. Step 3, "tiny MLP on geometry features
(~hundreds of parameters)", is a trained model. At that point the method is a
small supervised regressor plus a classical backend, and the central claim
("cannot overfit a training distribution, needs no paired data") is gone.

**e1:** the claim is stated as *no learned image-domain weights; one scalar
calibrated once from mask geometry*, and the escalation path is capped at step
2 (ridge on geometry features). If step 2 is not enough, the correct conclusion
is that the geometry link does not hold, not that it needs more capacity.

### C11. §6.2's applicability test asks for machinery the pipeline already has

> §6.2

Running Retinex or an intrinsic-image decomposition at inference to decide
whether the input is a cast shadow or ambient dimming is heavier than the
removal itself, and it re-solves a problem the mask has already answered. The
useful version of the test is geometric and costs nothing: no lit region within
reach of an instance, or an instance covering most of the frame, means the
intra-image reference of §1.1 does not exist.

**e1:** the degenerate condition (no admissible shadow patch or no admissible
lit patch at any scale) is detected during tiling and exported as the
`degenerate` signal, which feeds the reject logic.

### C12. Self-verification is only wired into a post-hoc reject

> §5.4

The guideline computes a confidence, then uses it only to accept, retry or
decline. But the same confidence is available *per patch*, before the output
exists, and a per-patch confidence is exactly the weight a robust estimator
wants. Using it only at the end throws away most of its value: a single badly
matched patch corrupts the output, and the method's only response is to reject
the whole image.

**e1:** the per-patch match confidence blends the transfer estimate with the
boundary-ring prior in the log-gain domain,

```
log g = conf · log g_transfer + (1 − conf) · log g_ring
```

so a patch that cannot find a trustworthy reference degrades gracefully toward
a conservative estimate instead of poisoning the instance. `use_prior=False`
recovers the literal guideline pipeline, and the ablation reports both.

### C13. **[measured]** The Stage-0 measurement as specified cannot see the effect it is looking for

> §Stage 0 steps 2–3, §8.2

Stage 0 computes "`R`'s radial band energy profile using soft Gaussian bands"
and takes "the band centroid `f_c`". Band *energy* means the sum of power over
the band. In two dimensions the number of DFT bins in an annulus at frequency
`f` grows like `f`, so for any residual with a noise floor — and a residual
between two separately captured frames always has one — the summed profile
rises with `f` regardless of the signal, and its centroid measures the noise
floor and the Nyquist limit rather than the shadow.

Measured on 473 ISTD+ train instances
([results/stage0_calibration.json](results/stage0_calibration.json)):

| estimator | Pearson vs `1/r` | Spearman | R² | median `f_c` |
|---|---|---|---|---|
| summed band energy (guideline, literal) | **−0.028** | **−0.168** | 0.001 | 0.141 |
| + mask-restricted residual | −0.087 | −0.282 | 0.008 | 0.138 |
| mean power per band (bin-count corrected) | +0.216 | +0.360 | 0.046 | 0.034 |
| + mask-restricted | +0.335 | +0.357 | 0.112 | 0.034 |
| + noise floor subtracted | +0.423 | +0.467 | 0.179 | 0.032 |
| + noise floor subtracted, mask-restricted | **+0.547** | **+0.476** | **0.299** | 0.032 |

The literal procedure returns a centroid of 0.14 cycles/pixel — a third of the
way to Nyquist — for shadows 100–300 pixels across, whose spectral content
belongs three decades lower. It is measuring noise, and it reports the
geometry link as *absent, with the wrong sign*. Gating experiment 8.2 run as
written would have killed contribution 3 for the wrong reason.

**e1:** the Stage-0 estimator divides band energy by the band's bin-weight sum,
subtracts the high-frequency plateau (median mean-power above 0.35 cyc/px), and
restricts the residual to a soft instance mask before transforming — the last
step because §3.1's own statement that the degradation is confined to the mask
implies everything outside it is pair misalignment, not signal. With that
estimator the relationship is real but weak: ρ = +0.48, R² = 0.30, `C ≈ 1.52`.
Both numbers are reported; the weak version is the one that should be believed,
and settling it needs DESOBA's instance-scale range (C7).

---

## What this does to the contribution ranking

The guideline ranks: (1) self-verification, (2) intra-image reference framing,
(3) geometry-derived spectral scale, (4) the pipeline.

* **(1) stands, and is now the only candidate for a headline.** Nothing in this
  review weakens it; C4, C5 and C8 strengthen it by replacing three signals
  that could not fail with three that can.
* **(2) needs restating.** The framing is sound; the specific mechanism offered
  for it (phase as an albedo proxy) is wrong (C2), and the specific distance
  offered for that mechanism does not work (C3). Fixed, it survives.
* **(3) is weaker than §11 feared, but for a different reason than §11 gives.**
  §11 demotes it on the redundancy argument, which is itself wrong (C1). The
  real evidence is C13: the effect is present but weak (R² = 0.30) on a dataset
  with 9× of radius leverage, and it is invisible to the guideline's own
  measurement procedure.
* **(4) is where the work is,** as the guideline already says.

§11's closing recommendation — lead with self-verification — is the right call.
This review moves the reason for it from intuition to measurement.
