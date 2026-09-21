# Self-Verifying Training-Free Shadow Removal via Intra-Image Amplitude Transfer

**One-line pitch.** A shadow image contains its own reference. We exploit that to remove shadows with zero trained weights, and — because the method is algorithmic rather than learned — it can check its own output and report when it failed.

---

## 1. Motivation

### 1.1 The asymmetry nobody has exploited

Zero-shot low-light enhancement methods require external priors (bright-channel priors, CLIP embeddings, pretrained diffusion priors) because a uniformly dark image contains no correct reference anywhere in the frame. There is nothing to compare against.

A shadow image is different. The lit region is the same scene, the same camera, the same instant, the same materials. **The reference is already in the image.** This asymmetry is why reference-free correction is tractable for shadows and not for low-light, and it is the foundation of everything below.

### 1.2 Why training-free is worth pursuing here

Supervised shadow removal has a data problem the field openly acknowledges. Collecting perfectly aligned shadow/shadow-free pairs is impractical outside controlled scenes; accurate masks are hard to annotate on complex imagery; and supervised methods degrade sharply under cross-dataset transfer. A method with no learned weights cannot overfit a training distribution, needs no paired data at inference, and runs in seconds.

The trade-off is honest: we will not beat supervised state-of-the-art on in-domain PSNR. The claim is reliability and generalization, not peak fidelity.

### 1.3 The gap this fills

Every learned method is opaque at inference. Given an output, there is no way to ask the network "did you get this right?" without ground truth. Deployed shadow removal — in photo pipelines, in preprocessing for detection/segmentation — has no ground truth. A method that silently hallucinates on hard inputs is worse than one that declines them.

**No published shadow removal method reports its own confidence.** That is the gap.

---

## 2. Contributions

Ranked by how well each survives scrutiny.

1. **Ground-truth-free self-verification.** A set of invariants derived from the method's own construction (spectral, physical, and round-trip) that together predict per-image output quality. Validated by correlating the reject signal against true PSNR on held-out data. This is the headline.

2. **Intra-image reference framing.** Formalizing shadow removal as illumination transfer from matched lit regions of the same image, with phase similarity as an illumination-invariant albedo proxy.

3. **Geometry-derived spectral scale.** Shadow spatial extent determines the spectral extent of its degradation via the Fourier scaling property, so the operating scale can be *computed* from the mask rather than learned or tuned. **Conditional** — see §8.1; if the gating ablation shows this is redundant with patch size, it is demoted to an implementation detail.

4. **A training-free amplitude transfer pipeline** as the demonstration vehicle. Lowest novelty; this is how the above are shown to work, not the claim itself.

---

## 3. Physical model and notation

### 3.1 The log-multiplicative model

Under a Lambertian assumption, an image is illumination times reflectance:

```
I(x) = L(x) · R(x)
```

A cast shadow attenuates direct illumination by a factor `k(x) < 1`:

```
I_shadow(x) = k(x) · L(x) · R(x)
```

Taking logs converts the multiplicative degradation to additive:

```
log I_shadow(x) − log I_free(x) = log k(x)
```

With `k` approximately constant over the umbra interior, the residual is `log k · M(x)` where `M` is the shadow mask. **This is the central fact: the degradation is spatially confined to the mask and spectrally shaped by the mask.**

This is homomorphic filtering (Oppenheim, Schafer & Stockham), restated. It is also why the amplitude spectrum carries illumination while phase carries structure — the fact underpinning the Fourier-based shadow removal literature.

### 3.2 Where the model breaks

Stated up front because these define the method's scope:

- **Chromatic shift.** Ambient skylight is bluer than direct sun, so `k` is per-channel, not scalar.
- **Penumbra.** `k` varies smoothly across the shadow boundary; "constant over the umbra" holds only in the interior.
- **Deep umbra.** Where attenuation drives signal below the sensor noise floor, texture is genuinely destroyed, not merely darkened. No multiplicative correction can recover it.
- **Indirect illumination.** Indoor scenes with bounce light violate the single-source assumption entirely.

### 3.3 Notation

| Symbol | Meaning |
|---|---|
| `I` | input shadow image |
| `M` | binary shadow mask |
| `M_i` | i-th connected shadow instance |
| `r_i` | characteristic radius of instance i, `√(area/π)` |
| `p` | a windowed patch |
| `A(p), φ(p)` | amplitude and phase spectra of patch p |
| `β` | low-frequency band fraction retained for transfer |
| `G` | multiplicative illumination gain field |
| `s_i` | patch size for instance i |

---

## 4. Method

### Stage 0 — Offline calibration (run once)

Fits the single scalar that relates shadow geometry to spectral scale.

**Procedure.** On a paired dataset with per-instance masks:

1. Compute the log residual `R = log I_shadow − log I_free`.
2. Compute `R`'s radial band energy profile using soft Gaussian bands (overlapping bands with linearly increasing center frequency and bandwidth, avoiding hard band boundaries).
3. Extract the band centroid `f_c` per shadow instance.
4. Measure `r_i = √(area/π)` from that instance's mask.
5. Regress `f_c` against `1/r`. The slope is `C`.

**Output:** one scalar. Total learned parameters in the entire system.

**Escalation path** if a single parameter is insufficient, in order — stop at the first that works:

1. Linear fit, 1 parameter
2. Ridge regression on geometry features: `√(area/π)`, perimeter-to-area ratio, eccentricity, mean intensity drop (~5–20 parameters)
3. Tiny MLP on the same geometry features (~hundreds of parameters, seconds to fit on CPU)

**Hard constraint:** the predictor's input is mask geometry only, never pixels. The moment image content enters, this becomes a learned frequency selector and the contribution collapses into prior work.

---

### Stage 1 — Mask and instance decomposition

1. Obtain `M` from an off-the-shelf shadow detector, or from dataset annotations where available. **Not trained by us** — a component, not a contribution.
2. Connected components → shadow instances `{M_1, ..., M_n}`.
3. Per instance: erode to obtain the umbra core, dilate-minus-erode to obtain the penumbra ring.
4. Per instance: compute `r_i` and derived scale parameters.

**Why per-instance and not per-patch.** A patch deep inside a large shadow sees only shadow; its local radius is undefined. Geometry must be measured on the instance, and patches inherit the scale of the component they belong to.

**Multi-scale within one component.** A tree shadow is one connected blob containing both a broad trunk shadow and fine leaf gaps; a single `r_i` misrepresents it. Detect via high perimeter-to-area ratio, then either split by morphological opening or fall back to a conservative wider band.

---

### Stage 2 — Scale-adaptive tiling

Per instance, tile into overlapping patches with a Hann window.

- Patch size `s_i` derived from `r_i` (see §8.1 for the ablation deciding whether `s_i`, `β`, or both carry the geometry signal).
- Overlap ≥ 50% so the Hann windows sum to unity on reconstruction.
- Classify patches as shadow (inside the umbra core) or lit reference (outside the dilated mask).

**Note on redundancy.** Patch size is itself a band limiter — an `s × s` patch cannot represent frequencies below `1/s`. This is unavoidable and must be accounted for before claiming that a separate `β` does independent work.

---

### Stage 3 — Reference matching

For each shadow patch, find the `k` best lit reference patches.

**Matching criterion: phase spectrum similarity.** Phase encodes structure and is approximately illumination-invariant, so phase distance approximates albedo/texture distance. This gives albedo-aware matching with no learned descriptor and no semantic model.

```
match(p_shadow) = argmin_k  d( φ(p_shadow), φ(p_lit) )
```

**Fallback when no adequate match exists** (an albedo present only inside the shadow): widen the search to coarser scales, and if still unmatched, flag the patch and let the reject logic handle it rather than transferring from a wrong-albedo reference.

---

### Stage 4 — Low-frequency amplitude transfer

Per shadow patch, with `M_β` the low-frequency band mask:

```
A_new = M_β ⊙ mean(A_lit_matched)  +  (1 − M_β) ⊙ A_shadow
φ_new = φ_shadow                              [unchanged]
p_corrected = F⁻¹( A_new · e^{jφ_new} )
```

Then inverse FFT per patch and stitch with the Hann window.

**Why phase is untouched.** Phase carries structure. Preserving it is what stops the transfer from importing the reference patch's texture, and it is also what makes the phase-preservation invariant (§5.1) checkable.

---

### Stage 5 — Gain field and output

1. Per instance, divide the corrected estimate by the input to obtain a raw gain map.
2. Feather across the penumbra ring so the gain converges smoothly to 1.0 at the boundary.
3. Set gain to exactly 1 outside the instance.
4. **Combine instances multiplicatively:** `G = ∏_i G_i`.
5. Output: `I_out = I ⊙ G`.

**Why multiplication, not averaging.** Illumination composes multiplicatively, so a region under two overlapping shadows has both attenuations undone in sequence. It also makes the out-of-instance value of 1 a natural no-op.

**Why a gain field rather than the corrected image directly.** The output is the *original pixels* times a smooth scalar field. High-frequency texture is preserved by construction, not by a learned detail-restoration stage.

---

## 5. Self-verification (primary contribution)

The method has no loss function at inference. Instead it checks invariants that follow from its own construction. None require ground truth.

### 5.1 Construction invariants

Violations indicate implementation or stitching failure.

| Check | Expectation | Failure meaning |
|---|---|---|
| **Phase preservation** | `φ` identical before/after per patch | Structure was altered — transfer leaked |
| **Out-of-band amplitude** | Amplitude outside `M_β` unchanged | Band masking failed |
| **Boundary continuity** | Gain → 1.0 smoothly across the penumbra ring | Residual shadow contour — the classic failure mode that forces learned methods to add refinement stages |

### 5.2 Physical constraints

Violations indicate a physically impossible correction.

| Check | Expectation | Failure meaning |
|---|---|---|
| **Direction** | `G(x) ≥ 1` everywhere | A shadow can only be brightened; `G < 1` is always wrong |
| **Magnitude bound** | `G` within plausible direct-to-ambient ratio (~1–6×) | Over-correction, likely a bad reference match |
| **Chromatic consistency** | Per-channel gains follow a plausible ambient/direct ratio (blue gain modestly lower than red) | Reference had different albedo, not just different illumination |

### 5.3 Reference round trip (strongest signal)

Take a **lit** patch, treat it as if shadowed, run the full pipeline on it, and compare against the untouched original.

**Ground truth is free** — the patch is its own target.

- Pipeline returns it unchanged → matching and transfer are sound for this image's content.
- Pipeline distorts a known-good region → it will distort the real shadow too.

Run across several lit patches per image to obtain a **per-image confidence score requiring no dataset labels**.

### 5.4 Reject and fallback

Combine the above into a single reject signal. On rejection, escalate:

1. Widen the band / increase patch size
2. Re-match with a different reference set
3. Correct DC only (global brightness offset)
4. Output the input unchanged

**A method that declines an input is more useful in deployment than one that silently hallucinates.**

### 5.5 Validating the validator

Offline, once: correlate the reject signal against true per-image PSNR on a paired benchmark. A strong correlation establishes that the self-check is trustworthy, licensing its use on unlabelled data.

**This experiment is the paper.** Without it, §5 is a set of heuristics; with it, it is a reliability result.

---

## 6. Scope and applicability

### 6.1 What this method is not for

**Ambient lighting variation** — a room dimmed by a roof overhang, a gradual falloff with no boundary — is a different problem. There is no characteristic radius (formally `r → ∞`, giving DC-only correction), and often no lit reference in frame at all. Without the lit reference, the intra-image asymmetry of §1.1 disappears and we are back in the low-light regime.

This is the correct scope boundary, not an evasion: the field already separates these, with Ambient Lighting Normalization treated as its own task and its own challenge track, addressed by methods that deliberately avoid shadow localization.

### 6.2 Applicability test

Cast shadows and ambient dimming differ by the **spatial scale of the illumination gradient** — penumbra transitions over pixels, ambient falloff over hundreds of pixels. Estimate the illumination field (Retinex or intrinsic decomposition) and examine where its energy sits:

- Compact support with measurable radius → method applies
- Energy concentrated at DC → flag, skip, fall back to global correction

This is an **applicability test, not a segmentation**, and it folds into the reject logic of §5.4.

---

## 7. Experimental protocol

### 7.1 Datasets

| Purpose | Dataset | Rationale |
|---|---|---|
| **Stage 0 calibration** | DESOBA | ~3,600 object-shadow instances with per-instance masks across many scales; 839 train / 160 test images |
| **Headline numbers** | ISTD+ | 1,330/540 triplets with masks; color-corrected pairs (the original ISTD color inconsistency would corrupt any spectral measurement) |
| **Scene diversity** | SRD | 2,680/408 pairs, more varied scenes; masks predicted (acceptable — we take off-the-shelf masks anyway) |
| **Generalization claim** | SRD → DESOBA | Published cross-dataset protocol; where a weightless method should look best |
| **Stress test** | WSRD+, INS | High-res complex scenes; indoor direct + indirect illumination |

**Critical caution: do not use DESOBAv2 for Stage 0.** Its shadow-free images come from a pretrained inpainting model, and inpainting carries its own frequency signature (typically smooth, high-frequency-deficient) that would contaminate a band-energy measurement. DESOBA's manual removal is the reason to prefer it despite being smaller. DESOBAv2 is acceptable for qualitative results only.

### 7.2 Baselines

- **Mask-based supervised:** ShadowFormer, BMNet
- **Mask-free supervised:** ShadowRefiner, HomoFormer
- **Latent diffusion:** Refusion (evidence for the VAE reconstruction ceiling)
- **Classical / training-free:** Guo et al. paired regions — *the critical baseline*, since it also pairs shadow and lit regions and also does not train the relighting

### 7.3 Metrics

- Standard: PSNR, SSIM, RMSE, LPIPS, reported on shadow region / non-shadow region / whole image
- **Reliability metrics (ours):** correlation between reject signal and true PSNR; accuracy/error trade-off curve as the reject threshold varies; coverage at fixed quality

The second group is where this method competes. The first group is where it must merely be respectable.

---

## 8. Gating experiments

**Run these before building the full system.** Each can invalidate the project cheaply.

### 8.1 Does geometry buy anything? (four-arm ablation)

| Arm | Patch size | Band β |
|---|---|---|
| 1 | fixed | fixed |
| 2 | fixed | geometry-derived |
| 3 | **geometry-derived** | **none** |
| 4 | geometry-derived | geometry-derived |

**Interpretation:**
- Arm 3 ≈ Arm 4 → drop `β` entirely; the contribution is geometry-derived *patch size*, a cleaner claim because patch size must be chosen regardless
- Arm 1 ≈ all others → the geometry link does not pay for itself; demote to implementation detail and lead with §5
- Arm 4 > all → both mechanisms contribute; keep as stated

**Known tension:** good phase matching may make `β` nearly irrelevant, since well-matched patches already differ mostly at low frequency. `β`'s value is inversely proportional to match quality.

### 8.2 The DESOBA scatter plot

Band centroid vs `√(area/π)` across all instances. Clean inverse relationship → contribution 3 holds. Noisy → escalate per §Stage 0, or abandon.

### 8.3 Does the self-check work?

Correlate reject signal with true per-image PSNR on ISTD+. **This is the gate on the headline contribution.**

### 8.4 Oracle upper bound

Transfer ground-truth shadow-free amplitude instead of matched-reference amplitude. If the oracle is poor, the amplitude-transfer formulation itself is inadequate and no amount of matching improvement will save it.

### 8.5 Mask sensitivity

Dilate/erode ground-truth masks by `n` pixels; plot quality vs `n`. Because our scale derives from mask geometry, we are **more** mask-sensitive than typical mask-based methods. Collapse at small `n` is a fatal weakness that must be found before submission, not during review.

---

## 9. Positioning against prior work

| Prior work | Overlap | Our difference |
|---|---|---|
| **FDA** (Yang & Soatto) | Low-frequency amplitude swap, training-free | Between two *images* for domain adaptation; ours is intra-image, region-to-region, geometry-scaled |
| **Guo et al.**, paired regions | Region pairing + relighting, classical | Pixel space, no spectral decomposition, no self-verification |
| **CANet** | Contextual patch matching + feature transfer between shadow/non-shadow patches | Learned feature transfer; ours is a spectral operation with no trained descriptor |
| **FSR-Net / NeFour / AmpFormer** | Amplitude carries lightness for shadows | All learned, all pixel/feature space |
| **SFNet / DFC** | Content-adaptive frequency band selection | They *learn* the band; we *derive* it from geometry |
| **ShadowRefiner** | FFT in shadow removal | FFT used as a fast global attention operator, not a physical decomposition; fully supervised two-stage training |
| **Latent-Fourier LLIE** | FFT on latents, amplitude = illumination | Low-light (no intra-image reference), learned diffusion |

**The differentiation that matters most:** CANet is the closest prior art and a reviewer will find it immediately. The defense is that we transfer spectral statistics with no learned transfer network and no training data, and that our formulation admits verification that a learned transfer cannot. Ablating against CANet directly is advisable.

---

## 10. Limitations (state these, don't hide them)

1. **Will not beat supervised SOTA on in-domain PSNR.** Competing methods reach 31–36 dB on ISTD+; expect mid-20s. The claim is reliability and generalization.
2. **Mask-sensitive by construction** — scale derives from mask geometry (§8.5).
3. **Deep umbra is unrecoverable** where texture fell below the noise floor. A multiplicative gain cannot invent signal.
4. **No adequate reference** when an albedo appears only inside the shadow. Detected by the matching fallback, handled by reject.
5. **Ambient lighting variation is out of scope** (§6.1).
6. **Indoor/indirect illumination** violates the single-source model; INS is the stress test.
7. **Multi-scale single components** (foliage shadows) need morphological splitting or conservative fallback.

---

## 11. Honest assessment

The geometry→spectral-scale idea, originally the intended headline, has been substantially weakened by two observations: patch size already band-limits the operation, and good phase matching reduces the marginal value of a separate band parameter. It may survive the §8.1 ablation as a patch-size claim, or it may not survive at all.

**Self-verification got stronger under the same scrutiny.** It is the least explored, it is something learned methods structurally cannot do, and it sidesteps the PSNR competition the method cannot win. It should lead.

The amplitude transfer is a recombination of published ingredients. On its own it is a workshop paper. Combined with a validated self-verification result it is a reliability contribution with a real claim.

**Next action:** run §8.2 and §8.3. Two experiments, a few days, and they determine which of three papers this is.
