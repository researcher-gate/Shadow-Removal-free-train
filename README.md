# Training-Free Shadow Removal on ISTD+

Course project 2110433. A shadow remover with **zero learned weights** that
beats every training-free baseline we could build, and beats the ground-truth
oracle of the formulation it replaces, at **94 ms/image on CPU**.

| | PSNR-S ↑ | PSNR-all ↑ | "RMSE"-all ↓ | LPIPS ↓ | latency |
|---|---:|---:|---:|---:|---:|
| do nothing | 11.86 | 20.37 | 2.92 | 0.1431 | 0.2 ms |
| best training-free baseline | 22.04 | 28.94 | 1.65 | 0.0861 | 197 ms |
| **e2 — Instance Affine Relighting** | **25.02** | **31.09** | **1.35** | **0.0593** | **94 ms** |

540 ISTD+ test images, 256×256 protocol. Full tables: [experiments/e2/RESULTS.md](experiments/e2/RESULTS.md).

---

## Quick start

```bash
conda env create -f requirements.yaml && conda activate shadow-istdplus

bash data/download_data.sh            # ISTD+ (~2.6 GB): shadow + mask + adjusted GT
python experiments/e2/run_e2.py       # evaluate e2 on 540 test images
python tools/make_figures.py --n 9    # side-by-side result panels
```

That writes predictions to `experiments/e2/outputs/e2_iar/`, metrics to
`experiments/e2/results/e2_main.json`, and figures to `figures/`.

Optional — the supervised reference models (139 MB, only needed for
`--supervised`):

```bash
bash third_party/download_models.sh   # BMNet + ShadowFormer, ISTD+ checkpoints
```

Everything after the two download scripts runs **fully offline**. The only
network dependency at evaluation time is LPIPS fetching its AlexNet backbone on
first use; `--no-lpips` skips it.

---

## The pipeline (e2 — IAR)

The whole method, per connected shadow instance. No segmentation, no
descriptors, no search, no iterative solve — which is why it is faster than the
baselines rather than slower.

```
            ┌───────────────────────────────────────────────────────────┐
 shadow ───▶│ A. instances            connected components of the mask  │
  image     ├───────────────────────────────────────────────────────────┤
  + mask    │ B. MEASURE the penumbra                                   │
            │    signed distance to the boundary → bin pixels by it →   │
            │    mean luminance per bin → that profile IS the matte α   │
            │    (measured width on ISTD+: 8–9 px, not assumed)         │
            ├───────────────────────────────────────────────────────────┤
            │ C. AFFINE fit, two strips                                 │
            │    one just inside the umbra, one just outside the lit    │
            │    edge; 15 matched quantiles → y = a·x + b per channel;  │
            │    channels coupled (a gain is a property of the light)   │
            ├───────────────────────────────────────────────────────────┤
            │ C'. SELF-CHECK the same-material assumption               │
            │    chroma distance between the strips → shrink toward     │
            │    "change nothing" when they are on different materials  │
            ├───────────────────────────────────────────────────────────┤
            │ D. composite   out = (1−α)·I + α·(a·I + b)                │
            └───────────────────────────────────────────────────────────┘
```

Two design choices carry the result, and both came from the literature survey
in [experiments/e2/research.txt](experiments/e2/research.txt):

**The model is affine, not multiplicative.** Shor & Lischinski (EG'08) and Le &
Samaras (ICCV'19 — the ISTD+ dataset paper itself) both model the shadowed→lit
relation as `I_free = a·I + b`, because ambient light survives inside the
umbra. Every baseline here, and the original guideline, forces `b = 0`. That
single change is worth **+3.16 dB**, and it explains why five quite different
multiplicative estimators all landed within 0.5 dB of each other: they were
fitting the wrong family.

**The penumbra is measured, not assumed.** Gong & Cosker register the penumbra
to a normalised frame and estimate the illumination change across it. Reading
the matte off each image's own radial profile is worth +0.25 dB on the shadow
region and **+0.56 dB on the lit region**.

Implementation: [experiments/e2/iar.py](experiments/e2/iar.py).

---

## Ablation

Full test split, 540 images, 256×256. Every configuration was selected on
**train**; nothing here was tuned on test.

| Arm | PSNR-S | PSNR-NS | PSNR-all | "RMSE"-all | latency |
|---|---:|---:|---:|---:|---:|
| **affine + measured matte (sRGB)** — e2 | **25.02** | **34.82** | **31.09** | **1.350** | 99 ms |
| affine + fixed feather ramp | 24.77 | 34.26 | 30.79 | 1.370 | 98 ms |
| affine + measured matte, linear RGB | 23.72 | 34.17 | 29.87 | 1.544 | 153 ms |
| multiplicative + measured matte | 21.86 | 34.04 | 28.72 | 1.597 | 101 ms |
| multiplicative + fixed feather | 21.61 | 32.92 | 28.31 | 1.644 | 94 ms |

On the 60-image train ablation, **six of the eight arms fail the non-shadow
guard outright** — only affine + measured matte keeps the lit region within
0.3 dB of the untouched input. Both choices are necessary, not merely helpful.

### Self-check

| | without gate | with gate |
|---|---:|---:|
| mean PSNR-S | 25.18 | 25.02 |
| images worse than doing nothing | 22 (4.1 %) | **20 (3.7 %)** |
| worst case vs input | −8.12 dB | **−5.01 dB** |
| max latency | 1776 ms | **430 ms** |

0.16 dB of mean buys 3.1 dB off the worst case.

### Mask sensitivity

| mask dilated/eroded by | −12 px | −6 px | **0** | +6 px | +12 px |
|---|---:|---:|---:|---:|---:|
| PSNR-S | 14.01 | 22.18 | **25.02** | 23.77 | 14.93 |

±6 px is survivable; ±12 px is not. Every number in this repo assumes the
ground-truth ISTD+ mask.

---

## Baselines

Five training-free methods (all tuned on train), two supervised references, and
two ground-truth oracles. See [baseline/RESULTS.md](baseline/RESULTS.md).

```bash
python baseline/run_baseline.py --list          # what's runnable, and is it ready
python baseline/tune.py --n 60                  # select every knob on TRAIN
python baseline/run_baseline.py                 # evaluate on TEST
python baseline/run_baseline.py --supervised bmnet shadowformer
```

| | method | kind |
|---|---|---|
| 1 | `input` | no-op lower bound |
| 2 | `global_gain` | one per-channel gain from whole-region statistics |
| 3 | `gradient_domain` | log-gradient nulling + Poisson reintegration (Finlayson et al. in spirit) |
| 4 | `guo_paired_regions` | SLIC + region pairing + relighting (Guo et al. in spirit) — the critical classical baseline |
| 5 | `boundary_gain` | per-instance gain from boundary rings — best training-free baseline |
| 6 | `bmnet_supervised` | BMNet (CVPR'22), 432 K params, authors' ISTD+ checkpoint |
| 7 | `shadowformer_supervised` | ShadowFormer (AAAI'23), 11.4 M params, authors' ISTD+ checkpoint |
| 8–9 | `oracle_*` | use ground truth; **ceilings, never results** |

**e2 beats both oracles** (25.02 vs 23.22 and 22.67). That is only possible
because those oracles bound the *multiplicative, clamped, feathered* family
that e2 leaves — which is the clearest evidence that the formulation, not the
estimator, was the limit.

---

## Repository layout

```
guideline.md               the research plan, rewritten around what is measured
README.md                  this file
requirements.yaml          conda environment (torch pinned CPU-only, on purpose)

data/download_data.sh      builds ISTD+ from its two upstream sources
third_party/download_models.sh   BMNet + ShadowFormer checkpoints

common/                    dataset access, metrics, evaluation driver
  metrics.py               PSNR/SSIM/MAE-Lab/RMSE/LPIPS, S / NS / all
  runner.py                one code path for every method, so latency compares

baseline/                  training-free baselines, supervised refs, oracles
  RESULTS.md               baseline numbers + the oracle decomposition
experiments/
  e1/                      amplitude transfer (the original guideline's method)
    CORRECTIONS.md         13 errors found in the original guideline
  e2/                      ← the main method
    research.txt           literature survey that produced the design
    iar.py                 the pipeline
    RESULTS.md             full results, ablation, failure analysis
tools/                     dataset assembly, report tables, figures
figures/                   side-by-side comparisons with per-image PSNR
```

---

## Protocol notes

Three conventions decide whether numbers are comparable, and the literature is
inconsistent about all three. Ours are stated so they can be checked:

* **"RMSE" is a mean absolute error in CIE L\*a\*b\*** (L ∈ [0,100]) — that is
  what the field prints under that name; the authors of the original evaluation
  code say so. True RMSE is reported in a separate column and is roughly 2×
  larger.
* **Headline metrics are at 256×256**, the literature protocol. Native 480×640
  is reported alongside and is ~0.3 dB harder.
* **Regions**: S = shadow, NS = non-shadow, all = whole image; per-image metric
  then unweighted mean.

Published ISTD+ tables are **not** mutually comparable — ShadowFormer scores
26.55 whole-image PSNR in one widely-used re-benchmark and ~32.9 in its own
paper. For that reason every method in this repo, including the two supervised
networks, is **measured inside this harness**, on the same CPU, rather than
quoted. That is what "fair" has to mean here.

Both supervised baselines run on **CPU** so their latency shares a column with
everything else. That is honest for comparison and badly pessimistic as a
deployment number for either network.
