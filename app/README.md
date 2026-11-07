# Shadow Removal — PoC app

Upload a photo → shadow detected → shadow removed. Runs entirely on your
machine, on CPU.

```bash
# 1. detector weights (198 MB, one time)
bash app/download_detector.sh          # or: sbu  |  all

# 2. UI
python app/ui.py                       # opens http://127.0.0.1:7860
```

If you skip step 1 the app still runs — it falls back to a weightless
classical detector and says so in the UI.

---

## What it is made of

```
  photo ──▶ DETECTOR ──▶ mask ──▶ e2 / IAR ──▶ shadow-free photo
            (learned)             (no weights)
```

| Stage | What | Weights |
|---|---|---|
| Detect | SILT (Hu et al., ICCV 2023), PVT-v2 | 49–86 M **learned** |
| Detect (fallback) | LAB luminance + blue-shift thresholding | 0 |
| Remove | **e2 — Instance Affine Relighting** | **0** |

**Only the remover is training-free.** The detector is a learned model. The
project's claim is about the removal step; the detector is a component, and
the UI labels which one ran.

### Two detector checkpoints

| | trained on | BER | use when |
|---|---|---|---|
| `istd` | ISTD | 1.11 | images like the benchmark — the detector and e2 then agree on what a shadow is |
| `sbu` | SBU | 4.36 | everyday photos — far more varied training scenes, generalises better |

### The remover

e2 fits `I_free = a·I + b` per shadow instance, per channel, from two thin
strips either side of the boundary, and composites through a penumbra matte
measured from the image's own radial intensity profile. ~90 ms per image.

On ISTD+ test (540 images, ground-truth masks) it scores **25.0 dB** PSNR over
the shadow region against **22.0** for the best training-free baseline, and it
beats both ground-truth oracles of the multiplicative formulation it replaces.
Full numbers: [`../RESULT.md`](../RESULT.md).

---

## Files

| | |
|---|---|
| `ui.py` | Gradio UI — the local web page |
| `pipeline.py` | `remove_shadow()` — detect → remove, with the scale handling |
| `detector.py` | SILT wrapper + the classical fallback |
| `selftest.py` | end-to-end check on ISTD+ images |
| `download_detector.sh` | fetches SILT and its checkpoint |
| `convert_checkpoint.py` | one-time safe re-save of a checkpoint (see Security) |

```bash
python app/selftest.py --n 6                     # with SILT
python app/selftest.py --n 6 --backend classical # no weights needed
```

`selftest.py` deliberately reports the result **twice** — once with the
ISTD+ ground-truth mask, once with the detector's own mask — so the detector's
error and the remover's error stay separable. Measured with the classical
fallback on 5 images: 14.5 dB input → 26.1 dB with the true mask → 23.9 dB
with a detected mask at IoU 0.50. **The mask costs about 2 dB**; that gap is
what a better detector buys.

---

## Security note

SILT's checkpoints are Python pickles downloaded from a third-party Drive
link. `torch.load`'s default (`weights_only=False`) **executes code from the
file**, so `detector.py` loads with `weights_only=True`, which does not.

If a checkpoint refuses to open safely, `convert_checkpoint.py` opens it once —
behind an explicit `--i-trust-this-file` flag, after printing the file's
SHA-256 so you can check it against the publisher — and re-saves only the
tensors in a format that always loads safely.

Do not relax the loader instead. The flag is spelled out because that is the
moment you are choosing to trust the file.

---

## Honest limits

- **e2 was tuned on ISTD+**: mostly single cast shadows on near-uniform
  ground, one camera. Expect less on arbitrary photos.
- **It fits in sRGB, not linear radiance.** That won by 1.30 dB on ISTD+ but
  is backwards from the physics, most likely because ISTD+'s ground truth was
  itself built by an affine colour correction in the encoded domain. This is
  the part most likely not to transfer — see [`../experiments/e2/RESULTS.md`](../experiments/e2/RESULTS.md) §6.
- **Shadows across material boundaries break it** — a shadow falling half on
  grass and half on pavement, or across a mural. The self-check detects this
  (the two strips disagree chromatically) and shrinks the correction toward
  no-change rather than guessing; the UI reports it as *self-check: declined*.
  On ISTD+ test, 3.7 % of images still end up worse than the untouched input.
- **Large images are processed at ≤1600 px** on the long side; the correction
  is then upsampled and applied to the full-resolution pixels, so detail is
  preserved but the mask is estimated at the reduced size.
- **CPU only.** `requirements.yaml` pins a CPU build of torch on purpose.

---

## Environment note

On this machine `python` on PATH may resolve to a Python 2.7 install. Use the
project interpreter explicitly if `python app/ui.py` fails with a syntax
error:

```bash
PYTHON=/c/Users/<you>/AppData/Local/Programs/Python/Python312/python.exe
"$PYTHON" app/ui.py
```
