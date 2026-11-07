"""Local web UI for the shadow-removal PoC.

    python app/ui.py                 # opens http://127.0.0.1:7860
    python app/ui.py --share         # temporary public link
    python app/ui.py --port 8080

Gradio serves a local page; nothing leaves the machine and no separate backend
has to be started. Upload a photo, get the shadow removed.

The three panels are deliberate. A demo that shows only the output hides which
half failed when the result is bad: the detector finding the wrong region and
the remover mis-estimating the light look identical in a single "after" image.
So the mask is always shown, and "Use my own mask" lets you bypass the
detector entirely to see what the remover does when the region is right.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from app import detector as D
from app.pipeline import overlay, remove_shadow

CSS = """
.small-note { font-size: 0.85em; opacity: 0.75; }
footer { display: none !important; }
"""

INTRO = """
# Shadow Removal — PoC

**Detect** the shadow, then **remove** it with *e2 (Instance Affine Relighting)* —
a training-free method with **zero learned weights** that runs in ~90 ms on CPU.

Upload a photo, or drop in one of the ISTD+ test images from
`data/ISTD_plus/test/A/`.
"""

CAVEAT = """
### Worth knowing

* **The remover has no learned weights.** Its parameters are a handful of
  scalars chosen on the ISTD+ *training* split. The shadow **detector** is a
  learned model (SILT, ICCV 2023) — that part is not training-free.
* **e2 was tuned on ISTD+**: mostly single cast shadows on near-uniform
  ground, one camera. On ISTD+ test it scores 25.0 dB PSNR over the shadow
  region against 22.0 for the best training-free baseline. On arbitrary
  photos expect less, and expect material-rich scenes to be harder.
* **The self-check can decline.** When the lit and shadowed sides of the
  boundary look like different materials, e2 shrinks its own correction
  toward "change nothing" rather than guess. A low *trust* in the details
  panel means it backed off on purpose.
"""


def build(demo_image: str | None = None):
    import gradio as gr

    avail = D.available_backends()
    choices = []
    if avail["silt_istd"]:
        choices.append("SILT (ISTD) — matches the benchmark")
    if avail["silt_sbu"]:
        choices.append("SILT (SBU) — better on everyday photos")
    choices.append("Classical (no weights, weaker)")
    default = choices[0]

    def run(image, det_choice, use_own_mask, own_mask, threshold, sensitivity,
            grow, trust_check):
        if image is None:
            return None, None, None, "Upload an image first."
        img = np.asarray(image)[:, :, :3]

        backend = "classical" if det_choice.startswith("Classical") else "silt"
        ckpt = "sbu" if "SBU" in det_choice else "istd"
        mask_override = np.asarray(own_mask)[:, :, :3].mean(2).astype(np.uint8) \
            if (use_own_mask and own_mask is not None) else None

        try:
            out, mask, info = remove_shadow(
                img, backend=backend, checkpoint=ckpt, threshold=float(threshold),
                sensitivity=float(sensitivity), mask_override=mask_override,
                dilate=int(grow), trust_check=bool(trust_check))
        except Exception as e:                      # keep the demo alive
            return None, None, None, f"**Failed:** {type(e).__name__}: {e}"

        det = info["detector"]
        e2 = info.get("e2", {})
        trust = e2.get("trust")
        lines = [
            f"**{info['total_ms']:.0f} ms total** "
            f"— detect {info['detect_ms']:.0f} ms, remove {info['remove_ms']:.0f} ms",
            "",
            f"- detector: `{det.get('backend')}`"
            + (f" ({det.get('checkpoint')}, {det.get('n_parameters', 0):,} params)"
               if det.get("backend") == "silt" else
               f" ({det.get('n_parameters', 0)} params)"),
            f"- shadow covers **{det.get('shadow_area_frac', 0) * 100:.1f}%** of the image",
        ]
        if "note" in info:
            lines.append(f"- ⚠️ {info['note']}")
        if e2:
            lines += [
                f"- shadow instances: {e2.get('n_instances', '?')}",
                f"- measured penumbra width: {e2.get('penumbra_px', float('nan')):.1f} px",
            ]
            if trust is not None and np.isfinite(trust):
                if trust >= 0.99:
                    lines.append("- self-check: **passed** — estimate used in full")
                elif trust <= 0.01:
                    lines.append("- self-check: **declined** — the two sides of the "
                                 "boundary look like different materials, so the "
                                 "shadow was left alone")
                else:
                    lines.append(f"- self-check: **partial ({trust:.2f})** — "
                                 "correction shrunk toward no-change")
            a, b = e2.get("a_rgb"), e2.get("b_rgb")
            if a and b:
                lines.append("- fitted map `I_free = a·I + b`: "
                             f"a = [{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}], "
                             f"b = [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}]")
        return out, overlay(img, mask), mask, "\n".join(lines)

    # Gradio 6 moved css/theme from the Blocks constructor to launch()
    with gr.Blocks(title="Shadow Removal PoC") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Input photo", type="numpy", height=320)
                det_choice = gr.Radio(choices, value=default, label="Shadow detector")
                with gr.Accordion("Options", open=False):
                    threshold = gr.Slider(0.1, 0.95, value=0.7, step=0.05,
                                          label="Detector threshold (SILT)")
                    sensitivity = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                            label="Sensitivity (classical only)")
                    grow = gr.Slider(-12, 12, value=0, step=1,
                                     label="Grow / shrink mask (px)")
                    trust_check = gr.Checkbox(
                        value=True, label="Enable the self-check",
                        info="Off = always apply the correction, even when the "
                             "method's own material test says not to")
                    use_own_mask = gr.Checkbox(
                        value=False, label="Use my own mask instead of the detector")
                    own_mask = gr.Image(label="Mask (white = shadow)",
                                        type="numpy", height=160)
                run_btn = gr.Button("Remove shadow", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    out_img = gr.Image(label="Shadow removed", height=320)
                    ov_img = gr.Image(label="Detected shadow", height=320)
                mask_img = gr.Image(label="Mask", height=200)
                details = gr.Markdown()
        gr.Markdown(CAVEAT, elem_classes="small-note")

        if demo_image and os.path.exists(demo_image):
            gr.Examples(examples=[[demo_image]], inputs=[image])

        run_btn.click(run,
                      [image, det_choice, use_own_mask, own_mask, threshold,
                       sensitivity, grow, trust_check],
                      [out_img, ov_img, mask_img, details])
        image.upload(run,
                     [image, det_choice, use_own_mask, own_mask, threshold,
                      sensitivity, grow, trust_check],
                     [out_img, ov_img, mask_img, details])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    sample = os.path.join(ROOT, "data", "ISTD_plus", "test", "A", "90-1.png")
    demo = build(sample if os.path.exists(sample) else None)
    print(f"\nShadow Removal PoC -> http://127.0.0.1:{args.port}\n")
    import gradio as gr
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share, inbrowser=not args.no_browser,
                css=CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
