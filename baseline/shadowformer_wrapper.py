"""ShadowFormer (Guo et al., AAAI 2023) as the strong supervised reference.

ShadowFormer is the mask-based supervised method the guideline names first
(sec. 7.2) and is the closest thing to SOTA with an ISTD+ checkpoint that is
actually published (`ShadowFormer_ISTDplus.pth`, 137 MB, from the authors'
release).  It is the right upper reference for the whole study: if e1 gets
anywhere near it while training free, that is the headline; if it does not, the
gap is the honest cost of refusing supervision.

Setup handled by `tools/fetch_supervised_baselines.py`:
  third_party/ShadowFormer/                      (repo, --depth 1)
  third_party/ShadowFormer/pretrained/ShadowFormer_ISTDplus.pth

Things worked around, all upstream rather than in our harness:
  1. `test.py` hard-codes CUDA and wraps the model in `DataParallel`; we build
     the architecture directly and strip the `module.` prefix ourselves.
  2. Inputs must be padded up to a multiple of `8 * win_size` (= 80) and the
     output cropped back -- the upstream test loop does this and getting it
     wrong silently costs ~1 dB, so it is reproduced exactly.
  3. torch 2.x needs `weights_only=False` for these checkpoints.

Runs on CPU so its latency sits in the same column as every other method.
That is the right comparison for this study but NOT a deployment number for a
transformer -- see the note in RESULTS.md.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.path.join(ROOT, "third_party", "ShadowFormer")
WEIGHTS = os.path.join(SF, "pretrained", "ShadowFormer_ISTDplus.pth")

# upstream defaults for the released ISTD/ISTD+ checkpoints (options.py / test.py)
ARGS = SimpleNamespace(
    arch="ShadowFormer", embed_dim=32, win_size=10,
    token_projection="linear", token_mlp="leff", train_ps=320,
)

_NET = None
_META = {}


def _load():
    global _NET
    if _NET is not None:
        return _NET
    import torch
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(
            f"{WEIGHTS} missing -- run tools/fetch_supervised_baselines.py")
    if SF not in sys.path:
        sys.path.insert(0, SF)
    from model import ShadowFormer

    net = ShadowFormer(img_size=ARGS.train_ps, embed_dim=ARGS.embed_dim,
                       win_size=ARGS.win_size,
                       token_projection=ARGS.token_projection,
                       token_mlp=ARGS.token_mlp)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(os.cpu_count() or 4)
    _META.update({
        "checkpoint": os.path.relpath(WEIGHTS, ROOT),
        "checkpoint_bytes": os.path.getsize(WEIGHTS),
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "missing_keys": len(missing), "unexpected_keys": len(unexpected),
        "missing_keys_sample": list(missing)[:8],
        "unexpected_keys_sample": list(unexpected)[:8],
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "arch_args": vars(ARGS), "device": "cpu",
    })
    _NET = net
    return net


def meta() -> dict:
    _load()
    return dict(_META)


def make_shadowformer():
    """fn(img_rgb_u8, mask_u8) -> (pred_rgb_u8, aux)."""
    import torch
    import torch.nn.functional as F
    net = _load()
    mult = 8 * ARGS.win_size

    def fn(img_u8, mask_u8):
        h, w = img_u8.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(
            img_u8.astype(np.float32).transpose(2, 0, 1) / 255.0))[None]
        m = torch.from_numpy(np.ascontiguousarray(
            mask_u8.astype(np.float32) / 255.0))[None, None]
        H = ((h + mult) // mult) * mult
        W = ((w + mult) // mult) * mult
        padh, padw = (H - h if h % mult else 0), (W - w if w % mult else 0)
        if padh or padw:
            x = F.pad(x, (0, padw, 0, padh), "reflect")
            m = F.pad(m, (0, padw, 0, padh), "reflect")
        with torch.no_grad():
            y = net(x, m)
        y = y.clamp(0, 1)[0].numpy().transpose(1, 2, 0)[:h, :w, :] * 255.0
        return np.clip(y + 0.5, 0, 255).astype(np.uint8), None

    return fn
