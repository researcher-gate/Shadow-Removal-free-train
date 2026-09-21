"""BMNet (Zhu et al., CVPR 2022, "Bijective Mapping Network for Shadow Removal")
as a supervised reference point in the baseline suite.

Why this one: the guideline names BMNet among the mask-based supervised
baselines (sec. 7.2), and it is the only strong published method whose
**ISTD+ weights ship inside its own repository** (`MainNet/pretrain/AISTD.pth`,
2 MB) -- AISTD is ISTD+ -- so the comparison needs no third-party file hosting
and cannot silently pick up the wrong checkpoint.

Three things had to be worked around; all of them are in the upstream repo, not
in our harness:

  1. `InvISPNet.__init__` hard-codes an absolute path to `condition.pth`
     (`/home/jieh/Projects/...`).  Patched to the repo-relative path.
  2. `eval.py` is wired to a distributed-training launcher with absolute paths
     to the author's own filesystem.  We bypass it and drive the network
     directly, using the preprocessing from `data/SIEN_dataset.py`:
     RGB in [0,1], native resolution, mask as a single channel in [0,1].
  3. The repo targets torch 1.7; we run 2.x.  Only the checkpoint load needs
     `weights_only=False`.

Runs on CPU so its latency is measured on the same hardware as every other
method in the table.  That is the honest comparison for this study but it is
NOT how the method would be deployed -- see the note in RESULTS.md.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BMNET = os.path.join(ROOT, "third_party", "BMNet", "MainNet")

import numpy as np


def _patch_condition_path():
    """Rewrite the upstream absolute path to condition.pth, once."""
    p = os.path.join(BMNET, "models", "archs", "EnhanceN_arch.py")
    src = open(p, encoding="utf-8").read()
    bad = "torch.load('/home/jieh/Projects/Shadow/MainNet/pretrain/condition.pth')"
    good = ("torch.load(__import__('os').path.join(__import__('os').path.dirname("
            "__import__('os').path.dirname(__import__('os').path.dirname("
            "__import__('os').path.abspath(__file__)))), 'pretrain', "
            "'condition.pth'), map_location='cpu', weights_only=False)")
    if bad in src:
        open(p, "w", encoding="utf-8").write(src.replace(bad, good))
        return "patched"
    return "already patched" if good in src else "pattern not found"


_NET = None
_META = {}


def _load():
    global _NET
    if _NET is not None:
        return _NET
    import torch
    status = _patch_condition_path()
    if BMNET not in sys.path:
        sys.path.insert(0, BMNET)
    from models.archs.EnhanceN_arch import InvISPNet

    net = InvISPNet()
    ckpt = os.path.join(BMNET, "pretrain", "AISTD.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(os.cpu_count() or 4)
    _META.update({
        "checkpoint": os.path.relpath(ckpt, ROOT),
        "checkpoint_bytes": os.path.getsize(ckpt),
        "condition_path_patch": status,
        "missing_keys": len(missing), "unexpected_keys": len(unexpected),
        "missing_keys_sample": list(missing)[:8],
        "unexpected_keys_sample": list(unexpected)[:8],
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "device": "cpu",
    })
    _NET = net
    return net


def meta() -> dict:
    _load()
    return dict(_META)


def make_bmnet():
    """fn(img_rgb_u8, mask_u8) -> (pred_rgb_u8, aux)."""
    import torch
    net = _load()

    def fn(img_u8, mask_u8):
        x = torch.from_numpy(
            np.ascontiguousarray(img_u8.astype(np.float32).transpose(2, 0, 1) / 255.0))[None]
        m = torch.from_numpy(
            np.ascontiguousarray((mask_u8.astype(np.float32) / 255.0)))[None, None]
        with torch.no_grad():
            out, _ = net(x, m, x, rev=False)
        y = out[0].clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0
        return np.clip(y + 0.5, 0, 255).astype(np.uint8), None

    return fn
