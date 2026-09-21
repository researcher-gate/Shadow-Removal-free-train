"""One-shot setup for the supervised reference baselines.

    python tools/fetch_supervised_baselines.py

Fetches into third_party/:

  BMNet          KevinJ-Huang/BMNet             weights ship in the repo
                 MainNet/pretrain/AISTD.pth     (2 MB; AISTD == ISTD+)
  ShadowFormer   GuoLanqing/ShadowFormer        weights from the authors'
                 pretrained/ShadowFormer_ISTDplus.pth   release (137 MB)

Both are run on CPU by the baseline harness so their latency is measured on the
same hardware as the training-free methods.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(ROOT, "third_party")

REPOS = [
    ("BMNet", "https://github.com/KevinJ-Huang/BMNet.git"),
    ("ShadowFormer", "https://github.com/GuoLanqing/ShadowFormer.git"),
]
SHADOWFORMER_ISTDPLUS_GDRIVE_ID = "10pBsJenoWGriZ9kjWOcE4l4Kzg-F1TFd"
PIP_EXTRAS = ["einops", "timm", "natsort"]


def sh(cmd, **kw):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def main():
    os.makedirs(TP, exist_ok=True)
    for name, url in REPOS:
        d = os.path.join(TP, name)
        if os.path.isdir(d):
            print(f"{name}: already cloned")
        else:
            sh(["git", "clone", "--depth", "1", url, d])

    w = os.path.join(TP, "ShadowFormer", "pretrained", "ShadowFormer_ISTDplus.pth")
    if os.path.exists(w):
        print(f"ShadowFormer weights: present ({os.path.getsize(w)/1e6:.0f} MB)")
    else:
        os.makedirs(os.path.dirname(w), exist_ok=True)
        sh([sys.executable, "-m", "gdown", SHADOWFORMER_ISTDPLUS_GDRIVE_ID, "-O", w])

    b = os.path.join(TP, "BMNet", "MainNet", "pretrain", "AISTD.pth")
    print(f"BMNet weights: {'present' if os.path.exists(b) else 'MISSING'} "
          f"({os.path.getsize(b)/1e6:.1f} MB)" if os.path.exists(b) else "BMNet weights MISSING")

    sh([sys.executable, "-m", "pip", "install", "--quiet", *PIP_EXTRAS])

    sys.path.insert(0, ROOT)
    from baseline.bmnet_wrapper import meta as bmeta
    from baseline.shadowformer_wrapper import meta as smeta
    for name, m in (("BMNet", bmeta()), ("ShadowFormer", smeta())):
        print(f"{name}: {m['n_parameters']:,} params, "
              f"missing={m['missing_keys']} unexpected={m['unexpected_keys']}")
    print("ready")


if __name__ == "__main__":
    main()
