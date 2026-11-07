"""Re-save a downloaded SILT checkpoint in a safe-to-load format. Run once.

WHY THIS EXISTS

SILT's released checkpoints are Python pickles. `torch.load` with its default
`weights_only=False` does not just read data: it executes whatever the pickle
instructs, so loading a 198 MB file fetched from a third-party Google Drive
link runs that file's code on your machine. `app/detector.py` therefore loads
with `weights_only=True`, which refuses to execute anything.

If the original checkpoint will not open in safe mode -- because it stores its
config as a pickled object rather than a plain dict -- this script opens it
ONCE with the unsafe reader, keeps only the tensors and plain values, and
writes a `*.safe.pth` that the app can load safely from then on.

You are choosing to trust the file at that moment, which is why the flag is
spelled out. Check the download first:

    sha256sum third_party/SILT/pretrained/silt_pvtb3_istd.pth

and satisfy yourself it came from
https://github.com/hanyangclarence/SILT before running this.

    python app/convert_checkpoint.py --i-trust-this-file
    python app/convert_checkpoint.py --i-trust-this-file --which sbu
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="istd", choices=["istd", "sbu"])
    ap.add_argument("--i-trust-this-file", action="store_true",
                    help="required: acknowledges that opening the original "
                         "checkpoint executes code from it")
    args = ap.parse_args()

    from app.detector import CKPTS
    src = CKPTS[args.which]
    if not os.path.exists(src):
        print(f"missing: {src}\nrun app/download_detector.sh first")
        return 1
    dst = src.replace(".pth", ".safe.pth")

    print(f"source : {src}")
    print(f"size   : {os.path.getsize(src) / 1e6:.1f} MB")
    print(f"sha256 : {sha256(src)}")

    if not args.i_trust_this_file:
        print("\nRefusing to open it. Opening this file executes code from it.")
        print("Verify the hash against the publisher, then re-run with "
              "--i-trust-this-file")
        return 2

    import torch
    print("\nopening with weights_only=False (executes pickle code) ...")
    ck = torch.load(src, map_location="cpu", weights_only=False)

    def plain(x):
        """Strip everything that is not a tensor or a plain container."""
        import torch as T
        if isinstance(x, T.Tensor):
            return x
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        try:                                    # OmegaConf and similar
            from omegaconf import OmegaConf
            if OmegaConf.is_config(x):
                return plain(OmegaConf.to_container(x, resolve=True))
        except Exception:
            pass
        return str(x)

    safe = {"state_dict": {str(k): v for k, v in ck["state_dict"].items()},
            "configs": plain(ck["configs"])}
    torch.save(safe, dst)
    print(f"wrote  : {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB)")

    print("verifying it reloads in safe mode ...")
    back = torch.load(dst, map_location="cpu", weights_only=True)
    assert set(back["state_dict"]) == set(safe["state_dict"])
    print("ok -- app/detector.py will now prefer this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
