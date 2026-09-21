"""Assemble ISTD+ (Adjusted ISTD) from its two sources.

ISTD+ is not distributed as one archive.  It is the original ISTD shadow images
and masks, plus Le & Samaras' *colour-adjusted* shadow-free images.  Using the
original ISTD shadow-free images instead would silently change the task: in
original ISTD the shadow-free frame has a global colour/exposure shift relative
to the shadow frame, so a method is scored partly on guessing that shift.  That
is exactly why the guideline calls for ISTD+.

Sources
  data/raw_ISTD/{train,test}/{train,test}_{A,B,C}   original ISTD  (HF mirror)
  data/adjusted_ISTD/train_C_fixed_ours             adjusted shadow-free (train)
  data/adjusted_ISTD/test_C_fixed_official          adjusted shadow-free (test)

Output
  data/ISTD_plus/{train,test}/{A,B,C}
  data/ISTD_plus/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_ISTD")
ADJ = os.path.join(ROOT, "data", "adjusted_ISTD")
OUT = os.path.join(ROOT, "data", "ISTD_plus")

SPLITS = {
    "train": {"raw": os.path.join(RAW, "train"), "pre": "train",
              "adj": os.path.join(ADJ, "train_C_fixed_ours")},
    "test":  {"raw": os.path.join(RAW, "test"), "pre": "test",
              "adj": os.path.join(ADJ, "test_C_fixed_official")},
}


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def md5_of_dir(d, limit=None):
    h = hashlib.md5()
    for n in sorted(os.listdir(d))[:limit]:
        h.update(n.encode())
        with open(os.path.join(d, n), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def main():
    wanted = sys.argv[1:] or list(SPLITS)
    manifest = {"dataset": "ISTD+ (Adjusted ISTD)",
                "shadow_and_mask_source": "original ISTD (Wang et al., CVPR18)",
                "shadow_free_source": "Adjusted ISTD (Le & Samaras, ICCV19 / PAMI21)",
                "splits": {}}
    for split, cfg in SPLITS.items():
        if split not in wanted:
            continue
        a_dir = os.path.join(cfg["raw"], cfg["pre"] + "_A")
        b_dir = os.path.join(cfg["raw"], cfg["pre"] + "_B")
        c_dir = cfg["adj"]
        for d in (a_dir, b_dir, c_dir):
            if not os.path.isdir(d):
                print(f"MISSING {d}", file=sys.stderr)
                return 1
        names = sorted(os.listdir(a_dir))
        missing = [n for n in names
                   if not os.path.exists(os.path.join(b_dir, n))
                   or not os.path.exists(os.path.join(c_dir, n))]
        if missing:
            print(f"{split}: {len(missing)} triplets incomplete, e.g. {missing[:5]}",
                  file=sys.stderr)
            return 1
        for sub, src in (("A", a_dir), ("B", b_dir), ("C", c_dir)):
            dd = os.path.join(OUT, split, sub)
            os.makedirs(dd, exist_ok=True)
            for n in names:
                link_or_copy(os.path.join(src, n), os.path.join(dd, n))
        manifest["splits"][split] = {
            "n_triplets": len(names),
            "A": "shadow image", "B": "shadow mask", "C": "shadow-free (adjusted)",
            "md5_first20_A": md5_of_dir(os.path.join(OUT, split, "A"), 20),
            "md5_first20_C": md5_of_dir(os.path.join(OUT, split, "C"), 20),
        }
        print(f"{split}: {len(names)} triplets")

    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath, encoding="utf-8"))
        old["splits"].update(manifest["splits"])
        manifest = old
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("manifest ->", mpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
