"""ISTD+ (Adjusted ISTD) dataset access.

Layout assembled by tools/prepare_istd_plus.py:

    data/ISTD_plus/{train,test}/
        A/  shadow image        (from original ISTD, unchanged)
        B/  shadow mask         (from original ISTD, unchanged)
        C/  shadow-free image   (ADJUSTED / color-corrected -> this is what makes it ISTD+)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "ISTD_plus")


@dataclass
class Sample:
    name: str
    path_shadow: str
    path_mask: str
    path_free: str

    def load(self):
        """Returns (shadow RGB uint8, mask uint8 {0,255}, free RGB uint8)."""
        img = cv2.imread(self.path_shadow, cv2.IMREAD_COLOR)[:, :, ::-1]
        mask = cv2.imread(self.path_mask, cv2.IMREAD_GRAYSCALE)
        free = cv2.imread(self.path_free, cv2.IMREAD_COLOR)[:, :, ::-1]
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        return np.ascontiguousarray(img), mask, np.ascontiguousarray(free)


def load_split(split: str, root: str = DATA_ROOT) -> list[Sample]:
    d = os.path.join(root, split)
    names = sorted(
        os.listdir(os.path.join(d, "A")),
        key=lambda n: tuple(int(x) for x in os.path.splitext(n)[0].split("-")),
    )
    out = []
    for n in names:
        out.append(
            Sample(
                name=os.path.splitext(n)[0],
                path_shadow=os.path.join(d, "A", n),
                path_mask=os.path.join(d, "B", n),
                path_free=os.path.join(d, "C", n),
            )
        )
    return out
