"""Shared evaluation driver: runs a method over a split, times it, scores it,
and writes a JSON result file.  Baseline and experiments use this identical
code path so their numbers and their latencies are comparable.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import metrics as M
from common.istd import load_split, Sample


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["device_used_for_lpips"] = "cpu"
    except Exception:
        pass
    return info


def latency_stats(ts: list[float]) -> dict:
    a = np.array(ts, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "total_s": float(a.sum() / 1000.0),
        "fps": float(1000.0 / a.mean()),
    }


def run_method(name: str, fn, split: str = "test", out_json: str | None = None,
               save_dir: str | None = None, limit: int | None = None,
               warmup: int = 2, extra_meta: dict | None = None,
               with_lpips: bool = True, eval_full: bool = True,
               progress_every: int = 50, needs_gt: bool = False) -> dict:
    """fn(img_rgb_u8, mask_u8) -> (pred_rgb_u8, aux_dict|None)

    `needs_gt=True` switches the signature to fn(img, mask, gt) -- for oracles
    only; their latency is meaningless as a deployment number and is reported
    purely for completeness.

    Latency timed around fn only, at native resolution (480x640), excluding
    disk I/O and metric computation.  `warmup` images are run once before
    timing starts to remove first-call / allocator effects.
    """
    samples = load_split(split)
    if limit:
        samples = samples[:limit]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for s in samples[:warmup]:
        img, mask, gt = s.load()
        fn(img, mask, gt) if needs_gt else fn(img, mask)

    per_image, per_image_full, lat, aux_all = [], [], [], {}
    t_wall0 = time.perf_counter()
    for i, s in enumerate(samples):
        img, mask, gt = s.load()
        t0 = time.perf_counter()
        pred, aux = fn(img, mask, gt) if needs_gt else fn(img, mask)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)

        pred = np.clip(pred, 0, 255).astype(np.uint8)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, s.name + ".png"), pred[:, :, ::-1])

        m256 = M.compute_metrics(pred, gt, mask, at_size=M.EVAL_SIZE, with_lpips=with_lpips)
        m256["name"] = s.name
        m256["latency_ms"] = lat[-1]
        per_image.append(m256)
        if eval_full:
            mf = M.compute_metrics(pred, gt, mask, at_size=None, with_lpips=False)
            mf["name"] = s.name
            per_image_full.append(mf)
        if aux:
            aux_all[s.name] = aux
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{name}] {i+1}/{len(samples)}  "
                  f"psnr_S={np.mean([d['psnr_S'] for d in per_image]):.3f} "
                  f"{lat[-1]:.1f}ms", flush=True)
    wall = time.perf_counter() - t_wall0

    result = {
        "method": name,
        "dataset": "ISTD+ (Adjusted ISTD)",
        "split": split,
        "n_images": len(samples),
        "native_resolution": "480x640",
        "eval_protocol": {
            "headline_resize": "256x256 bilinear (cv2.INTER_LINEAR), mask nearest",
            "rmse_note": "mae_lab_* is the number the literature prints as 'RMSE' (MAE in CIE Lab, L in [0,100]); rmse_lab_*/rmse_rgb_* are true RMSE",
            "psnr": "region-restricted MSE, MAX=255, per-image then averaged",
            "ssim": "per-channel SSIM map (gaussian_weights, sigma=1.5), averaged inside region",
            "lpips": "LPIPS-AlexNet on the whole 256x256 image",
        },
        "metrics_256": M.aggregate(per_image),
        "metrics_full_res": M.aggregate(per_image_full) if eval_full else None,
        "latency": latency_stats(lat),
        "wall_clock_including_io_and_metrics_s": wall,
        "machine": machine_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_image": per_image,
        "per_image_full_res": per_image_full if eval_full else None,
        "aux": aux_all or None,
    }
    if extra_meta:
        result["meta"] = extra_meta
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {out_json}")
    return result
