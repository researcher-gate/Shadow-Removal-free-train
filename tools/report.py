"""Turn result JSONs into the markdown tables used by the RESULTS.md files.

    from tools.report import load_results, table_quality, table_error, table_latency
"""
from __future__ import annotations

import json
import os


def load_results(paths: dict[str, str]) -> dict:
    """{label: json path} -> {label: parsed result}, skipping missing files."""
    out = {}
    for label, p in paths.items():
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[label] = json.load(f)
    return out


def _fmt(v, nd=2):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if f != f else f"{f:.{nd}f}"


def _md(header, rows, align=None):
    align = align or (["---"] + ["---:"] * (len(header) - 1))
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def table_quality(res: dict, key="metrics_256") -> str:
    head = ["Method", "PSNR S ↑", "PSNR NS ↑", "PSNR all ↑",
            "SSIM S ↑", "SSIM NS ↑", "SSIM all ↑"]
    rows = []
    for label, r in res.items():
        m = r.get(key) or {}
        rows.append([label, _fmt(m.get("psnr_S")), _fmt(m.get("psnr_NS")),
                     _fmt(m.get("psnr_ALL")),
                     _fmt(m.get("ssim_S"), 4), _fmt(m.get("ssim_NS"), 4),
                     _fmt(m.get("ssim_ALL"), 4)])
    return _md(head, rows)


def table_error(res: dict, key="metrics_256") -> str:
    head = ["Method", '"RMSE" S ↓', '"RMSE" NS ↓', '"RMSE" all ↓',
            "true RMSE-Lab all ↓", "RMSE-RGB all ↓", "LPIPS ↓"]
    rows = []
    for label, r in res.items():
        m = r.get(key) or {}
        rows.append([label,
                     _fmt(m.get("mae_lab_S")), _fmt(m.get("mae_lab_NS")),
                     _fmt(m.get("mae_lab_ALL")),
                     _fmt(m.get("rmse_lab_ALL")), _fmt(m.get("rmse_rgb_ALL")),
                     _fmt(m.get("lpips_ALL"), 4)])
    return _md(head, rows)


def table_latency(res: dict) -> str:
    head = ["Method", "mean ms", "median ms", "std ms", "min ms", "max ms",
            "p90 ms", "p95 ms", "p99 ms", "fps", "total s"]
    rows = []
    for label, r in res.items():
        L = r.get("latency") or {}
        rows.append([label, _fmt(L.get("mean_ms"), 1), _fmt(L.get("median_ms"), 1),
                     _fmt(L.get("std_ms"), 1), _fmt(L.get("min_ms"), 1),
                     _fmt(L.get("max_ms"), 1), _fmt(L.get("p90_ms"), 1),
                     _fmt(L.get("p95_ms"), 1), _fmt(L.get("p99_ms"), 1),
                     _fmt(L.get("fps"), 2), _fmt(L.get("total_s"), 1)])
    return _md(head, rows)


def machine_block(res: dict) -> str:
    for r in res.values():
        m = r.get("machine") or {}
        return "\n".join(f"- **{k}**: {v}" for k, v in m.items())
    return ""
