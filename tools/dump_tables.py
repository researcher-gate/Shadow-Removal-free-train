"""Print the markdown tables for baseline/RESULTS.md and experiments/e1/RESULTS.md.

    python tools/dump_tables.py baseline
    python tools/dump_tables.py e1
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.report import (load_results, machine_block, table_error,
                          table_latency, table_quality)

BASE = os.path.join(ROOT, "baseline", "results")
E1 = os.path.join(ROOT, "experiments", "e1", "results")

BASELINE_ORDER = [
    ("input (no-op)", "input.json"),
    ("global_gain", "global_gain.json"),
    ("gradient_domain", "gradient_domain.json"),
    ("guo_paired_regions", "guo_paired_regions.json"),
    ("boundary_gain", "boundary_gain.json"),
    ("ORACLE region gain", "oracle_region_gain_linear.json"),
    ("ORACLE smooth gain", "oracle_smooth_gain_linear.json"),
]

E1_ORDER = [
    ("e1 IAT (verify on)", os.path.join(E1, "e1_main.json")),
    ("e1 IAT (verify off)", os.path.join(E1, "e1_main_noverify.json")),
]

ARM_ORDER = [
    ("arm1  fixed s, fixed f_beta", "ablation_8_1_arm1_fixed_s_fixed_b.json"),
    ("arm2  fixed s, geo f_beta", "ablation_8_1_arm2_fixed_s_geo_b.json"),
    ("arm3  geo s, fixed f_beta", "ablation_8_1_arm3_geo_s_fixed_b.json"),
    ("arm4  geo s, geo f_beta", "ablation_8_1_arm4_geo_s_geo_b.json"),
    ("arm3-lit  geo s, no band", "ablation_8_1_arm3lit_geo_s_no_band.json"),
]


def section(title, res, full=True):
    print(f"\n### {title}\n")
    print(table_quality(res))
    print()
    print(table_error(res))
    if full and any(r.get("metrics_full_res") for r in res.values()):
        print("\n**Native resolution (480x640), same metrics**\n")
        print(table_quality(res, key="metrics_full_res"))
        print()
        print(table_error(res, key="metrics_full_res"))
    print("\n**Latency** (per image, inference only, native resolution)\n")
    print(table_latency(res))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if which == "baseline":
        res = load_results({k: os.path.join(BASE, v) for k, v in BASELINE_ORDER})
        section("ISTD+ test (540 images), 256x256 protocol", res)
        print("\n### Machine\n")
        print(machine_block(res))
    elif which == "e1":
        base = load_results({k: os.path.join(BASE, v) for k, v in BASELINE_ORDER})
        e1 = load_results(dict(E1_ORDER))
        merged = {}
        merged.update({k: v for k, v in e1.items()})
        merged.update(base)
        section("e1 vs baselines, ISTD+ test (540 images), 256x256", merged)
        arms = load_results({k: os.path.join(E1, v) for k, v in ARM_ORDER})
        if arms:
            print("\n### Gating 8.1 -- four-arm ablation (256x256)\n")
            print(table_quality(arms))
            print()
            print(table_error(arms))
            print()
            print(table_latency(arms))
        ms = {}
        for p in sorted(glob.glob(os.path.join(E1, "mask_sensitivity_*.json"))):
            n = os.path.basename(p).replace("mask_sensitivity_", "").replace(".json", "")
            ms[f"mask {n} px"] = json.load(open(p, encoding="utf-8"))
        if ms:
            order = sorted(ms, key=lambda k: int(k.split()[1]))
            print("\n### Gating 8.5 -- mask sensitivity (256x256)\n")
            print(table_quality({k: ms[k] for k in order}))
    else:
        print("usage: dump_tables.py [baseline|e1]")


if __name__ == "__main__":
    main()
