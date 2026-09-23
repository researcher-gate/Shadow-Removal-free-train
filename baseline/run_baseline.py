"""Run every baseline (and the two oracles) on the ISTD+ test split.

    python baseline/run_baseline.py                 # everything, 540 images
    python baseline/run_baseline.py --limit 40      # quick smoke run
    python baseline/run_baseline.py --only boundary_gain_linear
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline import methods as B
from baseline.methods import ORACLES, REGISTRY
from common.runner import run_method

RES = os.path.join(ROOT, "baseline", "results")
OUTDIR = os.path.join(ROOT, "baseline", "outputs")

FACTORY = {
    "global_gain": B.make_global_gain,
    "boundary_gain": B.make_boundary_gain,
    "guo_paired_regions": B.make_guo_paired,
    "gradient_domain": B.make_gradient_domain,
}


def supervised_registry(which):
    """Pretrained supervised references.  Not training-free, not tuned by us --
    they are the upper reference the guideline's sec. 7.2 asks for, run through
    the identical harness so their metrics AND their latency are comparable.

    Both run on CPU: that keeps the latency column measured on one machine, but
    it is not a deployment number for either network.
    """
    reg, meta = {}, {}
    if "bmnet" in which:
        from baseline.bmnet_wrapper import make_bmnet, meta as bmeta
        reg["bmnet_supervised"] = make_bmnet()
        meta["bmnet_supervised"] = bmeta()
    if "shadowformer" in which:
        from baseline.shadowformer_wrapper import make_shadowformer, meta as smeta
        reg["shadowformer_supervised"] = make_shadowformer()
        meta["shadowformer_supervised"] = smeta()
    return reg, meta


def tuned_registry():
    """Final baselines = the configuration each family won with on TRAIN.

    Falls back to the full untuned registry if tune.py has not been run, but
    the reported numbers should always come from the tuned set: a baseline
    whose knobs were never fitted is a straw man, and one whose knobs were
    fitted on test is a leak.
    """
    p = os.path.join(RES, "tuning_train.json")
    if not os.path.exists(p):
        print("WARNING: no tuning_train.json; using untuned defaults")
        return {k: v() for k, v in REGISTRY.items()}, {}
    t = json.load(open(p, encoding="utf-8"))
    reg = {"input": B.method_input}
    cfgs = {}
    for fam, rec in t["families"].items():
        cfg = rec["best"]["cfg"]
        reg[fam] = FACTORY[fam](**cfg)
        cfgs[fam] = cfg
    return reg, cfgs


def describe():
    """Print every method the runner can execute, and whether it is ready.

    Kinds:
      training-free  weightless, mask-guided -- e1's actual peer group
      supervised     pretrained by the original authors on ISTD+ (AISTD)
      oracle         uses the ground-truth shadow-free image; a ceiling, not a
                     baseline, and never quotable as a result
    """
    tuned_path = os.path.join(RES, "tuning_train.json")
    tuned = json.load(open(tuned_path, encoding="utf-8")) if os.path.exists(tuned_path) else None

    # measured on ISTD+ test, 480x640, 8 CPU cores; used only for the estimate
    MS = {"input": 0.3, "global_gain": 111, "gradient_domain": 207,
          "guo_paired_regions": 2643, "boundary_gain": 154,
          "oracle_region_gain_linear": 190, "oracle_smooth_gain_linear": 152,
          "bmnet_supervised": 3450, "shadowformer_supervised": 10000}
    OVERHEAD_MS = 350          # per image: disk I/O + metrics at both resolutions

    rows = []
    for fam in REGISTRY if tuned is None else ["input"] + list(tuned["families"]):
        if tuned is None and fam != "input":
            cfg, ready, why = None, False, "run baseline/tune.py first"
        elif fam == "input":
            cfg, ready, why = None, True, ""
        else:
            cfg, ready, why = tuned["families"][fam]["best"]["cfg"], True, ""
        rows.append((fam, "training-free", cfg, ready, why))

    for name, mod, pretty in (
            ("bmnet_supervised", "baseline.bmnet_wrapper", "BMNet (CVPR22)"),
            ("shadowformer_supervised", "baseline.shadowformer_wrapper",
             "ShadowFormer (AAAI23)")):
        try:
            m = __import__(mod, fromlist=["meta"]).meta()
            ready, why = True, f"{pretty}, {m['n_parameters']:,} params"
        except Exception as e:
            ready = False
            why = (f"{pretty}: run third_party/download_models.sh"
                   if isinstance(e, FileNotFoundError) else
                   f"{pretty}: {type(e).__name__}: {e}")
        rows.append((name, "supervised", None, ready, why))

    for name in ORACLES:
        rows.append((name, "oracle", None, True, "uses ground truth -- a ceiling"))

    print("Methods available to baseline/run_baseline.py   (split=test, 540 images)\n")
    print(f"  {'name':26s} {'kind':14s} {'ready':6s} {'est. run':>9s}  notes")
    print(f"  {'-'*26} {'-'*14} {'-'*6} {'-'*9}  {'-'*40}")
    total = 0.0
    for name, kind, cfg, ready, why in rows:
        est = (MS.get(name, 500) + OVERHEAD_MS) * 540 / 1000.0
        if ready:
            total += est
        est_s = f"{est/60:.0f} min" if est < 3600 else f"{est/3600:.1f} h"
        note = why or (str(cfg) if cfg else "")
        print(f"  {name:26s} {kind:14s} {'yes' if ready else 'NO':6s} {est_s:>9s}  {note}")
    print(f"\n  everything ready, end to end: ~{total/3600:.1f} h "
          f"(~{(total - (MS['shadowformer_supervised']+OVERHEAD_MS)*540/1000)/3600:.1f} h "
          f"without ShadowFormer)")
    print("\nExamples:")
    print("  python baseline/run_baseline.py                          # training-free + oracles")
    print("  python baseline/run_baseline.py --no-oracles             # training-free only")
    print("  python baseline/run_baseline.py --only boundary_gain     # one method")
    print("  python baseline/run_baseline.py --supervised bmnet       # add BMNet")
    print("  python baseline/run_baseline.py --limit 20 --tag _smoke  # quick check")
    print("  python baseline/run_baseline.py --no-lpips               # offline, no cached backbone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-oracles", action="store_true")
    ap.add_argument("--supervised", nargs="*", default=[],
                    choices=["bmnet", "shadowformer"],
                    help="pretrained supervised references to include "
                         "(slow on CPU: BMNet ~3.5 s/img, ShadowFormer ~10 s/img)")
    ap.add_argument("--save-images", nargs="*", default=["boundary_gain"],
                    help="methods whose outputs are written to disk")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-lpips", action="store_true",
                    help="skip LPIPS. It is the only metric that needs a network "
                         "on first use (torchvision fetches the AlexNet backbone, "
                         "~244 MB, into ~/.cache/torch). Once cached the run is "
                         "fully offline; use this on a machine where it is not.")
    ap.add_argument("--list", action="store_true",
                    help="list every runnable method with its readiness and "
                         "estimated runtime, then exit")
    args = ap.parse_args()

    if args.list:
        describe()
        return

    reg, cfgs = tuned_registry()
    sup_reg, sup_meta = supervised_registry(args.supervised)
    reg.update(sup_reg)
    todo = list(reg.items())
    oracles = [] if args.no_oracles else list(ORACLES.items())
    if args.only:
        todo = [(k, v) for k, v in todo if k in args.only]
        oracles = [(k, v) for k, v in oracles if k in args.only]

    summary = {}
    for name, fn in todo:
        print(f"== {name}  {cfgs.get(name, {})}")
        save = os.path.join(OUTDIR, name) if name in (args.save_images or []) else None
        r = run_method(name, fn, split=args.split,
                       out_json=os.path.join(RES, f"{name}{args.tag}.json"),
                       save_dir=save, limit=args.limit,
                       with_lpips=not args.no_lpips,
                       extra_meta=({"supervised": True,
                                    "trained_by": "original authors, ISTD+ "
                                                  "(AISTD) checkpoint",
                                    "model": sup_meta[name]}
                                   if name in sup_meta else
                                   {"config": cfgs.get(name),
                                    "config_selected_on": "ISTD+ train, 60 images, "
                                                          "see results/tuning_train.json"}))
        summary[name] = {"metrics_256": r["metrics_256"],
                         "metrics_full_res": r["metrics_full_res"],
                         "latency": r["latency"]}

    for name, fn in oracles:
        print(f"== {name} (ORACLE - uses ground truth)")
        r = run_method(name, fn, split=args.split,
                       out_json=os.path.join(RES, f"{name}{args.tag}.json"),
                       limit=args.limit, needs_gt=True,
                       with_lpips=not args.no_lpips,
                       extra_meta={"oracle": True,
                                   "note": "uses ground-truth shadow-free image; "
                                           "an upper bound, not a baseline"})
        summary[name] = {"metrics_256": r["metrics_256"],
                         "metrics_full_res": r["metrics_full_res"],
                         "latency": r["latency"], "oracle": True}

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"summary{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split,
                   "n_images": args.limit or "all",
                   "results": summary}, f, indent=2)
    print("summary ->", path)


if __name__ == "__main__":
    main()
