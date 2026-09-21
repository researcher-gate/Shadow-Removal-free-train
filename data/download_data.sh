#!/usr/bin/env bash
# Build the ISTD+ dataset from its two upstream sources.
#
#   bash data/download_data.sh              # download, extract, assemble
#   bash data/download_data.sh --test-only  # skip the 1330 training triplets
#   VERIFY_ONLY=1 bash data/download_data.sh
#
# Idempotent: re-running verifies what is there and fetches only what is not.
# Works in Git Bash on Windows and in any POSIX shell.
#
# WHY TWO SOURCES
#   ISTD+ is not distributed as one archive.  It is the original ISTD shadow
#   images and masks, combined with Le & Samaras' colour-ADJUSTED shadow-free
#   images.  Using original ISTD's shadow-free frames instead would silently
#   change the task: in original ISTD the shadow-free frame carries a global
#   colour/exposure shift relative to the shadow frame, so a method gets scored
#   partly on guessing that shift, and any spectral measurement of the residual
#   is contaminated by it.  That is exactly why the guideline calls for ISTD+.
#
#   1. Original ISTD (Wang et al., CVPR 2018)   -> shadow images + masks
#      Mirrored on Hugging Face (CK234/tmp_ISTD) because the authors' own link
#      is a Google Drive folder that rate-limits badly at this size.
#   2. Adjusted ISTD (Le & Samaras, ICCV 2019 / PAMI 2021)  -> shadow-free
#      1.07 GB zip from the SID project page.
#
# LAYOUT PRODUCED
#   data/downloads/ISTD_adjusted.zip     the raw archive (kept, for re-extract)
#   data/raw_ISTD/{train,test}/...       original ISTD as mirrored
#   data/adjusted_ISTD/...               colour-adjusted shadow-free images
#   data/ISTD_plus/{train,test}/{A,B,C}  <- what common/istd.py reads
#   data/ISTD_plus/manifest.json         provenance + spot-check hashes
#
#   A = shadow image, B = shadow mask, C = shadow-free (ADJUSTED).
#   train: 1330 triplets, test: 540 triplets.  Files are hard-linked where the
#   filesystem allows, so ISTD_plus costs almost no extra disk.
#
# DISK: ~2.6 GB for everything.  Once ISTD_plus exists you may delete
#       data/downloads/ and data/raw_ISTD/ -- but only if hard links worked
#       (they do on NTFS and ext4); check with `du -sh data/ISTD_plus` first.
#
# CITE BOTH PAPERS if you use this: the adjustment is Le & Samaras', the
# dataset itself is Wang et al.'s.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$HERE"

ADJUSTED_GDRIVE_ID="1rsCSWrotVnKFUqu9A_Nw9Uf-bJq_ryOv"
ADJUSTED_ZIP="downloads/ISTD_adjusted.zip"
ADJUSTED_SHA="96402db0c581b6d636437df06df484b73f8e176e3ef96c27e5007837ee5c08bc"
ADJUSTED_BYTES=1068770897
HF_REPO="CK234/tmp_ISTD"

PYTHON="${PYTHON:-python}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
TEST_ONLY=0
[ "${1:-}" = "--test-only" ] && TEST_ONLY=1

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

count() { [ -d "$1" ] && find "$1" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ' || echo 0; }

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum    >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else "$PYTHON" -c "
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())" "$1"
  fi
}

need_python_pkg() {
  "$PYTHON" -c "import $1" >/dev/null 2>&1 || {
    log "installing $2"; "$PYTHON" -m pip install --quiet "$2"; }
}

expect() {  # expect <dir> <n> <label>
  local got; got="$(count "$1")"
  if [ "$got" = "$2" ]; then log "$3: $got files ok"; return 0; fi
  warn "$3: expected $2 png, found $got"; return 1
}

# ---------------------------------------------------------------- 1. adjusted
fetch_adjusted() {
  log "Adjusted ISTD shadow-free images (Le & Samaras)"
  if [ -f "$ADJUSTED_ZIP" ] && [ "$(stat -c%s "$ADJUSTED_ZIP" 2>/dev/null || stat -f%z "$ADJUSTED_ZIP")" = "$ADJUSTED_BYTES" ]; then
    log "archive present, verifying checksum (1 GB, takes a moment)"
    [ "$(sha_of "$ADJUSTED_ZIP")" = "$ADJUSTED_SHA" ] || die "archive corrupt; delete $HERE/$ADJUSTED_ZIP and retry"
    log "archive checksum ok"
  else
    [ "$VERIFY_ONLY" = "1" ] && die "archive missing"
    need_python_pkg gdown gdown
    mkdir -p downloads
    log "downloading 1.07 GB from Google Drive"
    "$PYTHON" -m gdown "$ADJUSTED_GDRIVE_ID" -O "$ADJUSTED_ZIP"
    [ "$(sha_of "$ADJUSTED_ZIP")" = "$ADJUSTED_SHA" ] || die "download checksum wrong.
     Google Drive sometimes serves an HTML quota page instead of the file.
     Delete $HERE/$ADJUSTED_ZIP, wait, and retry, or fetch by hand from
     https://drive.google.com/file/d/$ADJUSTED_GDRIVE_ID/view"
  fi

  if expect "adjusted_ISTD/test_C_fixed_official" 540 "adjusted test_C" 2>/dev/null \
     && { [ "$TEST_ONLY" = "1" ] || expect "adjusted_ISTD/train_C_fixed_ours" 1330 "adjusted train_C" 2>/dev/null; }; then
    log "already extracted"
    return
  fi
  [ "$VERIFY_ONLY" = "1" ] && die "archive not extracted"
  log "extracting"
  "$PYTHON" - "$ADJUSTED_ZIP" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.endswith(".png") and not n.startswith("__MACOSX")]
z.extractall("adjusted_ISTD", members=names)
print(f"  extracted {len(names)} images")
PY
}

# ------------------------------------------------------------- 2. original
fetch_original() {
  log "Original ISTD shadow images + masks (Wang et al., HF mirror $HF_REPO)"
  local need_train=1
  [ "$TEST_ONLY" = "1" ] && need_train=0

  local ok=1
  expect "raw_ISTD/test/test_A" 540 "test_A" 2>/dev/null || ok=0
  expect "raw_ISTD/test/test_B" 540 "test_B" 2>/dev/null || ok=0
  if [ "$need_train" = "1" ]; then
    expect "raw_ISTD/train/train_A" 1330 "train_A" 2>/dev/null || ok=0
    expect "raw_ISTD/train/train_B" 1330 "train_B" 2>/dev/null || ok=0
  fi
  [ "$ok" = "1" ] && { log "already downloaded"; return; }
  [ "$VERIFY_ONLY" = "1" ] && die "original ISTD incomplete"

  need_python_pkg huggingface_hub huggingface_hub
  log "downloading (~2 GB, resumable; re-run if it drops)"
  TEST_ONLY="$TEST_ONLY" HF_REPO="$HF_REPO" "$PYTHON" - <<'PY'
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import snapshot_download
patterns = ["test/*"] if os.environ.get("TEST_ONLY") == "1" else None
snapshot_download(repo_id=os.environ["HF_REPO"], repo_type="dataset",
                  local_dir="raw_ISTD", max_workers=8, allow_patterns=patterns)
print("  done")
PY
}

# ------------------------------------------------------------- 3. assemble
assemble() {
  log "assembling data/ISTD_plus"
  local splits="train test"
  [ "$TEST_ONLY" = "1" ] && splits="test"
  # shellcheck disable=SC2086
  "$PYTHON" "$ROOT/tools/prepare_istd_plus.py" $splits
}

verify_final() {
  log "final check"
  expect "ISTD_plus/test/A" 540 "ISTD_plus/test/A" || die "test split incomplete"
  expect "ISTD_plus/test/B" 540 "ISTD_plus/test/B" || die "test split incomplete"
  expect "ISTD_plus/test/C" 540 "ISTD_plus/test/C" || die "test split incomplete"
  if [ "$TEST_ONLY" != "1" ]; then
    expect "ISTD_plus/train/A" 1330 "ISTD_plus/train/A" || die "train split incomplete"
    expect "ISTD_plus/train/B" 1330 "ISTD_plus/train/B" || die "train split incomplete"
    expect "ISTD_plus/train/C" 1330 "ISTD_plus/train/C" || die "train split incomplete"
  fi
  "$PYTHON" - <<'PY'
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.getcwd()))
from common.istd import load_split
for split in ("test",) if os.environ.get("TEST_ONLY") == "1" else ("train", "test"):
    s = load_split(split)
    img, mask, free = s[0].load()
    assert img.shape == free.shape and mask.shape == img.shape[:2], "shape mismatch"
    print(f"  {split}: {len(s)} triplets, first={s[0].name}, shape={img.shape}, "
          f"mask is binary={set(np.unique(mask).tolist()) <= {0, 255}}")
PY
}

fetch_adjusted
fetch_original
[ "$VERIFY_ONLY" = "1" ] || assemble
TEST_ONLY="$TEST_ONLY" verify_final

log "done"
echo
echo "Dataset ready at data/ISTD_plus/  (A=shadow, B=mask, C=shadow-free adjusted)"
echo "Provenance: data/ISTD_plus/manifest.json"
echo
echo "Next:"
echo "  bash third_party/download_models.sh        # supervised baselines"
echo "  python baseline/tune.py --n 60             # select baseline knobs on TRAIN"
echo "  python baseline/run_baseline.py            # evaluate on TEST"
