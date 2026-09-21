#!/usr/bin/env bash
# Fetch the pretrained supervised baselines used by baseline/run_baseline.py.
#
#   bash third_party/download_models.sh              # everything
#   bash third_party/download_models.sh bmnet        # just BMNet
#   bash third_party/download_models.sh shadowformer # just ShadowFormer
#   VERIFY_ONLY=1 bash third_party/download_models.sh
#
# Idempotent: re-running re-verifies and skips anything already correct.
# Works in Git Bash on Windows and in any POSIX shell.
#
# What lands where
#   third_party/BMNet/MainNet/pretrain/AISTD.pth                2.0 MB
#       BMNet (Zhu et al., CVPR 2022).  Weights are committed to the upstream
#       repo, so cloning is the whole download.  "AISTD" is ISTD+.
#   third_party/ShadowFormer/pretrained/ShadowFormer_ISTDplus.pth   137 MB
#       ShadowFormer (Guo et al., AAAI 2023).  Weights come from the authors'
#       Google Drive release; the repo does not carry them.
#
# The BMNet source hard-codes an absolute path to condition.pth. It is patched
# at load time by baseline/bmnet_wrapper.py, not here, so this script never
# edits third-party source.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BMNET_URL="https://github.com/KevinJ-Huang/BMNet.git"
SHADOWFORMER_URL="https://github.com/GuoLanqing/ShadowFormer.git"
SHADOWFORMER_GDRIVE_ID="10pBsJenoWGriZ9kjWOcE4l4Kzg-F1TFd"

BMNET_WEIGHTS="BMNet/MainNet/pretrain/AISTD.pth"
BMNET_COND="BMNet/MainNet/pretrain/condition.pth"
SF_WEIGHTS="ShadowFormer/pretrained/ShadowFormer_ISTDplus.pth"

BMNET_SHA="e8d035509399bac281203974baa2011474c810bd51094fc03ee4d0b6d59c296d"
BMNET_COND_SHA="51f516206be11e7c8757b4f8debe0a905e35f0832d231645d351f70777dd453f"
SF_SHA="2748060149908df37cc65f0695ef61d64cd25847aba0c35af36823f9b780f5b2"

PYTHON="${PYTHON:-python}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

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

check() {  # check <path> <expected sha> <label>
  [ -f "$1" ] || { warn "$3: missing"; return 1; }
  local got; got="$(sha_of "$1")"
  if [ "$got" = "$2" ]; then
    log "$3: ok ($(du -h "$1" | cut -f1))"
    return 0
  fi
  warn "$3: checksum mismatch"
  warn "  expected $2"
  warn "  got      $got"
  return 1
}

clone_if_needed() {  # clone_if_needed <dir> <url>
  if [ -d "$1/.git" ]; then
    log "$1: repo already present"
  else
    [ -e "$1" ] && die "$1 exists but is not a git checkout; remove it and retry"
    log "cloning $1"
    git clone --depth 1 "$2" "$1"
  fi
}

need_python_pkg() {  # need_python_pkg <import name> <pip name>
  "$PYTHON" -c "import $1" >/dev/null 2>&1 || {
    log "installing $2"
    "$PYTHON" -m pip install --quiet "$2"
  }
}

fetch_bmnet() {
  log "BMNet (Zhu et al., CVPR 2022) -- ISTD+/AISTD checkpoint"
  [ "$VERIFY_ONLY" = "1" ] || clone_if_needed "BMNet" "$BMNET_URL"
  check "$BMNET_WEIGHTS" "$BMNET_SHA" "BMNet AISTD.pth" \
    || die "BMNet weights missing or corrupt. They are committed upstream, so a
     fresh clone should restore them:  rm -rf third_party/BMNet && rerun"
  check "$BMNET_COND" "$BMNET_COND_SHA" "BMNet condition.pth" \
    || die "BMNet condition.pth missing or corrupt (same fix as above)"
}

fetch_shadowformer() {
  log "ShadowFormer (Guo et al., AAAI 2023) -- ISTD+ checkpoint"
  [ "$VERIFY_ONLY" = "1" ] || clone_if_needed "ShadowFormer" "$SHADOWFORMER_URL"
  if ! check "$SF_WEIGHTS" "$SF_SHA" "ShadowFormer_ISTDplus.pth"; then
    [ "$VERIFY_ONLY" = "1" ] && die "ShadowFormer weights missing"
    need_python_pkg gdown gdown
    mkdir -p "$(dirname "$SF_WEIGHTS")"
    log "downloading 137 MB from Google Drive"
    "$PYTHON" -m gdown "$SHADOWFORMER_GDRIVE_ID" -O "$SF_WEIGHTS"
    check "$SF_WEIGHTS" "$SF_SHA" "ShadowFormer_ISTDplus.pth" \
      || die "download completed but the checksum is wrong. Google Drive
     sometimes serves an HTML quota page instead of the file -- delete
     $SF_WEIGHTS, wait, and retry, or fetch it by hand from
     https://drive.google.com/file/d/$SHADOWFORMER_GDRIVE_ID/view"
  fi
}

WANT="${*:-all}"
case "$WANT" in
  all)          fetch_bmnet; fetch_shadowformer ;;
  bmnet)        fetch_bmnet ;;
  shadowformer) fetch_shadowformer ;;
  *)            die "usage: $0 [all|bmnet|shadowformer]" ;;
esac

if [ "$VERIFY_ONLY" != "1" ]; then
  log "python deps for the wrappers"
  need_python_pkg einops einops
  need_python_pkg timm timm
  need_python_pkg natsort natsort

  log "smoke-testing the wrappers (loads weights, no evaluation)"
  cd ..
  "$PYTHON" - <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
ok = True
for label, mod, fn in (("BMNet", "baseline.bmnet_wrapper", "meta"),
                       ("ShadowFormer", "baseline.shadowformer_wrapper", "meta")):
    try:
        m = getattr(__import__(mod, fromlist=[fn]), fn)()
        flag = "ok" if m["missing_keys"] == m["unexpected_keys"] == 0 else "KEY MISMATCH"
        print(f"  {label:14s} {m['n_parameters']:>11,} params  "
              f"missing={m['missing_keys']} unexpected={m['unexpected_keys']}  {flag}")
        ok &= m["missing_keys"] == m["unexpected_keys"] == 0
    except FileNotFoundError as e:
        print(f"  {label:14s} not installed ({e})")
    except Exception as e:
        print(f"  {label:14s} FAILED: {type(e).__name__}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY
fi

log "done"
echo
echo "Run them with:"
echo "  python baseline/run_baseline.py --supervised bmnet shadowformer"
echo
echo "Note: both run on CPU so their latency is measured on the same hardware as"
echo "the training-free methods. On this class of machine expect roughly"
echo "3.5 s/image for BMNet and 7-18 s/image for ShadowFormer (~30 min and"
echo "1-2.5 h respectively over the 540-image ISTD+ test split)."
