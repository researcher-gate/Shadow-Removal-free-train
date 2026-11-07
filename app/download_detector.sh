#!/usr/bin/env bash
# Fetch the SILT shadow detector used by the PoC.
#
#   bash app/download_detector.sh            # ISTD checkpoint (default)
#   bash app/download_detector.sh sbu        # SBU checkpoint (better on photos)
#   bash app/download_detector.sh all
#
# SILT (Hu et al., ICCV 2023) -- https://github.com/hanyangclarence/SILT
# PVT-v2 backbone, ~198 MB per checkpoint, runs on CPU.
#
#   istd  trained on ISTD, BER 1.11. Same dataset family e2 was tuned on, so
#         detector and remover agree on what a shadow is. Best for benchmark
#         images.
#   sbu   trained on SBU, BER 4.36. Far more varied scenes; generalises better
#         to everyday photos. Best for a demo on your own pictures.
#
# SECURITY: these are pickled checkpoints. app/detector.py loads them with
# torch.load(weights_only=True), which will not execute code from the file.
# If a checkpoint refuses to open that way, convert it deliberately with
# app/convert_checkpoint.py rather than relaxing the loader.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SILT_DIR="$ROOT/third_party/SILT"
PYTHON="${PYTHON:-python}"

ISTD_ID="1jT2yySs_ZxG_oyD-D5xkxeyPBc1igqpL"
SBU_ID="1CvO6xoXdUw72xGFyhHfroi4LjGyEjBKD"

log() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

if [ ! -d "$SILT_DIR/.git" ]; then
  log "cloning SILT"
  git clone --depth 1 https://github.com/hanyangclarence/SILT.git "$SILT_DIR"
else
  log "SILT repo already present"
fi

mkdir -p "$SILT_DIR/pretrained"
"$PYTHON" -c "import gdown" 2>/dev/null || {
  log "installing gdown"; "$PYTHON" -m pip install --quiet gdown; }

fetch() {  # fetch <id> <file> <label>
  local out="$SILT_DIR/pretrained/$2"
  if [ -f "$out" ]; then
    log "$3: already present ($(du -h "$out" | cut -f1))"
    return
  fi
  log "$3: downloading ~198 MB"
  "$PYTHON" -m gdown "$1" -O "$out"
  [ -s "$out" ] || die "download produced an empty file"
  log "$3: $(du -h "$out" | cut -f1)"
  "$PYTHON" - "$out" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
print("    sha256:", h.hexdigest())
PY
}

case "${1:-istd}" in
  istd) fetch "$ISTD_ID" silt_pvtb3_istd.pth "SILT PVT-v2-B3 (ISTD)" ;;
  sbu)  fetch "$SBU_ID"  silt_pvtb5_sbu.pth  "SILT PVT-v2-B5 (SBU)" ;;
  all)  fetch "$ISTD_ID" silt_pvtb3_istd.pth "SILT PVT-v2-B3 (ISTD)"
        fetch "$SBU_ID"  silt_pvtb5_sbu.pth  "SILT PVT-v2-B5 (SBU)" ;;
  *)    die "usage: $0 [istd|sbu|all]" ;;
esac

log "done"
echo
echo "Start the UI with:"
echo "  python app/ui.py"
