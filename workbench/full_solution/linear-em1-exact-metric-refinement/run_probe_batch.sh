#!/usr/bin/env bash
# Launch the headroom probe for several (layer, role) pairs on the six shard caches.
# CPU only; never touches the GPU.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CACHE="$ROOT/artifacts/official_eval/cache/proxy-v3-calibration"
PACK="$ROOT/artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
OUT="$HERE/probe_out"
mkdir -p "$OUT"

cache_of() {
  case $(( $1 % 6 )) in
    0) echo "$CACHE/56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt" ;;
    1) echo "$CACHE/56dc805d6e5a3aef-linear-b9e7d8eab443a1677733.pt" ;;
    2) echo "$CACHE/56dc805d6e5a3aef-linear-da87174d8303c3079167.pt" ;;
    3) echo "$CACHE/56dc805d6e5a3aef-linear-afa29784c167f5f4beb7.pt" ;;
    4) echo "$CACHE/56dc805d6e5a3aef-linear-0a1a5e3d3972cea2c1b8.pt" ;;
    5) echo "$CACHE/56dc805d6e5a3aef-linear-2a592e71b8c327b6dc92.pt" ;;
  esac
}

run() {  # run <layer> <role> <window> <variants>
  local layer=$1 role=$2 window=$3 variants=$4
  local tag="L${layer}-${role}-w${window}-${variants//,/_}"
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -u "$HERE/probe_headroom.py" \
    --cache "$(cache_of "$layer")" --pack "$PACK" --layer "$layer" --role "$role" \
    --window "$window" --variants "$variants" > "$OUT/$tag.log" 2>&1
  echo "done $tag"
}

for spec in "$@"; do
  # spec = layer:role:window:variants
  IFS=: read -r l r w v <<< "$spec"
  run "$l" "$r" "$w" "$v" &
done
wait
echo "ALL DONE"
