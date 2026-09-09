"""Prune the regenerable proxy-v3 calibration cache.

The calibration cache is keyed by solution identity, so any cache belonging to a
dead candidate can never be hit again; `--calibration-cache-mode auto` rebuilds
it when missing.  What is NOT regenerable cheaply is the dense input cache
`qwen3.5-4b-proxy-v2.pt`, which is never touched by this script.

Naming:  <solution-sha256-prefix-16>-<linear|attention|both>-<config-hash>.pt

Usage
-----
    # dry run (default): only report
    python workbench/cache_cleanup/prune_calibration_cache.py

    # keep only the current root, delete the rest
    python workbench/cache_cleanup/prune_calibration_cache.py --keep 56dc805d6e5a3aef

    # keep root + fallback, actually delete
    python workbench/cache_cleanup/prune_calibration_cache.py \
        --keep 56dc805d6e5a3aef --keep 839adb1e617c3115 --apply

    # resolve the current root's prefix yourself
    python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('solution.py').read_bytes()).hexdigest()[:16])"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "artifacts" / "official_eval" / "cache"
CALIB_DIR = CACHE_DIR / "proxy-v3-calibration"
DENSE_CACHE = "qwen3.5-4b-proxy-v2.pt"

GIB = 1024**3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="append", default=[],
                    help="solution SHA prefix (16 hex) to keep; repeatable")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry run)")
    args = ap.parse_args()

    keep = sorted({k.lower() for k in args.keep})

    if not CALIB_DIR.exists():
        print(f"calibration cache dir missing: {CALIB_DIR}")
        return 1
    if not (CACHE_DIR / DENSE_CACHE).exists():
        print(f"!! dense cache {DENSE_CACHE} missing - refusing to run")
        return 1

    files = [f for f in os.listdir(CALIB_DIR) if f.endswith(".pt")]
    keep_files = [f for f in files if f.startswith(tuple(keep))]
    targets = [f for f in files if not f.startswith(tuple(keep))]

    total = sum(os.path.getsize(CALIB_DIR / f) for f in files)
    keep_bytes = sum(os.path.getsize(CALIB_DIR / f) for f in keep_files)
    del_bytes = sum(os.path.getsize(CALIB_DIR / f) for f in targets)

    print(f"calibration cache : {len(files)} files / {total / GIB:.2f} GB")
    print(f"  keep  ({', '.join(keep) or 'none'}) : {len(keep_files)} files / {keep_bytes / GIB:.2f} GB")
    print(f"  delete                       : {len(targets)} files / {del_bytes / GIB:.2f} GB")
    print(f"  dense cache (never touched)  : {os.path.getsize(CACHE_DIR / DENSE_CACHE) / GIB:.2f} GB")

    if not args.apply:
        print("\n[dry run] nothing deleted. re-run with --apply to delete.")
        return 0
    if not keep:
        print("\n!! refusing to delete with an empty keep set")
        return 1

    done = freed = 0
    for i, f in enumerate(targets, 1):
        p = CALIB_DIR / f
        try:
            size = os.path.getsize(p)
            os.remove(p)
            done += 1
            freed += size
        except OSError as exc:
            print(f"  failed: {f}: {exc}")
        if i % 100 == 0:
            print(f"  {i}/{len(targets)}  freed {freed / GIB:.1f} GB", flush=True)

    print(f"\ndeleted {done} files, freed {freed / GIB:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
