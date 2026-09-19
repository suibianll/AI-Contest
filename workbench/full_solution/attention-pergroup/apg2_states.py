"""A-PG2 state check: control equivalence and per-group reachability.

Runs `hif4_calibration_attention` on every real attention layer for
  1. the parent root (workspace solution.py),
  2. the candidate with `_ATTN_ROTATION_PER_GROUP = False` (control),
  3. the candidate with the mechanism on,
and compares the returned states field by field (tensor bytes for tensors),
then exercises the deployed Q/K/V dynamic APIs on the candidate state.

    .venv/Scripts/python.exe apg2_states.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidate_apg2" / "solution.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().to("cpu").contiguous().numpy().tobytes()


def state_signature(state) -> dict:
    if torch.is_tensor(state):
        return {"kind": "tensor", "sha": hashlib.sha256(tensor_bytes(state)).hexdigest()[:16],
                "shape": tuple(state.shape), "dtype": str(state.dtype)}
    if isinstance(state, dict):
        return {str(k): state_signature(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return {"kind": "seq", "items": [state_signature(v) for v in state]}
    return {"kind": "scalar", "value": state}


def diff_states(left, right, path="") -> list:
    """Return the list of field paths that differ."""
    out: list = []
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return [path + " <tensor/non-tensor>"]
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            return [path + " <shape/dtype>"]
        if tensor_bytes(left) != tensor_bytes(right):
            return [path + " <bytes>"]
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left:
                out.append(path + f".{key} <missing-left>")
            elif key not in right:
                out.append(path + f".{key} <missing-right>")
            else:
                out.extend(diff_states(left[key], right[key], path + f".{key}"))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path + " <length>"]
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(diff_states(a, b, path + f"[{index}]"))
        return out
    if left != right:
        out.append(path + f" <{left!r} vs {right!r}>")
    return out


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent = load("apg2_parent", ROOT / "solution.py")
    candidate = load("apg2_candidate", CANDIDATE)
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [
        layer for layer in range(len(pack["calibration_qkv"][0]))
        if all(pack["calibration_qkv"][s][layer] is not None for s in range(splits))
    ]
    print(f"device={dev} q_heads={qh} kv_heads={kvh} head_dim={hd} layers={layers}", flush=True)

    def run(module, layer: int):
        calibration = [
            {
                role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(splits)
        ]
        start = time.perf_counter()
        states = module.hif4_calibration_attention(calibration, qh, kvh, hd)
        wall = time.perf_counter() - start
        return states, wall

    all_ok = True
    for layer in layers:
        parent_states, parent_wall = run(parent, layer)
        candidate._ATTN_ROTATION_PER_GROUP = False
        control_states, control_wall = run(candidate, layer)
        candidate._ATTN_ROTATION_PER_GROUP = True
        active_states, active_wall = run(candidate, layer)

        control_diff = diff_states(parent_states, control_states)
        print(f"\nlayer {layer}")
        print(f"  parent wall {parent_wall:.3f}s  control wall {control_wall:.3f}s  "
              f"active wall {active_wall:.3f}s", flush=True)
        print(f"  CONTROL (per-group off) vs parent diffs: {control_diff or 'IDENTICAL'}")
        if control_diff:
            all_ok = False

        for side in ("q_state", "k_state", "v_state"):
            parent_sig = state_signature(parent_states[side])
            active_sig = state_signature(active_states[side])
            changed = [
                key for key in set(parent_sig) | set(active_sig)
                if parent_sig.get(key) != active_sig.get(key)
            ]
            print(f"  {side}: changed fields vs parent = {changed}")
        q_state = active_states["q_state"]
        for key in ("pg_rotation_arm", "pg_rotation_candidates", "pg_rotation_accepted_groups"):
            print(f"    {key} = {q_state.get(key)}")
        print(f"    rotation_block = {q_state.get('rotation_block')}")
        block = q_state.get("rotation_block")
        if isinstance(block, list) and any(int(v) > 0 for v in block):
            states = active_states
            sample = pack["calibration_qkv"][0][layer]
            for role in ("q", "k", "v"):
                pair = tuple(t.to(dev) for t in v2._pair(sample[("q", "k", "v").index(role)]))
                args = (pair[0], pair[1], qh if role == "q" else kvh, hd, states[role + "_state"])
                out = getattr(candidate, f"hif4_dynamic_quantize_{role}")(*args)
                finite = all(
                    torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v)
                )
                print(f"    dynamic {role}: keys={sorted(out)} finite={bool(finite)}")
            for side in ("q_state", "k_state", "v_state"):
                v2.validate_state(states[side])
            print("    validate_state: OK")

    print("\nCONTROL OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
