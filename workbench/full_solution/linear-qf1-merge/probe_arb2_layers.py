"""Why does the same A-RB2 constant give +0.0011 on v250 but +0.0113 on v252?

Calibrates layer 0 (and 5) with four modules and diffs the returned q/k/v
states field by field, then records which rotation blocks the search actually
evaluates (via a wrapper on `_attention_deployed_mse`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sig(value, depth=0):
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), str(value.dtype),
                value.detach().to("cpu").contiguous().numpy().tobytes())
    if isinstance(value, dict):
        return {str(k): sig(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return ["seq", [sig(v) for v in value]]
    return value


def diff(left, right, path=""):
    out = []
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return [path + " <tensor/non-tensor>"]
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
            return [path + " <shape/dtype>"]
        if sig(left)[3] != sig(right)[3]:
            out.append(path + " <bytes>")
        return out
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left:
                out.append(path + f".{key} <missing-left>")
            elif key not in right:
                out.append(path + f".{key} <missing-right>")
            else:
                out.extend(diff(left[key], right[key], path + f".{key}"))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path + " <length>"]
        for i, (a, b) in enumerate(zip(left, right)):
            out.extend(diff(a, b, path + f"[{i}]"))
        return out
    if left != right:
        out.append(path + f" <{left!r} vs {right!r}>")
    return out


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    mods = {
        "v250": load("m_v250", ROOT / "solution.py"),
        "c2": load("m_c2", ROOT / "workbench/full_solution/attention-rotation-blocks/candidate2.py"),
        "v252": load("m_v252", HERE / "candidate_v252" / "solution.py"),
        "v253": load("m_v253", HERE / "candidate_v253" / "solution.py"),
    }
    for name, mod in mods.items():
        print(f"{name}: blocks={getattr(mod, '_ATTN_ROTATION_BLOCKS')}", flush=True)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])

    for layer in (0, 5):
        calibration = [
            {
                role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(splits)
        ]
        states = {}
        blocks_seen = {}
        for name, mod in mods.items():
            seen = []
            orig = mod._attention_deployed_mse

            def patched(*args, _seen=seen, _orig=orig, **kwargs):
                q_state = args[4] if len(args) > 4 else kwargs.get("q_state")
                if isinstance(q_state, dict):
                    _seen.append(q_state.get("rotation_block"))
                return _orig(*args, **kwargs)

            mod._attention_deployed_mse = patched
            states[name] = mod.hif4_calibration_attention(calibration, qh, kvh, hd)
            mod._attention_deployed_mse = orig
            blocks_seen[name] = [b for b in seen if b is not None]
        print(f"\nlayer {layer}")
        for name in mods:
            q = states[name]["q_state"]
            print(f"  {name}: rotation_block={q.get('rotation_block')} "
                  f"rotation={'yes' if q.get('rotation') is not None else 'no'} "
                  f"arm={q.get('agr1_arm')} blocks_evaluated={sorted(set(blocks_seen[name]))}")
        for pair in (("v250", "c2"), ("v250", "v252"), ("c2", "v253"), ("v252", "v253")):
            left, right = pair
            d = diff(states[left]["q_state"], states[right]["q_state"])
            print(f"  diff {left}->{right} q_state:", d[:12] if d else "IDENTICAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
