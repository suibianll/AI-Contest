"""v194 attn-a2-calibration-fused bitwise-equivalence verification.

Checks, on small fixed-seed synthetic NVFP4 Q/K/V windows:
1. Parent calibration self-determinism (run twice, bitwise identical).
2. Parent vs candidate hif4_calibration_attention states: every field bitwise
   identical (torch.equal on uint8 views; exact equality for scalars).
3. hif4_dynamic_quantize_q/k/v params bitwise identical under both states,
   finite dequantized outputs, reference_hif4 state/params legality.
4. Repo-independent single-file six-API import + finite-output smoke run in a
   subprocess from a temp directory.
5. Rough time.perf_counter calibration comparison (recorded, not a gate).
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

Q_HEADS, KV_HEADS, HEAD_DIM, TOKENS, WINDOWS = 4, 2, 64, 16, 4
APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_calib(device: torch.device, seed: int = 4194, spike: float = 0.0):
    items = []
    for index in range(WINDOWS):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        q = torch.randn(TOKENS, Q_HEADS * HEAD_DIM, generator=generator)
        k = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, generator=generator)
        v = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, generator=generator)
        if spike > 0:
            q[:, 0] *= spike
            k[:, 0] *= spike
            q[:, HEAD_DIM] *= spike
            k[:, HEAD_DIM] *= spike
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        items.append({
            "q": (q, torch.ones(TOKENS, Q_HEADS * HEAD_DIM // 16, device=device)),
            "k": (k, torch.ones(TOKENS, KV_HEADS * HEAD_DIM // 16, device=device)),
            "v": (v, torch.ones(TOKENS, KV_HEADS * HEAD_DIM // 16, device=device)),
        })
    return items


def base_states():
    common = {
        "multiplier": None,
        "permutation": None,
        "importance": None,
        "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
        "error_threshold": 1e-7,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "version": 2,
    }
    return {
        "q_state": dict(common, num_heads=Q_HEADS, head_dim=HEAD_DIM),
        "k_state": dict(common, center_mode=0, num_heads=KV_HEADS, head_dim=HEAD_DIM),
        "v_state": dict(common, num_heads=KV_HEADS, head_dim=HEAD_DIM),
    }


def compare(label: str, a, b, path: str = "root"):
    if torch.is_tensor(a):
        assert torch.is_tensor(b), f"{label}/{path}: tensor vs {type(b)}"
        assert a.dtype == b.dtype, f"{label}/{path}: dtype {a.dtype} vs {b.dtype}"
        assert tuple(a.shape) == tuple(b.shape), f"{label}/{path}: shape"
        if a.dtype == torch.bool:
            assert torch.equal(a, b), f"{label}/{path}: bool values differ"
        else:
            av = a.detach().contiguous().view(torch.uint8)
            bv = b.detach().contiguous().view(torch.uint8)
            assert torch.equal(av, bv), f"{label}/{path}: tensor bytes differ"
        return
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{label}/{path}: dict vs {type(b)}"
        assert set(a) == set(b), (
            f"{label}/{path}: key diff "
            f"{sorted(set(a) ^ set(b), key=str)}"
        )
        for key in a:
            compare(label, a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), (
            f"{label}/{path}: sequence mismatch"
        )
        for index, (xa, xb) in enumerate(zip(a, b)):
            compare(label, xa, xb, f"{path}[{index}]")
        return
    if isinstance(a, float):
        assert isinstance(b, float), f"{label}/{path}: float vs {type(b)}"
        same = a == b or (math.isnan(a) and math.isnan(b))
        assert same, f"{label}/{path}: {a!r} vs {b!r}"
        return
    assert a == b and type(a) is type(b), f"{label}/{path}: {a!r} vs {b!r}"


def timed_calibrate(module, items, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    states = module.hif4_calibration_attention(items, Q_HEADS, KV_HEADS, HEAD_DIM)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return states, time.perf_counter() - started


STANDALONE_SMOKE = r'''
import importlib.util

import torch

spec = importlib.util.spec_from_file_location("candidate_standalone", "solution.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
apis = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
for name in apis:
    assert callable(getattr(m, name, None)), name

g = torch.Generator().manual_seed(7)
w = torch.randn(64, 64, generator=g)
ws = torch.ones(64, 4)
acts = [(torch.randn(16, 64, generator=g), torch.ones(16, 4)) for _ in range(2)]
result = m.hif4_calibration_and_quantize_weight(w, ws, acts)
assert set(result) == {"weight_params", "activation_state"}
assert bool(torch.isfinite(m._dequantize_hif4(result["weight_params"])).all())
act_params = m.hif4_dynamic_quantize_activation(*acts[0], result["activation_state"])
assert bool(torch.isfinite(m._dequantize_hif4(act_params)).all())

items = []
for _ in range(3):
    items.append({
        "q": (torch.randn(8, 256, generator=g), torch.ones(8, 16)),
        "k": (torch.randn(8, 128, generator=g), torch.ones(8, 8)),
        "v": (torch.randn(8, 128, generator=g), torch.ones(8, 8)),
    })
states = m.hif4_calibration_attention(items, 4, 2, 64)
for role, heads in (("q", 4), ("k", 2), ("v", 2)):
    x, s = items[0][role]
    params = getattr(m, "hif4_dynamic_quantize_" + role)(
        x, s, heads, 64, states[role + "_state"]
    )
    assert bool(torch.isfinite(m._dequantize_hif4(params)).all())
print("STANDALONE_OK")
'''


def main():
    parent = load(ROOT / "solution.py", "v194_parent")
    candidate_path = HERE / "candidate" / "solution.py"
    candidate = load(candidate_path, "v194_candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "device": str(device),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }

    items = make_calib(device)

    states_parent, parent_seconds = timed_calibrate(parent, items, device)
    states_parent2, _ = timed_calibrate(parent, items, device)
    compare("parent-self", states_parent, states_parent2)
    result["parent_self_determinism"] = "PASS"

    states_candidate, candidate_seconds = timed_calibrate(candidate, items, device)
    compare("parent-vs-candidate", states_parent, states_candidate)
    result["calibration_state_bitwise"] = "PASS"
    result["a2_arm"] = states_candidate["q_state"].get("a2_arm")
    result["gate_loss_identity"] = states_candidate["q_state"].get("a2_gate_loss_identity")
    result["gate_loss_rotation"] = states_candidate["q_state"].get("a2_gate_loss_rotation")

    # Spiky-channel dataset (seed/spike probed to make rotation win the gate):
    # exercises the accept branch (learned_rotation/learned_center writes).
    spiky = make_calib(device, seed=555, spike=10.0)
    spiky_parent, _ = timed_calibrate(parent, spiky, device)
    spiky_candidate, _ = timed_calibrate(candidate, spiky, device)
    compare("parent-vs-candidate-spiky", spiky_parent, spiky_candidate)
    assert spiky_candidate["q_state"].get("a2_arm") == "rotation", (
        "spiky dataset no longer triggers the accept branch; re-probe seed/spike"
    )
    assert "learned_rotation" in spiky_candidate["q_state"]
    result["spiky_calibration_state_bitwise_accept_branch"] = "PASS"
    result["spiky_a2_arm"] = spiky_candidate["q_state"].get("a2_arm")

    # Direct trainer check: same dense windows through both _a2_train_rotation
    # implementations must give bitwise-identical rotation, center, and info.
    dense_windows = [
        {
            "q": parent._dequantize_nvfp4_float32(*item["q"]).to(torch.float32),
            "k": parent._dequantize_nvfp4_float32(*item["k"]).to(torch.float32),
            "v": parent._dequantize_nvfp4_float32(*item["v"]).to(torch.float32),
        }
        for item in items
    ]
    rot_p, info_p, center_p = parent._a2_train_rotation(
        dense_windows, Q_HEADS, KV_HEADS, HEAD_DIM, device
    )
    rot_c, info_c, center_c = candidate._a2_train_rotation(
        dense_windows, Q_HEADS, KV_HEADS, HEAD_DIM, device
    )
    compare("train-rotation", rot_p, rot_c)
    compare("train-center", center_p, center_c)
    compare("train-info", info_p, info_c)
    result["train_rotation_bitwise"] = "PASS"

    # Direct gate check: the two legacy per-arm calls vs one fused call must
    # produce exactly equal floats, on both identity and rotation arms.
    gate_states = base_states()
    gate_window = items[-1]
    legacy_identity = parent._a2_true_path_gate_loss(
        gate_window, Q_HEADS, KV_HEADS, HEAD_DIM, gate_states, None, device
    )
    legacy_rotation = parent._a2_true_path_gate_loss(
        gate_window, Q_HEADS, KV_HEADS, HEAD_DIM, gate_states, rot_p, device, center_p
    )
    fused_parent, fused_candidate = candidate._a2_true_path_gate_loss(
        gate_window, Q_HEADS, KV_HEADS, HEAD_DIM, gate_states, rot_p, device, center_p
    )
    norm = max(fused_parent, 1e-12)
    assert legacy_identity == fused_parent / norm, (
        f"identity arm: {legacy_identity!r} vs {fused_parent / norm!r}"
    )
    assert legacy_rotation == fused_candidate / norm, (
        f"rotation arm: {legacy_rotation!r} vs {fused_candidate / norm!r}"
    )
    result["fused_gate_loss_exact"] = "PASS"

    gate = make_calib(device)[-1]
    for role, heads in (("q", Q_HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
        dense, scale = gate[role]
        quantize = getattr(parent, "hif4_dynamic_quantize_" + role)
        params_parent = quantize(dense, scale, heads, HEAD_DIM, states_parent[role + "_state"])
        quantize_c = getattr(candidate, "hif4_dynamic_quantize_" + role)
        params_candidate = quantize_c(dense, scale, heads, HEAD_DIM, states_candidate[role + "_state"])
        compare(f"dynamic-{role}", params_parent, params_candidate)
        dequant = candidate._dequantize_hif4(params_candidate)
        assert bool(torch.isfinite(dequant).all()), f"dynamic-{role}: non-finite"
        ref.validate_state(states_candidate[role + "_state"])
        ref.validate_hif4_params(params_candidate, dense.shape)
    result["dynamic_output_bitwise"] = "PASS"
    result["state_and_output_legality"] = "PASS"

    for name in APIS:
        assert callable(getattr(candidate, name, None)), name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(candidate_path, tmp_path / "solution.py")
        (tmp_path / "smoke.py").write_text(STANDALONE_SMOKE, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "smoke.py"], cwd=tmp, capture_output=True, text=True,
            timeout=600,
        )
        assert completed.returncode == 0, (
            f"standalone smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
        assert "STANDALONE_OK" in completed.stdout
    result["standalone_six_api_import_and_finite_output"] = "PASS"

    result["calibration_seconds"] = {
        "parent": round(parent_seconds, 4),
        "candidate": round(candidate_seconds, 4),
        "ratio_candidate_over_parent": round(candidate_seconds / parent_seconds, 4),
        "note": "small synthetic input; recorded only, not a gate",
    }

    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
