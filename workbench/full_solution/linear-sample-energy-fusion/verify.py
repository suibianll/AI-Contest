"""Verify v202 order/parameter equivalence and standalone import."""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
torch.set_num_threads(1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same(left, right, path="state"):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), (path, set(left) ^ set(right))
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
        return
    assert left == right, (path, left, right)


def make_inputs(device: torch.device):
    generator = torch.Generator(device=device).manual_seed(20209)
    out_features = 128
    channels = 128
    weight_quant = torch.randn(
        out_features, channels, device=device, generator=generator
    ) * 0.4
    weight_scale = torch.ones(
        out_features, channels // 16, device=device
    )
    calibration = []
    for rows in (24, 31, 19):
        activation_quant = torch.randn(
            rows, channels, device=device, generator=generator
        ) * 0.35
        activation_scale = torch.ones(rows, channels // 16, device=device)
        calibration.append((activation_quant, activation_scale))
    dynamic_quant = torch.randn(
        17, channels, device=device, generator=generator
    ) * 0.35
    dynamic_scale = torch.ones(17, channels // 16, device=device)
    return (
        weight_quant,
        weight_scale,
        calibration,
        dynamic_quant,
        dynamic_scale,
    )


def main():
    root = load(ROOT / "solution.py", "v202_root")
    candidate = load(HERE / "candidate" / "solution.py", "v202_candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        weight_quant,
        weight_scale,
        calibration,
        dynamic_quant,
        dynamic_scale,
    ) = make_inputs(device)

    with torch.inference_mode():
        parent_result = root.hif4_calibration_and_quantize_weight(
            weight_quant, weight_scale, calibration
        )
        candidate_result = candidate.hif4_calibration_and_quantize_weight(
            weight_quant, weight_scale, calibration
        )

    assert_same(
        parent_result["weight_params"],
        candidate_result["weight_params"],
        "weight_params",
    )
    parent_state = parent_result["activation_state"]
    candidate_state = candidate_result["activation_state"]
    assert torch.equal(
        parent_state["gptq_block_order"], candidate_state["gptq_block_order"]
    )
    for key, value in parent_state.items():
        assert key in candidate_state, key
        if key != "gptq_block_order":
            assert_same(value, candidate_state[key], f"activation_state.{key}")
    assert torch.is_tensor(candidate_state["compiled_sample_energy_order"])
    assert torch.equal(
        candidate_state["compiled_sample_energy_order"],
        candidate_state["gptq_block_order"],
    )

    # The fused state must make the post-calibration rebuild unreachable.
    def fail_rebuild(*args, **kwargs):
        raise AssertionError("post-calibration sample-energy rebuild was called")

    candidate._combined_sample_energy_block_order = fail_rebuild
    with torch.inference_mode():
        parent_dynamic = root.hif4_dynamic_quantize_activation(
            dynamic_quant, dynamic_scale, parent_state
        )
        candidate_dynamic = candidate.hif4_dynamic_quantize_activation(
            dynamic_quant, dynamic_scale, candidate_state
        )
    assert_same(parent_dynamic, candidate_dynamic, "dynamic_activation")

    isolated = HERE / "isolated"
    isolated.mkdir(exist_ok=True)
    isolated_file = isolated / "solution.py"
    isolated_file.write_bytes((HERE / "candidate" / "solution.py").read_bytes())
    code = (
        "import importlib.util,sys\n"
        "s=importlib.util.spec_from_file_location('x',sys.argv[1])\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        "for n in ('hif4_calibration_and_quantize_weight',"
        "'hif4_dynamic_quantize_activation','hif4_calibration_attention',"
        "'hif4_dynamic_quantize_q','hif4_dynamic_quantize_k',"
        "'hif4_dynamic_quantize_v'): assert callable(getattr(m,n,None)),n\n"
        "print('isolated import OK')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(isolated_file)],
        cwd=isolated,
        check=True,
        capture_output=True,
        text=True,
    )

    result = {
        "device": str(device),
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256(
            (ROOT / "solution.py").read_bytes()
        ).hexdigest(),
        "weight_params_bitwise_equal": True,
        "activation_state_bitwise_equal": True,
        "compiled_order_equals_parent_order": True,
        "post_calibration_rebuild_called": False,
        "dynamic_activation_bitwise_equal": True,
        "isolated_import": "PASS",
    }
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
