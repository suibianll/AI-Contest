"""Focused v216 standalone-import and fixed-order behavior check."""

from pathlib import Path
import importlib.util
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_params(params: dict[str, torch.Tensor], rows: int, channels: int) -> None:
    expected = {
        "scale_factor": (rows, channels // 64, 1, 1, 1),
        "scale_lv2": (rows, channels // 64, 8, 1, 1),
        "scale_lv3": (rows, channels // 64, 8, 2, 1),
        "sign": (rows, channels // 64, 8, 2, 4),
        "mant": (rows, channels // 64, 8, 2, 4),
    }
    if set(params) != set(expected):
        raise AssertionError(f"unexpected HiF4 keys: {set(params)}")
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(params[key].shape)} != {shape}")


def main() -> None:
    candidate = load_solution(HERE / "candidate" / "solution.py", "v216_candidate")
    required = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    )
    missing = [name for name in required if not hasattr(candidate, name)]
    if missing:
        raise AssertionError(f"missing public API: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(216)
    weight_quant = torch.randn((64, 128), generator=generator, device=device)
    weight_scale = torch.ones((64, 8), dtype=torch.float32, device=device)
    activation_quant = torch.randn((12, 128), generator=generator, device=device)
    activation_scale = torch.ones((12, 8), dtype=torch.float32, device=device)
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant,
        weight_scale,
        [(activation_quant, activation_scale)],
    )
    if set(result) != {"weight_params", "activation_state"}:
        raise AssertionError(f"unexpected calibration result keys: {set(result)}")
    assert_params(result["weight_params"], 64, 128)
    state = result["activation_state"]
    order = state.get("gptq_block_order")
    if not torch.is_tensor(order) or tuple(order.shape) != (2,):
        raise AssertionError("v216 did not receive the compiled block order")

    observed = candidate.hif4_dynamic_quantize_activation(
        activation_quant, activation_scale, state
    )
    assert_params(observed, 12, 128)
    dense = candidate._static_actorder_dense_from_state(
        activation_quant, activation_scale, state
    )
    expected = candidate._combined_dynamic_fast_reordered_gptq(
        dense, state, order
    )
    if expected is None:
        raise AssertionError("compiled order path was not reachable")
    for key in expected:
        if not torch.equal(observed[key], expected[key]):
            raise AssertionError(f"v216 fixed-order output mismatch for {key}")

    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "v216_reference")
    reference.validate_state(state)
    reference.validate_hif4_params(result["weight_params"], (64, 128))
    reference.validate_hif4_params(observed, (12, 128))
    print(
        f"v216 verify passed: device={device.type} order={order.detach().cpu().tolist()} "
        f"outputs={tuple(observed['mant'].shape)}"
    )


if __name__ == "__main__":
    main()

