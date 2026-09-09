"""Focused v211 legality and standalone-import check."""

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
    candidate = load_solution(HERE / "candidate" / "solution.py", "v211_candidate")
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
    generator = torch.Generator(device=device).manual_seed(211)
    weight_quant = torch.randn((64, 128), generator=generator, device=device)
    weight_scale = torch.ones((64, 8), dtype=torch.float32, device=device)
    activations = [
        (
            torch.randn((12, 128), generator=generator, device=device),
            torch.ones((12, 8), dtype=torch.float32, device=device),
        ),
        (
            torch.randn((10, 128), generator=generator, device=device),
            torch.ones((10, 8), dtype=torch.float32, device=device),
        ),
    ]
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    if set(result) != {"weight_params", "activation_state"}:
        raise AssertionError(f"unexpected calibration result keys: {set(result)}")
    assert_params(result["weight_params"], 64, 128)
    state = result["activation_state"]
    if not isinstance(state, dict) or state.get("v211_attempted") != 1:
        raise AssertionError("v211 did not reach its fixed output-code fit")
    if state.get("v211_coordinate") != "deployment-output-joint-4-code-group":
        raise AssertionError("v211 coordinate provenance is missing")
    if int(state.get("v211_group_size", 0)) != 4:
        raise AssertionError("v211 group size is not the fixed 4-element configuration")
    if int(state.get("v211_fit_rows", 0)) != 22:
        raise AssertionError("v211 did not consume every supplied calibration row")

    activation_params = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], state
    )
    assert_params(activation_params, 12, 128)

    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "v211_reference")
    reference.validate_state(state)
    reference.validate_hif4_params(result["weight_params"], (64, 128))
    reference.validate_hif4_params(activation_params, (12, 128))
    print(
        f"v211 verify passed: device={device.type} arm={state.get('v211_arm')} "
        f"fit_rows={state.get('v211_fit_rows')} "
        f"accepted={state.get('v211_accepted')} "
        f"pairs={state.get('v211_accepted_pairs')} "
        f"loss_parent={state.get('v211_loss_parent')} "
        f"loss_candidate={state.get('v211_loss_candidate')}"
    )


if __name__ == "__main__":
    main()
