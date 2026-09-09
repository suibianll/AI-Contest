"""Diagnostic probe: which activation mantissa path does deployment use?"""

from pathlib import Path
import importlib.util
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    candidate = load_solution(HERE / "candidate" / "solution.py", "probe_candidate")
    parent = load_solution(ROOT / "solution.py", "probe_parent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(227)
    rows, channels = 64, 128
    weight_quant = torch.randn((rows, channels), generator=generator, device=device)
    weight_scale = torch.ones((rows, 8), dtype=torch.float32, device=device)
    activations = [
        (
            torch.randn((12, channels), generator=generator, device=device),
            torch.ones((12, 8), dtype=torch.float32, device=device),
        ),
        (
            torch.randn((10, channels), generator=generator, device=device),
            torch.ones((10, 8), dtype=torch.float32, device=device),
        ),
    ]
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    state = result["activation_state"]
    print("state keys:", sorted(state))
    for key in ("gram", "h_inv", "importance", "gptq_block_order", "max_refine_ratio"):
        value = state.get(key)
        if torch.is_tensor(value):
            print(f"  {key}: tensor {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key}: {value!r}")

    calls = {"boundary": 0, "boundary_with_table": 0, "adaround": 0}
    original_boundary = candidate._jrb1_boundary_mantissa
    original_adaround = candidate._adaround_mantissa

    def wrapped_boundary(x_abs, scale, boundaries):
        calls["boundary"] += 1
        if boundaries is not None:
            calls["boundary_with_table"] += 1
        return original_boundary(x_abs, scale, boundaries)

    def wrapped_adaround(*args, **kwargs):
        calls["adaround"] += 1
        return original_adaround(*args, **kwargs)

    candidate._jrb1_boundary_mantissa = wrapped_boundary
    candidate._adaround_mantissa = wrapped_adaround

    probe_state = dict(state)
    probe_state["activation_boundaries"] = torch.full(
        (7,), 0.5, device=device, dtype=torch.float32
    )
    probe_state["activation_boundaries"][3] = 0.25
    parent_params = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], state
    )
    probe_params = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], probe_state
    )
    codes = lambda p: torch.round(p["mant"].to(torch.float32) / 0.25)
    changed = int((codes(probe_params) != codes(parent_params)).sum())
    print("probe call counts:", calls)
    print("probe changed mantissa codes:", changed)
    print("gram in state:", state.get("gram") is not None)


if __name__ == "__main__":
    main()
