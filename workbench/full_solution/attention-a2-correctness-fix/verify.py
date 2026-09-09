"""Focused static and small-tensor verification for the A2 correctness fix."""

from pathlib import Path
import ast
import importlib.util
import json
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidate" / "solution.py"


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latest_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if not matches:
        raise AssertionError(f"missing function {name}")
    return matches[-1]


def main() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calibration = latest_function(tree, "hif4_calibration_attention")
    broad_handlers = [
        node
        for node in ast.walk(calibration)
        if isinstance(node, ast.ExceptHandler) and node.type is not None
        and isinstance(node.type, ast.Name) and node.type.id == "Exception"
    ]
    if broad_handlers:
        raise AssertionError("latest Attention calibration still swallows Exception")
    if "grad_theta = grad_theta / window_count" not in source:
        raise AssertionError("theta gradient is not averaged over windows")
    if "grad_center = grad_center / window_count" not in source:
        raise AssertionError("center gradient is not averaged over windows")

    candidate = load_solution(CANDIDATE, "attention_a2_correctness_fix")
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
    generator = torch.Generator(device=device).manual_seed(222)
    q_quant = torch.randn((20, 128), generator=generator, device=device)
    q_scale = torch.ones((20, 8), dtype=torch.float32, device=device)
    k_quant = torch.randn((20, 64), generator=generator, device=device)
    k_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    v_quant = torch.randn((20, 64), generator=generator, device=device)
    v_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    windows = [
        {"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)},
        {"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)},
    ]
    states = candidate.hif4_calibration_attention(windows, 2, 1, 64)
    if set(states) != {"q_state", "k_state", "v_state"}:
        raise AssertionError(f"unexpected Attention state keys: {set(states)}")
    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "a2_fix_reference")
    reference.validate_state(states)
    if states["q_state"].get("a2_arm") not in {"identity", "rotation"}:
        raise AssertionError(f"unexpected A2 arm: {states['q_state'].get('a2_arm')}")

    original_parent = candidate._V189_CALIBRATION_ATTENTION
    try:
        def raise_sentinel(*_args, **_kwargs):
            raise RuntimeError("a2-correctness-fix-sentinel")

        candidate._V189_CALIBRATION_ATTENTION = raise_sentinel
        try:
            candidate.hif4_calibration_attention([], 2, 1, 64)
        except RuntimeError as error:
            if str(error) != "a2-correctness-fix-sentinel":
                raise
        else:
            raise AssertionError("parent calibration error was silently swallowed")
    finally:
        candidate._V189_CALIBRATION_ATTENTION = original_parent

    report = {
        "status": "passed",
        "device": device.type,
        "a2_arm": states["q_state"].get("a2_arm"),
        "checks": {
            "six_api_import": True,
            "legal_attention_state": True,
            "mean_theta_gradient": True,
            "mean_center_gradient": True,
            "broad_exception_removed": True,
            "exception_propagation": True,
        },
    }
    (HERE / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"A2 correctness-fix verify passed: device={device.type} "
        f"arm={states['q_state'].get('a2_arm')}"
    )


if __name__ == "__main__":
    main()
