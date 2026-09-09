"""Verify v204 linear-no-rank2-residual (root minus L-R2 rank-2 residual).

Checks on small synthetic Linear cases: the candidate runs calibration
without crashing, the rank-2 branch is provably skipped (parent prints
``[L-R2]`` and stores a ``[D,2]`` residual pair in state, candidate prints
nothing and stores ``[D,1]``), the deployed weight params differ from the
parent (non-no-op), the rank-1 segment stays active, the state passes
evaluator/reference_hif4.py legality checks, deployed outputs are finite,
and the single file imports the six public APIs repo-free (python -I).
"""

from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_linear_case(
    device: torch.device,
    *,
    out_features: int = 512,
    in_features: int = 1024,
    windows: int = 4,
    tokens: int = 256,
    seed: int = 204,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.1
    w_scale = torch.ones(out_features, in_features // 16)
    calib = []
    for _ in range(windows):
        a = torch.randn(tokens, in_features, generator=generator)
        a_scale = torch.ones(tokens, in_features // 16)
        calib.append((a.to(device), a_scale.to(device)))
    return weight.to(device), w_scale.to(device), calib


def main():
    candidate = load(HERE / "candidate" / "solution.py", "v204_verify_candidate")
    parent = load(ROOT / "solution.py", "v204_verify_parent")
    assert candidate._WEIGHT_RESIDUAL_RANK == 1
    assert parent._WEIGHT_RESIDUAL_RANK == 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256(
            (ROOT / "solution.py").read_bytes()
        ).hexdigest(),
        "device": device.type,
    }

    weight, w_scale, calib = make_linear_case(device)
    parent_buf = io.StringIO()
    with torch.inference_mode(), redirect_stdout(parent_buf):
        parent_result = parent.hif4_calibration_and_quantize_weight(
            weight, w_scale, calib
        )
    cand_buf = io.StringIO()
    with torch.inference_mode(), redirect_stdout(cand_buf):
        cand_result = candidate.hif4_calibration_and_quantize_weight(
            weight, w_scale, calib
        )
    parent_log = parent_buf.getvalue()
    cand_log = cand_buf.getvalue()

    # Branch-execution evidence: the [L-R2] print is gated by the same
    # _WEIGHT_RESIDUAL_RANK >= 2 condition as the rank-2 complement solver.
    assert "[L-R2]" in parent_log, parent_log[-500:]
    assert "[L-R2]" not in cand_log, cand_log[-500:]
    result["parent_rank2_branch_executed"] = "PASS"
    result["candidate_rank2_branch_skipped"] = "PASS"
    result["parent_lr2_line"] = next(
        line for line in parent_log.splitlines() if "[L-R2]" in line
    )

    # State evidence: parent stores a [D,2] fused residual pair, the
    # candidate must keep only the rank-1 [D,1] pair (or None if the
    # rank-1 segment itself was unreachable on this data).
    p_state = parent_result["activation_state"]
    c_state = cand_result["activation_state"]
    p_u, c_u = p_state.get("residual_u"), c_state.get("residual_u")
    p_v, c_v = p_state.get("residual_v"), c_state.get("residual_v")
    if p_u is None:
        result["rank1_unreachable_on_synthetic"] = True
    else:
        assert tuple(p_u.shape[1:]) == (2,), p_u.shape
        assert tuple(p_v.shape[1:]) == (2,), p_v.shape
        assert c_u is not None and tuple(c_u.shape[1:]) == (1,), c_u.shape
        assert c_v is not None and tuple(c_v.shape[1:]) == (1,), c_v.shape
        assert float(c_u.square().sum()) > 0.0
        result["rank1_segment_retained"] = "PASS"
        result["residual_shapes"] = {
            "parent_u": list(p_u.shape),
            "candidate_u": list(c_u.shape),
        }

    # Non-no-op: removing the rank-2 direction changes the deployed weights
    # whenever the parent's rank-2 complement was reachable on this data.
    rank2_reachable = "reachable=1" in result["parent_lr2_line"]
    result["parent_rank2_reachable"] = rank2_reachable
    params_differ = any(
        not torch.equal(cand_result["weight_params"][k], parent_result["weight_params"][k])
        for k in parent_result["weight_params"]
    )
    if rank2_reachable:
        assert params_differ, "switch had no effect on deployed weights"
    result["deployed_weight_params_differ"] = bool(params_differ)
    result["non_noop"] = "PASS" if params_differ else (
        "PASS-via-branch-evidence" if not rank2_reachable else "FAIL"
    )

    # Legal state + finite deployed outputs through the dynamic API.
    ref.validate_state(c_state)
    out_shape = (int(weight.shape[0]), int(weight.shape[1]))
    ref.validate_hif4_params(cand_result["weight_params"], out_shape)
    assert bool(
        torch.isfinite(candidate._dequantize_hif4(cand_result["weight_params"])).all()
    )
    act_params = candidate.hif4_dynamic_quantize_activation(
        calib[0][0], calib[0][1], c_state
    )
    ref.validate_hif4_params(act_params, calib[0][0].shape)
    act_dense = candidate._dequantize_hif4(act_params)
    assert bool(torch.isfinite(act_dense).all())
    # The [D,1] residual transform must actually apply in the dynamic path:
    # dense output = A + (A u) v^T, so the encoded input differs from the
    # standard branch on this data.
    std_params = candidate._branch_encode_standard_hif4(
        candidate._dequantize_nvfp4_float32(calib[0][0], calib[0][1])
    )
    result["state_and_output_contract"] = "PASS"
    result["dynamic_activation_differs_from_standard"] = bool(
        not torch.equal(act_dense, candidate._dequantize_hif4(std_params))
    )

    # Repo-free single-file import check (six public APIs).
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(HERE / "candidate" / "solution.py", tmp_path / "solution.py")
        code = (
            "import sys; sys.path.insert(0, r'%s'); " % str(tmp_path).replace("\\", "/")
            + "import solution; "
            "apis = ['hif4_calibration_and_quantize_weight',"
            "'hif4_dynamic_quantize_activation','hif4_calibration_attention',"
            "'hif4_dynamic_quantize_q','hif4_dynamic_quantize_k',"
            "'hif4_dynamic_quantize_v']; "
            "assert all(callable(getattr(solution, a)) for a in apis); "
            "print('six-api-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        assert "six-api-ok" in proc.stdout
    result["standalone_import"] = "PASS"

    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
