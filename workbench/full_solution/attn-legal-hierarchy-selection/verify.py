"""Focused A4 legal-hierarchy and standalone contract checks."""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
TOKENS = 8
WINDOWS = 5


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_states():
    common = {
        "multiplier": None,
        "permutation": None,
        "importance": None,
        "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
        "error_threshold": 1.0e-7,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "version": 2,
    }
    return {
        "q_state": dict(common, num_heads=Q_HEADS, head_dim=HEAD_DIM),
        "k_state": dict(
            common,
            center_mode=0,
            num_heads=KV_HEADS,
            head_dim=HEAD_DIM,
        ),
        "v_state": dict(common, num_heads=KV_HEADS, head_dim=HEAD_DIM),
    }


def make_windows(device: torch.device):
    windows = []
    for index in range(WINDOWS):
        generator = torch.Generator(device=device).manual_seed(20300 + index)
        q = torch.randn(
            TOKENS, Q_HEADS * HEAD_DIM, device=device, generator=generator
        ) * 0.02
        k = torch.randn(
            TOKENS, KV_HEADS * HEAD_DIM, device=device, generator=generator
        ) * 0.02
        v = torch.randn(
            TOKENS, KV_HEADS * HEAD_DIM, device=device, generator=generator
        )
        q[:, 3::HEAD_DIM] = torch.randn(
            TOKENS, Q_HEADS, device=device, generator=generator
        ) * 8.0
        k[:, 27::HEAD_DIM] = torch.randn(
            TOKENS, KV_HEADS, device=device, generator=generator
        ) * 8.0
        windows.append(
            {
                "q": (
                    q,
                    torch.ones(
                        TOKENS, Q_HEADS * HEAD_DIM // 16, device=device
                    ),
                ),
                "k": (
                    k,
                    torch.ones(
                        TOKENS, KV_HEADS * HEAD_DIM // 16, device=device
                    ),
                ),
                "v": (
                    v,
                    torch.ones(
                        TOKENS, KV_HEADS * HEAD_DIM // 16, device=device
                    ),
                ),
            }
        )
    return windows


def main():
    module = load(HERE / "candidate" / "solution.py", "v203_verify")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = make_windows(device)
    original_parent = module._A4_PARENT_CALIBRATION
    module._A4_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    try:
        with torch.inference_mode():
            states = module.hif4_calibration_attention(
                windows, Q_HEADS, KV_HEADS, HEAD_DIM
            )
    finally:
        module._A4_PARENT_CALIBRATION = original_parent

    audit = states["q_state"]
    assert audit["a4_attempted"] > 0, audit
    assert audit["a4_ranked"] == audit["a4_attempted"], audit
    assert audit["a4_fit_windows"] == WINDOWS - 1, audit
    assert audit["a4_selected_groups"] <= KV_HEADS, audit
    assert audit["a4_arm"] in {"accepted", "parent", "no_boundary"}, audit
    assert "a4_scale_offsets" not in states["v_state"]
    if audit["a4_arm"] == "accepted":
        assert audit["a4_accepted"] == 1
        q_offsets = states["q_state"]["a4_scale_offsets"]
        k_offsets = states["k_state"]["a4_scale_offsets"]
        assert bool(torch.all((q_offsets >= -1) & (q_offsets <= 1)))
        assert bool(torch.all((k_offsets >= -1) & (k_offsets <= 1)))
        assert torch.count_nonzero(q_offsets) + torch.count_nonzero(k_offsets) > 0

    parent = base_states()
    q_params = module.hif4_dynamic_quantize_q(
        *windows[0]["q"], Q_HEADS, HEAD_DIM, parent["q_state"]
    )
    zero_offsets = torch.zeros(Q_HEADS * HEAD_DIM // 64, dtype=torch.int8)
    zero_reencoded = module._a4_apply_hierarchy_offsets(
        module._a4_parent_dense(
            module._dequantize_nvfp4_float32(*windows[0]["q"]),
            parent["q_state"],
            Q_HEADS,
            HEAD_DIM,
            is_k=False,
        ),
        q_params,
        zero_offsets,
        parent["q_state"]["importance"],
    )
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        assert torch.equal(q_params[key], zero_reencoded[key]), key

    for role, heads, state, window in (
        ("q", Q_HEADS, states["q_state"], windows[0]["q"]),
        ("k", KV_HEADS, states["k_state"], windows[0]["k"]),
    ):
        params = getattr(module, "hif4_dynamic_quantize_" + role)(
            *window, heads, HEAD_DIM, state
        )
        ref.validate_state(state)
        ref.validate_hif4_params(params, window[0].shape)
        assert bool(torch.isfinite(module._dequantize_hif4(params)).all())

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
        "a4_arm": audit["a4_arm"],
        "a4_attempted": audit["a4_attempted"],
        "a4_ranked": audit["a4_ranked"],
        "a4_selected_groups": audit["a4_selected_groups"],
        "a4_selected_blocks": audit["a4_selected_blocks"],
        "a4_fit_parent_mse": audit.get("a4_fit_parent_mse"),
        "a4_fit_candidate_mse": audit.get("a4_fit_candidate_mse"),
        "a4_holdout_parent_mse": audit.get("a4_holdout_parent_mse"),
        "a4_holdout_candidate_mse": audit.get("a4_holdout_candidate_mse"),
        "zero_offset_reencode_bitwise_equal": True,
        "state_and_output_contract": "PASS",
        "isolated_import": "PASS",
    }
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
