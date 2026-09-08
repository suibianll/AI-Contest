"""Focused v199 hard-boundary and deployment verification."""

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
        generator = torch.Generator(device=device).manual_seed(19900 + index)
        q = torch.randn(
            TOKENS, Q_HEADS * HEAD_DIM, device=device, generator=generator
        ) * 0.02
        k = torch.randn(
            TOKENS, KV_HEADS * HEAD_DIM, device=device, generator=generator
        ) * 0.02
        v = torch.randn(
            TOKENS, KV_HEADS * HEAD_DIM, device=device, generator=generator
        )
        # Put reciprocal range pressure in different coordinates of the same
        # 64-block.  This makes the first hard boundary reachable without
        # providing the selector an oracle.
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


def capture_dense(module, role, window, state):
    heads = Q_HEADS if role == "q" else KV_HEADS
    api = getattr(module, "hif4_dynamic_quantize_" + role)
    captured = []
    original = module._dense_to_hif4

    def spy(dense, **kwargs):
        if not captured:
            captured.append(dense.detach().clone())
        return original(dense, **kwargs)

    module._dense_to_hif4 = spy
    try:
        api(*window[role], heads, HEAD_DIM, state)
    finally:
        module._dense_to_hif4 = original
    return captured[0]


def main():
    module = load(HERE / "candidate" / "solution.py", "v199_verify")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = make_windows(device)
    original_parent = module._V199_PARENT_CALIBRATION
    module._V199_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    try:
        with torch.inference_mode():
            states = module.hif4_calibration_attention(
                windows, Q_HEADS, KV_HEADS, HEAD_DIM
            )
    finally:
        module._V199_PARENT_CALIBRATION = original_parent

    audit = states["q_state"]
    assert audit["v199_attempted"] > 0, audit
    assert audit["v199_fit_windows"] == WINDOWS - 1, audit
    assert audit["v199_selected_blocks"] <= KV_HEADS, audit
    assert audit["v199_arm"] in {"accepted", "parent", "no_boundary"}, audit

    # The implementation must expose a real reciprocal state whenever it
    # accepts, and must leave V untouched in every case.
    assert "diag_scale" not in states["v_state"]
    if audit["v199_arm"] == "accepted":
        assert audit["v199_accepted"] == 1
        assert "diag_scale" in states["q_state"]
        assert "diag_scale" in states["k_state"]
        assert "v199_u" in states["q_state"]
        assert "v199_u" in states["k_state"]
        assert torch.count_nonzero(states["q_state"]["v199_u"]) > 0

    parent = base_states()
    q_dense = capture_dense(module, "q", windows[0], parent["q_state"])
    k_dense = capture_dense(module, "k", windows[0], parent["k_state"])
    q_params = module.hif4_dynamic_quantize_q(
        *windows[0]["q"], Q_HEADS, HEAD_DIM, states["q_state"]
    )
    k_params = module.hif4_dynamic_quantize_k(
        *windows[0]["k"], KV_HEADS, HEAD_DIM, states["k_state"]
    )
    ref.validate_state(states["q_state"])
    ref.validate_state(states["k_state"])
    ref.validate_state(states["v_state"])
    ref.validate_hif4_params(q_params, windows[0]["q"][0].shape)
    ref.validate_hif4_params(k_params, windows[0]["k"][0].shape)
    assert bool(torch.isfinite(module._dequantize_hif4(q_params)).all())
    assert bool(torch.isfinite(module._dequantize_hif4(k_params)).all())

    # Apply a manually selected coordinate and verify the continuous QK
    # product is invariant after the parent final transforms.
    u = torch.zeros(KV_HEADS, 1, device=device)
    u[0, 0] = 0.31
    q_factor, k_factor = module._v199_factor_vectors(
        u, Q_HEADS, KV_HEADS, HEAD_DIM, device
    )
    q_scaled = q_dense * q_factor.reshape(1, -1)
    k_scaled = k_dense * k_factor.reshape(1, -1)
    q_grouped = q_dense.reshape(TOKENS, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM)
    q_scaled_grouped = q_scaled.reshape(
        TOKENS, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM
    )
    k_grouped = k_dense.reshape(TOKENS, KV_HEADS, HEAD_DIM)
    k_scaled_grouped = k_scaled.reshape(TOKENS, KV_HEADS, HEAD_DIM)
    lhs = torch.einsum("tgjd,tgd->gjt", q_grouped, k_grouped)
    rhs = torch.einsum("tgjd,tgd->gjt", q_scaled_grouped, k_scaled_grouped)
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-6)
    assert float((lhs - rhs).abs().max()) < 2.0e-5

    # diag_scale == ones is exactly the parent deployment path.
    ones_q = dict(parent["q_state"], diag_scale=torch.ones(Q_HEADS * HEAD_DIM))
    ones_k = dict(parent["k_state"], diag_scale=torch.ones(KV_HEADS * HEAD_DIM))
    for role, heads, state, ones_state in (
        ("q", Q_HEADS, parent["q_state"], ones_q),
        ("k", KV_HEADS, parent["k_state"], ones_k),
    ):
        api = getattr(module, "hif4_dynamic_quantize_" + role)
        a = api(*windows[0][role], heads, HEAD_DIM, state)
        b = api(*windows[0][role], heads, HEAD_DIM, ones_state)
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            assert torch.equal(a[key], b[key]), (role, key)

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
        "v199_arm": audit["v199_arm"],
        "v199_attempted": audit["v199_attempted"],
        "v199_selected_blocks": audit["v199_selected_blocks"],
        "v199_u_nonzero": audit["v199_u_nonzero"],
        "v199_fit_parent_mse": audit.get("v199_fit_parent_mse"),
        "v199_fit_candidate_mse": audit.get("v199_fit_candidate_mse"),
        "v199_holdout_parent_mse": audit.get("v199_holdout_parent_mse"),
        "v199_holdout_candidate_mse": audit.get("v199_holdout_candidate_mse"),
        "reciprocal_identity_max_abs": float((lhs - rhs).abs().max()),
        "state_and_output_contract": "PASS",
        "isolated_import": "PASS",
    }
    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
