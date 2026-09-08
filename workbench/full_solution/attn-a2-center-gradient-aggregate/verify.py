"""Verify v195 center-gradient aggregation on small synthetic tensors.

Checks:
  (a) the candidate's center update really aggregates every training window:
      a spy on ``_m_attention_backward`` records each window's dk3 per step,
      and an independent Adam replay using the summed gradients must match
      the candidate's returned center, while the same replay using only the
      last window's gradient must match the parent's returned center;
  (b) all produced tensors and deployed outputs are finite;
  (c) candidate and parent produce a different center / final selection on a
      small synthetic calibration scenario.
Plus a repository-independent single-file six-API import check.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)

Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
SIX_APIS = (
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


def make_dense_windows(count: int, tokens: int = 24, seed: int = 1950):
    windows = []
    for index in range(count):
        generator = torch.Generator().manual_seed(seed + index)
        windows.append({
            "q": torch.randn(tokens, Q_HEADS * HEAD_DIM, generator=generator),
            "k": torch.randn(tokens, KV_HEADS * HEAD_DIM, generator=generator),
            "v": torch.randn(tokens, KV_HEADS * HEAD_DIM, generator=generator),
        })
    return windows


def make_calib_qkv_list(count: int, tokens: int = 16, seed: int = 2950):
    items = []
    for index in range(count):
        generator = torch.Generator().manual_seed(seed + index)
        item = {}
        for role, heads in (("q", Q_HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
            width = heads * HEAD_DIM
            quant = torch.randn(tokens, width, generator=generator)
            scale = torch.ones(tokens, width // 16)
            item[role] = (quant, scale)
        items.append(item)
    return items


def train_with_center_spy(module, windows, device):
    """Run ``_a2_train_rotation`` while recording each window's dk3 sum."""
    recorded = []
    original = module._m_attention_backward

    def spy(d_out, q_hat, k_hat, v, q_heads, kv_heads, head_dim):
        d_q, d_k = original(d_out, q_hat, k_hat, v, q_heads, kv_heads, head_dim)
        recorded.append(
            d_k.reshape(d_k.shape[0], kv_heads, head_dim).sum(dim=0).detach().clone()
        )
        return d_q, d_k

    module._m_attention_backward = spy
    try:
        rotation, info, center = module._a2_train_rotation(
            windows, Q_HEADS, KV_HEADS, HEAD_DIM, device
        )
    finally:
        module._m_attention_backward = original
    return rotation, info, center, recorded


def replay_center(recorded, n_windows, steps, lr, mode):
    """Independent Adam replay of the center update.

    mode="all"  -> per-step gradient is the sum over all windows (candidate);
    mode="last" -> per-step gradient is only the last window's (parent bug).
    """
    center = torch.zeros_like(recorded[0])
    exp_avg = torch.zeros_like(center)
    exp_avg_sq = torch.zeros_like(center)
    for step in range(steps):
        chunk = recorded[step * n_windows:(step + 1) * n_windows]
        grad = torch.stack(chunk).sum(dim=0) if mode == "all" else chunk[-1]
        exp_avg = 0.9 * exp_avg + 0.1 * grad
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad.square()
        bias1 = 1 - 0.9 ** (step + 1)
        bias2 = 1 - 0.999 ** (step + 1)
        center = center - lr * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)
    return center


def main():
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
    }
    candidate = load(HERE / "candidate" / "solution.py", "v195_candidate")
    parent = load(ROOT / "solution.py", "v195_parent")

    # Textual sanity: the fix is present in the candidate and absent in parent.
    cand_text = (HERE / "candidate" / "solution.py").read_text(encoding="utf-8")
    assert "grad_center = grad_center + dk3.sum(dim=0)" in cand_text
    assert "grad_center = torch.zeros_like(center)" in cand_text
    parent_text = (ROOT / "solution.py").read_text(encoding="utf-8")
    assert "grad_center = grad_center + dk3.sum(dim=0)" not in parent_text
    result["patch_present_only_in_candidate"] = "PASS"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = make_dense_windows(count=4)
    n_windows = len(windows)

    rot_c, info_c, center_c, rec_c = train_with_center_spy(candidate, windows, device)
    rot_p, info_p, center_p, rec_p = train_with_center_spy(parent, windows, device)
    steps = int(info_c["steps"])
    assert steps == int(info_p["steps"])
    assert len(rec_c) == steps * n_windows and len(rec_p) == steps * n_windows

    # (a) Candidate center == aggregate replay; parent center == last-window
    # replay.  The two replay modes must genuinely disagree on this data.
    lr = float(candidate._A2_TRAIN_LR)
    replay_all_c = replay_center(rec_c, n_windows, steps, lr, "all")
    replay_last_c = replay_center(rec_c, n_windows, steps, lr, "last")
    replay_last_p = replay_center(rec_p, n_windows, steps, lr, "last")
    assert not torch.allclose(replay_all_c, replay_last_c, atol=1e-6, rtol=1e-4), (
        "test is insensitive: per-window center gradients do not differ enough"
    )
    torch.testing.assert_close(
        center_c.to(replay_all_c.device), replay_all_c, atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        center_p.to(replay_last_p.device), replay_last_p, atol=1e-5, rtol=1e-4
    )
    result["center_uses_all_windows_candidate"] = "PASS"
    result["parent_center_is_last_window_only"] = "PASS"

    # (c, trainer level) candidate and parent centers differ.
    assert not torch.allclose(center_c, center_p, atol=1e-6, rtol=1e-4)
    result["candidate_vs_parent_center_differ"] = {
        "status": "PASS",
        "max_abs_diff": float((center_c - center_p).abs().max()),
    }

    # (b) finite trained tensors and bounded orthogonality error.
    assert bool(torch.isfinite(rot_c).all() and torch.isfinite(center_c).all())
    assert bool(torch.isfinite(rot_p).all() and torch.isfinite(center_p).all())
    assert info_c["ortho_error"] <= candidate._A2_ORTHO_TOLERANCE
    assert info_c["final_train_loss"] == info_c["final_train_loss"]
    result["finite_trained_tensors"] = {
        "status": "PASS",
        "candidate_final_train_loss": float(info_c["final_train_loss"]),
        "parent_final_train_loss": float(info_p["final_train_loss"]),
        "ortho_error": float(info_c["ortho_error"]),
    }

    # (b/c, calibration level) full calibration on both, legality + finite
    # deployed outputs, and candidate/parent selections differ.
    calib = make_calib_qkv_list(count=5)
    with torch.inference_mode():
        states_c = candidate.hif4_calibration_attention(calib, Q_HEADS, KV_HEADS, HEAD_DIM)
        states_p = parent.hif4_calibration_attention(calib, Q_HEADS, KV_HEADS, HEAD_DIM)

    audit_c = states_c["k_state"]
    audit_p = states_p["k_state"]
    arm_c = audit_c.get("a2_arm")
    arm_p = audit_p.get("a2_arm")
    center_cal_c = audit_c.get("learned_center")
    center_cal_p = audit_p.get("learned_center")
    differs = arm_c != arm_p
    if center_cal_c is not None and center_cal_p is not None:
        differs = differs or not torch.allclose(center_cal_c, center_cal_p)
    if audit_c.get("a2_train_loss") is not None and audit_p.get("a2_train_loss") is not None:
        differs = differs or audit_c["a2_train_loss"] != audit_p["a2_train_loss"]
    assert differs, "candidate and parent calibration outcomes are identical"
    result["calibration_outcome_differs"] = {
        "status": "PASS",
        "candidate_arm": arm_c,
        "parent_arm": arm_p,
        "candidate_train_loss": audit_c.get("a2_train_loss"),
        "parent_train_loss": audit_p.get("a2_train_loss"),
        "candidate_gate_loss_rotation": audit_c.get("a2_gate_loss_rotation"),
        "parent_gate_loss_rotation": audit_p.get("a2_gate_loss_rotation"),
    }

    for role, heads in (("q", Q_HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
        state = states_c[role + "_state"]
        ref.validate_state(state)
        params = getattr(candidate, "hif4_dynamic_quantize_" + role)(
            *calib[0][role], heads, HEAD_DIM, state
        )
        ref.validate_hif4_params(params, calib[0][role][0].shape)
        assert bool(torch.isfinite(candidate._dequantize_hif4(params)).all())
    result["state_and_output_contract"] = "PASS"

    # Repository-independent single-file import of all six APIs.
    with tempfile.TemporaryDirectory() as tmp:
        solo = Path(tmp) / "solution.py"
        shutil.copyfile(HERE / "candidate" / "solution.py", solo)
        spec = importlib.util.spec_from_file_location("v195_standalone", solo)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in SIX_APIS:
            assert callable(getattr(module, name, None)), f"missing API: {name}"
    result["standalone_six_api_import"] = "PASS"

    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
