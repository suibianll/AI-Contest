"""Coordinate-consistent error decomposition for plan P1 (standalone CLI).

Reconstructs the pre-quantization continuous tensors (``Q_t/K_t/V_t`` and the
Linear ``X_t/W_t``) of a deployed solution under its *own* coordinate system,
then measures per-operand quantization effects with 2^3 Attention arms and a
four-arm Linear decomposition.  The standard/reference outputs are computed
exactly as in ``proxy_v3_eval._score`` so that arm 111 reproduces the player
gain already recorded for the same SHA/case.

Design constraints from the active plan:
- never modify ``official_eval.py``, ``reference_hif4.py``, or the root
  ``solution.py``;
- default scoring paths and call graphs stay untouched (this module is not
  imported by the normal evaluator entry);
- diagnostic timings are never fed to the official time model;
- unrepresentable/unsupported transforms raise ``UNSUPPORTED_DIAGNOSTIC``
  instead of silently skipping attribution.

Usage::

  .venv\\Scripts\\python.exe evaluator\\coordinate_diagnostics.py ^
    --solution solution.py --scenario both --shards 0 ^
    --cache artifacts\\official_eval\\cache\\qwen2.5-0.5b-proxy-v2.pt ^
    --output-dir artifacts\\proxy_v3\\coordinate-diagnostics-v186\\run-001
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    import official_eval as v2
    import proxy_v3_eval as v3
except ModuleNotFoundError:  # pragma: no cover - package import path
    from . import official_eval as v2
    from . import proxy_v3_eval as v3

_OUTPUT_NAMES = ("000", "100", "010", "001", "110", "101", "011", "111")

_UNSUPPORTED = "UNSUPPORTED_DIAGNOSTIC"


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) * (a - b)).mean())


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(f"{_UNSUPPORTED}: {message}")


# --------------------------------------------------------------------------- #
# Linear continuous-prefix mirrors (read-only, follow solution.py exactly)
# --------------------------------------------------------------------------- #
def linear_weight_continuous(
    weight_raw: torch.Tensor,
    state: Mapping[str, Any],
    sol: Any,
) -> torch.Tensor:
    """W_t: the pre-quantization weight in the deployed coordinate.

    Mirrors solution.py L8149-8156 (pair transform, weight side) followed by
    the rank-r residual subtraction L8405.  smooth_inv == 1/best_d.
    """
    smooth_inv = state.get("smooth_inv")
    if smooth_inv is None:
        d = torch.ones(
            weight_raw.shape[-1], dtype=torch.float32, device=weight_raw.device
        )
    else:
        d = smooth_inv.reciprocal().to(
            device=weight_raw.device, dtype=torch.float32
        )
    perm = state.get("permutation")
    if perm is None:
        perm = torch.arange(
            weight_raw.shape[-1], dtype=torch.int64, device=weight_raw.device
        )
    else:
        perm = perm.detach().to(
            device=weight_raw.device, dtype=torch.int64
        ).reshape(-1)
    w = sol._linear_pair_transform(
        weight_raw,
        d,
        perm,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        weight_side=True,
    )
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=w.device, dtype=torch.float32)
        v = residual_v.to(device=w.device, dtype=torch.float32)
        w = w - (w @ v) @ u.transpose(0, 1)
    elif state.get("rank1_u") is not None and state.get("rank1_v") is not None:
        u = state["rank1_u"].to(device=w.device, dtype=torch.float32)
        v = state["rank1_v"].to(device=w.device, dtype=torch.float32)
        w = w - (w @ v).unsqueeze(-1) * u
    return w.to(torch.float32)


def linear_activation_continuous(
    activation_raw: torch.Tensor,
    state: Mapping[str, Any],
    sol: Any,
) -> torch.Tensor:
    """X_t: the pre-quantization activation in the deployed coordinate.

    Mirrors solution.py L8741-8775 (smooth_inv -> permutation -> block
    hadamard -> rank residual).
    """
    x = activation_raw.to(torch.float32)
    if state.get("smooth_inv") is not None:
        scale = sol._safe_positive_vector(
            state["smooth_inv"], int(x.shape[-1])
        ).to(x.device)
        x = x * scale.reshape(1, -1)
    perm = state.get("permutation")
    if perm is not None:
        order = perm.detach().to(device=x.device, dtype=torch.int64).reshape(-1)
        x = x.index_select(-1, order)
    block_smooth_size = int(state.get("block_smooth_size", 0))
    if block_smooth_size != 0:
        x = sol._block_hadamard_transform(
            x, block_smooth_size, int(state.get("block_smooth_seed", 0))
        )
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=x.device, dtype=torch.float32)
        v = residual_v.to(device=x.device, dtype=torch.float32)
        x = x + (x @ u) @ v.transpose(0, 1)
    elif state.get("rank1_u") is not None and state.get("rank1_v") is not None:
        u = state["rank1_u"].to(device=x.device, dtype=torch.float32)
        v = state["rank1_v"].to(device=x.device, dtype=torch.float32)
        x = x + (x @ u).unsqueeze(-1) * v
    return x.to(torch.float32)


# --------------------------------------------------------------------------- #
# Attention arms
# --------------------------------------------------------------------------- #
def attention_arms(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    ref: torch.Tensor,
) -> dict[str, Any]:
    """Eight fixed-coordinate arms (float vs HiF4 per operand)."""
    device = q_t.device
    q_t, k_t, v_t = q_t.to(device), k_t.to(device), v_t.to(device)
    q_h, k_h, v_h = q_h.to(device), k_h.to(device), v_h.to(device)
    ref = ref.to(device)
    qkv_t = (q_t, k_t, v_t)
    qkv_h = (q_h, k_h, v_h)
    outputs: dict[str, torch.Tensor] = {}
    mse_to_ref: dict[str, float] = {}
    mse_to_t: dict[str, float] = {}
    o_t: torch.Tensor | None = None
    for arm in _OUTPUT_NAMES:
        use_h = [int(c) for c in arm]
        tensors = [
            (qkv_h[0] if use_h[0] else qkv_t[0]),
            (qkv_h[1] if use_h[1] else qkv_t[1]),
            (qkv_h[2] if use_h[2] else qkv_t[2]),
        ]
        out = v2._attention(
            *(value[None] for value in tensors), q_heads, kv_heads, head_dim
        )
        outputs[arm] = out
        mse_to_ref[arm] = _mse(out, ref)
        if arm == "000":
            o_t = out
        mse_to_t[arm] = _mse(out, o_t) if o_t is not None else 0.0
    o_h = outputs["111"]
    b = outputs["000"] - ref
    e = o_h - outputs["000"]
    be = {
        "mean_B2": float((b * b).mean()),
        "mean_E2": float((e * e).mean()),
        "mean_BE": float((b * e).mean()),
        "lhs_mse_oh_ref": _mse(o_h, ref),
        "rhs_sum": float((b * b).mean() + (e * e).mean() + 2.0 * (b * e).mean()),
    }
    return {
        "ref_energy": float((ref * ref).mean()),
        "mse_to_ref": mse_to_ref,
        "mse_to_t": mse_to_t,
        "mse_o_t_vs_ref": mse_to_ref["000"],
        "mse_player_vs_ref": mse_to_ref["111"],
        "mse_player_vs_t": mse_to_t["111"],
        "be_decomposition": be,
    }


# --------------------------------------------------------------------------- #
# Linear arms
# --------------------------------------------------------------------------- #
def linear_arms(
    x_t: torch.Tensor,
    w_t: torch.Tensor,
    x_h: torch.Tensor,
    w_h: torch.Tensor,
    ref: torch.Tensor,
) -> dict[str, Any]:
    """Four-arm X/W decomposition with the signed residual expansion."""
    device = x_t.device
    x_t = x_t.to(device)
    w_t = w_t.to(device)
    x_h = x_h.to(device)
    w_h = w_h.to(device)
    ref = ref.to(device)
    y00 = x_t @ w_t.t()
    y10 = x_h @ w_t.t()
    y01 = x_t @ w_h.t()
    y11 = x_h @ w_h.t()
    mse_to_ref = {
        "X_tW_t": _mse(y00, ref),
        "X_hW_t": _mse(y10, ref),
        "X_tW_h": _mse(y01, ref),
        "X_hW_h": _mse(y11, ref),
    }
    e_x = x_h - x_t
    e_w = w_h - w_t
    t1 = e_x @ w_t.t()
    t2 = x_t @ e_w.t()
    t3 = e_x @ e_w.t()
    d = y11 - y00
    terms = {
        "ms_T1": float((t1 * t1).mean()),
        "ms_T2": float((t2 * t2).mean()),
        "ms_T3": float((t3 * t3).mean()),
        "cross_12": float((t1 * t2).mean()),
        "cross_13": float((t1 * t3).mean()),
        "cross_23": float((t2 * t3).mean()),
        "lhs_ms_Yhh_minus_Ytt": float((d * d).mean()),
        "rhs_exact": float(((t1 + t2 + t3) * (t1 + t2 + t3)).mean()),
        "ms_E_X": float((e_x * e_x).mean()),
        "ms_E_W": float((e_w * e_w).mean()),
    }
    return {
        "ref_energy": float((ref * ref).mean()),
        "mse_to_ref": mse_to_ref,
        "player_mse": mse_to_ref["X_hW_h"],
        "expansion": terms,
    }


# --------------------------------------------------------------------------- #
# Case drivers
# --------------------------------------------------------------------------- #
def run_attention_cases(
    sol: Any,
    pack: v2.PreparedPack,
    attention_states: Mapping[int, Mapping[str, Any]],
    device: torch.device,
    details_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case, seed in zip(pack.attention_cases, details_seed):
        states = attention_states[case.layer]
        pairs = pack.test_qkv[case.test_window][case.layer]
        q_pair, k_pair, value_pair = (v2._move_pair(pair, device) for pair in pairs)
        q_state, k_state, v_state = states["q_state"], states["k_state"], states["v_state"]
        # continuous (deployed coordinate)
        q_raw = v2.dequantize_nvfp4(*q_pair).to(torch.float32)
        k_raw = v2.dequantize_nvfp4(*k_pair).to(torch.float32)
        v_raw = v2.dequantize_nvfp4(*value_pair).to(torch.float32)
        q_t = sol._attention_state_transform_dense(
            q_raw, q_state, pack.q_heads, pack.head_dim, is_k=False
        ).to(torch.float32)
        k_t = sol._attention_state_transform_dense(
            k_raw, k_state, pack.kv_heads, pack.head_dim, is_k=True
        ).to(torch.float32)
        v_t = v_raw.to(torch.float32)
        # quantized (player)
        q_params = sol.hif4_dynamic_quantize_q(
            q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, q_state
        )
        k_params = sol.hif4_dynamic_quantize_k(
            k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, k_state
        )
        v_params = sol.hif4_dynamic_quantize_v(
            value_pair[0], value_pair[1], pack.kv_heads, pack.head_dim, v_state
        )
        q_h = v2.dequantize_hif4(v2._cpu_params(q_params), q_raw.shape).to(torch.float32)
        k_h = v2.dequantize_hif4(v2._cpu_params(k_params), k_raw.shape).to(torch.float32)
        v_h = v2.dequantize_hif4(v2._cpu_params(v_params), v_raw.shape).to(torch.float32)
        ref = v2._attention(
            *(value.to(device)[None] for value in (q_raw, k_raw, v_raw)),
            pack.q_heads, pack.kv_heads, pack.head_dim,
        )
        info = attention_arms(
            q_t, k_t, v_t, q_h, k_h, v_h,
            pack.q_heads, pack.kv_heads, pack.head_dim, ref,
        )
        info.update({
            "case_id": seed.get("case_id"),
            "layer": seed.get("layer"),
            "calibration_indices": seed.get("calibration_indices"),
            "test_window": seed.get("test_window"),
            "test_split": seed.get("test_split"),
            "test_length": seed.get("test_length"),
            "seed_player_gain": seed.get("gain"),
        })
        out.append(info)
    return out


def run_linear_cases(
    sol: Any,
    pack: v2.PreparedPack,
    weight_states: Mapping[tuple[int, str], tuple[Any, Mapping[str, torch.Tensor]]],
    device: torch.device,
    details_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    standard_weight_cache: dict[tuple[int, str], torch.Tensor] = {}
    for case, seed in zip(pack.linear_cases, details_seed):
        state, weight_params = weight_states[(case.layer, case.role)]
        activation_pair = v2._move_pair(
            pack.test_activations[case.role][case.test_window][case.layer], device
        )
        activation_params = sol.hif4_dynamic_quantize_activation(
            activation_pair[0], activation_pair[1], state
        )
        x_raw = v2.dequantize_nvfp4(*activation_pair).to(torch.float32)
        w_raw = v2.dequantize_nvfp4(
            *pack.weights[case.layer][case.role]
        ).to(torch.float32)
        x_t = linear_activation_continuous(x_raw, state, sol)
        w_t = linear_weight_continuous(w_raw, state, sol)
        x_h = v2.dequantize_hif4(
            v2._cpu_params(activation_params), x_raw.shape
        ).to(torch.float32)
        w_h = v2.dequantize_hif4(
            dict(weight_params), w_raw.shape
        ).to(torch.float32)
        ref = x_raw.to(device) @ w_raw.to(device).t()
        info = linear_arms(x_t, w_t, x_h, w_h, ref)
        info.update({
            "case_id": seed.get("case_id"),
            "layer": seed.get("layer"),
            "role": seed.get("role"),
            "role_family": seed.get("role_family"),
            "calibration_indices": seed.get("calibration_indices"),
            "test_window": seed.get("test_window"),
            "test_split": seed.get("test_split"),
            "test_length": seed.get("test_length"),
            "input_width": seed.get("input_width"),
            "output_width": seed.get("output_width"),
            "shape_bucket": seed.get("shape_bucket"),
            "seed_player_gain": seed.get("gain"),
        })
        out.append(info)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_weights(path: Path) -> Mapping[tuple[int, str], tuple[Any, Mapping[str, torch.Tensor]]]:
    return {}


def diagnose(
    solution_path: Path,
    cache_path: Path,
    scenario: str,
    shards: Sequence[int],
    device_name: str,
    output_dir: Path,
    calibration_cache_mode: str = "auto",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    raw = v2.load_pack(cache_path)
    rows: list[dict[str, Any]] = []
    for shard in shards:
        pack = v3.prepare_shard(raw, shard, scenario, ood=False)
        identity = v3._calibration_identity(solution_path, pack, device)
        cache_artifact = v3.default_calibration_cache_path(identity)
        sol = v2.load_solution(solution_path)
        if cache_artifact.is_file():
            weight_states, attention_states = v3.load_calibration_artifact(
                cache_artifact, identity, pack
            )
        else:
            weight_states, attention_states, seconds, calls = v3._calibrate(
                sol, pack, device
            )
            if calibration_cache_mode in {"auto", "write"}:
                v3.save_calibration_artifact(
                    cache_artifact, identity, weight_states, attention_states
                )
        wall_start = time.perf_counter()
        # Reference player details (same math as proxy_v3_eval._score) used as
        # reproducibility seeds.
        seed_linear, seed_attention = v3._score(
            sol, pack, device,
            weight_states, attention_states,
            {name: 0.0 for name in v2.REQUIRED_APIS},
            {name: 0 for name in v2.REQUIRED_APIS},
        )
        scoring_wall = time.perf_counter() - wall_start
        side: dict[str, Any] = {
            "shard": shard,
            "scenario": scenario,
            "seed_scoring_wall_seconds": scoring_wall,
        }
        if pack.attention_cases and scenario in {"both", "attention"}:
            side["attention"] = run_attention_cases(
                sol, pack, attention_states, device, seed_attention
            )
        if pack.linear_cases and scenario in {"both", "linear"}:
            side["linear"] = run_linear_cases(
                sol, pack, weight_states, device, seed_linear
            )
        rows.append(side)
        v3.cleanup_solution_modules()
        del pack
        gc.collect()
    del raw
    gc.collect()
    payload = {
        "protocol": "coordinate-diagnostics-v1",
        "plan": "2026-09-05-coordinate-consistent-error-and-official-probes",
        "source": str(solution_path.resolve()),
        "source_sha256": v2.sha256_file(solution_path),
        "cache": str(cache_path.resolve()),
        "device": str(device),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards": [int(value) for value in shards],
        "rows": rows,
    }
    return payload


def _summarise_attention(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    flat = [item for row in rows for item in row.get("attention", [])]
    if not flat:
        return {}
    names = _OUTPUT_NAMES
    summary: dict[str, Any] = {
        "cases": len(flat),
        "mse_o_t_vs_ref": [float(item["mse_o_t_vs_ref"]) for item in flat],
        "mse_player_vs_ref": [float(item["mse_player_vs_ref"]) for item in flat],
        "mse_player_vs_t": [float(item["mse_player_vs_t"]) for item in flat],
    }
    for arm in names:
        vals = [float(item["mse_to_ref"][arm]) for item in flat]
        summary[f"arm_{arm}_mean_mse_to_ref"] = float(sum(vals) / len(vals))
    for key in ("mean_B2", "mean_E2", "mean_BE"):
        vals = [float(item["be_decomposition"][key]) for item in flat]
        summary[f"be_{key}_mean"] = float(sum(vals) / len(vals))
    be_diff = [
        float(item["be_decomposition"]["lhs_mse_oh_ref"])
        - float(item["be_decomposition"]["rhs_sum"])
        for item in flat
    ]
    summary["be_lhs_minus_rhs_max_abs"] = float(max(abs(v) for v in be_diff))
    return summary


def _summarise_linear(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    flat = [item for row in rows for item in row.get("linear", [])]
    if not flat:
        return {}
    summary: dict[str, Any] = {"cases": len(flat)}
    for name in ("X_tW_t", "X_hW_t", "X_tW_h", "X_hW_h"):
        vals = [float(item["mse_to_ref"][name]) for item in flat]
        summary[f"{name}_mean_mse_to_ref"] = float(sum(vals) / len(vals))
    for key in (
        "ms_T1", "ms_T2", "ms_T3", "cross_12", "cross_13", "cross_23",
        "lhs_ms_Yhh_minus_Ytt", "rhs_exact", "ms_E_X", "ms_E_W",
    ):
        vals = [float(item["expansion"][key]) for item in flat]
        summary[f"expansion_{key}_mean"] = float(sum(vals) / len(vals))
    expansion_diff = [
        abs(float(item["expansion"]["lhs_ms_Yhh_minus_Ytt"])
            - float(item["expansion"]["rhs_exact"]))
        for item in flat
    ]
    summary["expansion_lhs_minus_rhs_max_abs"] = float(max(expansion_diff))
    return summary


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Coordinate-consistent error decomposition",
        "",
        f"- source: `{payload['source']}`",
        f"- source_sha256: `{payload['source_sha256'][:16]}...`",
        f"- cache: `{payload['cache']}`",
        f"- device: `{payload['device']}`  shards: `{payload['shards']}`",
        f"- created: {payload['created_at']}",
        "",
    ]
    for row in payload["rows"]:
        lines.append(f"## shard {row['shard']} ({row['scenario']})")
        attn = _summarise_attention([row])
        if attn:
            lines.append("")
            lines.append("### Attention aggregate")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("|---|---|")
            for key in sorted(attn):
                value = attn[key]
                if isinstance(value, float):
                    value = f"{value:.6e}"
                lines.append(f"| {key} | {value} |")
        lin = _summarise_linear([row])
        if lin:
            lines.append("")
            lines.append("### Linear aggregate")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("|---|---|")
            for key in sorted(lin):
                value = lin[key]
                if isinstance(value, float):
                    value = f"{value:.6e}"
                lines.append(f"| {key} | {value} |")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scenario", choices=("both", "linear", "attention"), default="both")
    parser.add_argument("--shards", default="0", help="comma separated shard ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-cache-mode", choices=("off", "auto", "read", "write"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shards = [int(value.strip()) for value in args.shards.split(",") if value.strip()]
    output_dir = args.output_dir.resolve()
    payload = diagnose(
        args.solution.resolve(),
        args.cache.resolve(),
        args.scenario,
        shards,
        args.device,
        output_dir,
        args.calibration_cache_mode,
    )
    run_id = output_dir.name
    json_path = output_dir / f"diagnostics-{run_id}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (output_dir / f"diagnostics-{run_id}.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
