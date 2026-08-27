"""Cap oracle: decompose the per-component mse budget into the weight side
and the activation side on real GPT-2 data.

For each Linear component we compute four arms:
  A  both_player  -- current production path (weight+activation HiF4 player)
  B  w_ref_act_player -- weight perfect (= NVFP4 reference), activation player
                        -> ceiling if we made the weight side lossless
  C  w_player_act_ref -- weight player, activation perfect
                        -> ceiling if we made the activation side lossless
  D  both_ref      -- both perfect (sanity, must be ~1.0)

Standard denominator (std) is frozen reference_hif4 in all arms.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import (  # noqa: E402
    causal_attention,
    collect_real_data,
    load_solution,
    std_hif4,
    to_gqa_kv,
)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean())


def run_linear_arms(
    solution: ModuleType,
    weight_pair,
    calibration_pairs,
    test_pairs,
    mode: str,
) -> dict[str, float]:
    calibrated = solution.hif4_calibration_and_quantize_weight(
        *weight_pair, calibration_pairs
    )
    weight_params = calibrated["weight_params"]
    activation_state = calibrated["activation_state"]

    weight_reference = solution._dequantize_nvfp4_float32(*weight_pair)
    weight_standard = std_hif4(solution, weight_reference)
    weight_player = solution._dequantize_hif4(weight_params).to(torch.float32)

    # Fixed-frame correction: player weights/activations live in the
    # smooth/permutation frame (X_t . W_t == X . W exactly).  Mixing the
    # untransformed NVFP4 reference with a transformed player operand is a
    # coordinate-frame mismatch (that is what made arms B/C look like huge
    # negative scores before).  Transform the reference through the same
    # equivalent transform before substituting either side.
    smooth_inv_raw = activation_state.get("smooth_inv")
    if smooth_inv_raw is None:
        d_w = torch.ones(
            weight_reference.shape[1],
            dtype=torch.float32,
            device=weight_reference.device,
        )
    else:
        d_w = smooth_inv_raw.to(
            device=weight_reference.device, dtype=torch.float32
        ).reciprocal().reshape(-1)
    perm_raw = activation_state.get("permutation")
    if perm_raw is None:
        perm = torch.arange(
            weight_reference.shape[1],
            dtype=torch.int64,
            device=weight_reference.device,
        )
    else:
        perm = perm_raw.to(
            device=weight_reference.device, dtype=torch.int64
        ).reshape(-1)
    bss = int(activation_state.get("block_smooth_size", 0))
    bseed = int(activation_state.get("block_smooth_seed", 0))

    def _w_frame(t: torch.Tensor) -> torch.Tensor:
        return solution._linear_pair_transform(
            t, d_w, perm, bss, bseed, weight_side=True
        )

    def _a_frame(t: torch.Tensor) -> torch.Tensor:
        return solution._linear_pair_transform(
            t, d_w, perm, bss, bseed, weight_side=False
        )

    arms = {"A": [], "B": [], "C": [], "D": []}
    for activation_pair in test_pairs:
        activation_reference = solution._dequantize_nvfp4_float32(
            *activation_pair
        )
        # Same coordinate frame for every operand.
        w_ref_f = _w_frame(weight_reference)
        a_ref_f = _a_frame(activation_reference)
        reference = a_ref_f @ w_ref_f.T
        standard = std_hif4(solution, activation_reference) @ weight_standard.T
        player_activation = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_activation(
                *activation_pair, activation_state
            )
        ).to(torch.float32)

        mse_std = _mse(standard, reference)
        combos = {
            "A": player_activation @ weight_player.T,
            "B": player_activation @ w_ref_f.T,
            "C": a_ref_f @ weight_player.T,
            "D": reference,
        }
        for name, player in combos.items():
            arms[name].append((mse_std - _mse(player, reference)) / mse_std)
    return {name: sum(v) / len(v) for name, v in arms.items()}


def run_attention_arms(
    solution: ModuleType,
    qkv_pairs,
    q_heads,
    kv_heads,
    head_dim,
    mode: str,
) -> dict[str, float]:
    qkv_calibration = []
    for pair in qkv_pairs["calib"]:
        qkv_calibration.append(
            {
                "q": nvfp4_encode(pair["q"][0], mode) if False else pair["q"],
                "k": pair["k"],
                "v": pair["v"],
            }
        )
    states = solution.hif4_calibration_attention(
        qkv_calibration, q_heads, kv_heads, head_dim
    )
    q_state = states["q_state"]
    k_state = states["k_state"]

    def _frame_q(t: torch.Tensor):
        x = t.to(torch.float32)
        m = q_state.get("multiplier")
        if m is not None:
            x = x * m.to(device=x.device, dtype=torch.float32).reshape(1, -1)
        p = q_state.get("permutation")
        if p is not None:
            x = x.index_select(-1, p.to(device=x.device, dtype=torch.int64).reshape(-1))
        r = q_state.get("rotation")
        if r is not None:
            x = solution._apply_attention_rotation(x, q_heads, head_dim, r)
        return x

    def _frame_k(t: torch.Tensor):
        x = t.to(torch.float32)
        cm = int(k_state.get("center_mode", 0))
        if cm != 0:
            x = solution._center_attention_k(x, kv_heads, head_dim, cm)
        m = k_state.get("multiplier")
        if m is not None:
            x = x * m.to(device=x.device, dtype=torch.float32).reshape(1, -1)
        p = k_state.get("permutation")
        if p is not None:
            x = x.index_select(-1, p.to(device=x.device, dtype=torch.int64).reshape(-1))
        r = k_state.get("rotation")
        if r is not None:
            x = solution._apply_attention_rotation(x, kv_heads, head_dim, r)
        return x

    arms = {"A": [], "B": [], "C": [], "D": []}
    for pair in qkv_pairs["test"]:
        q_ref = solution._dequantize_nvfp4_float32(*pair["q"])
        k_ref = solution._dequantize_nvfp4_float32(*pair["k"])
        v_ref = solution._dequantize_nvfp4_float32(*pair["v"])
        q_std = std_hif4(solution, q_ref)
        k_std = std_hif4(solution, k_ref)
        v_std = std_hif4(solution, v_ref)
        q_pl = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_q(
                *pair["q"], q_heads, head_dim, states["q_state"]
            )
        ).to(torch.float32)
        k_pl = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_k(
                *pair["k"], kv_heads, head_dim, states["k_state"]
            )
        ).to(torch.float32)
        v_pl = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_v(
                *pair["v"], kv_heads, head_dim, states["v_state"]
            )
        ).to(torch.float32)

        # Same-frame attention: push the NVFP4 references through the
        # player's Q/K frame (center/multiplier/permutation/rotation).
        # center/rotation are NOT output-equivalent transforms (softmax is
        # nonlinear), so `qk_perfect` isolates the Q/K *encoding* loss in
        # the player frame while the transform-side loss shows up in D.
        q_ref_f = _frame_q(q_ref)
        k_ref_f = _frame_k(k_ref)

        reference = causal_attention(
            q_ref[None], k_ref[None], v_ref[None], q_heads, kv_heads, head_dim, True
        )
        standard = causal_attention(
            q_std[None], k_std[None], v_std[None], q_heads, kv_heads, head_dim, True
        )
        player = causal_attention(
            q_pl[None], k_pl[None], v_pl[None], q_heads, kv_heads, head_dim, True
        )
        perfect = causal_attention(
            q_pl[None], k_pl[None], v_ref[None], q_heads, kv_heads, head_dim, True
        )
        qk_perfect = causal_attention(
            q_ref_f[None], k_ref_f[None], v_pl[None], q_heads, kv_heads, head_dim, True
        )
        both_ref_f = causal_attention(
            q_ref_f[None], k_ref_f[None], v_ref[None], q_heads, kv_heads, head_dim, True
        )
        mse_std = _mse(standard, reference)
        combos = {
            "A": player,
            "B": perfect,
            "C": qk_perfect,
            "D": both_ref_f,
        }
        for name, out in combos.items():
            arms[name].append((mse_std - _mse(out, reference)) / mse_std)
    return {name: sum(v) / len(v) for name, v in arms.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution.py")
    parser.add_argument("--model", default="models/gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--calib", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--mode", default="amax6")
    args = parser.parse_args()

    solution = load_solution(Path(args.solution))
    model, weights, calibration, tests, q_heads, head_dim = collect_real_data(
        args.model,
        args.layers,
        args.seq,
        args.calib,
        args.test,
        device=args.device,
        token_offset=args.token_offset,
    )
    hidden = int(model.config.n_embd)
    layer_count = len(weights)
    kv_heads = q_heads  # GPT-2 plain MHA

    print(f"mode={args.mode} device={args.device}")
    comp_scores = {
        name: {"A": [], "B": [], "C": [], "D": []}
        for name in ("q", "k", "v", "o", "fc", "proj")
    }
    for layer_index in range(layer_count):
        for name in comp_scores:
            weight_pair = nvfp4_encode(weights[layer_index][name], args.mode)
            calibration_pairs = [
                nvfp4_encode(
                    calibration["act"][name][b * layer_count + layer_index],
                    args.mode,
                )
                for b in range(args.calib)
            ]
            test_pairs = [
                nvfp4_encode(
                    tests["act"][name][b * layer_count + layer_index],
                    args.mode,
                )
                for b in range(args.test)
            ]
            arms = run_linear_arms(
                solution, weight_pair, calibration_pairs, test_pairs, args.mode
            )
            for k in comp_scores[name]:
                comp_scores[name][k].append(arms[k])

    qkv_calib = []
    for b in range(args.calib):
        dense = calibration["qkv"][b * layer_count + layer_index].reshape(-1, 3 * hidden)
        q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
        qkv_calib.append(
            {
                "q": nvfp4_encode(q_dense, args.mode),
                "k": nvfp4_encode(to_gqa_kv(k_dense, q_heads, kv_heads, head_dim), args.mode),
                "v": nvfp4_encode(to_gqa_kv(v_dense, q_heads, kv_heads, head_dim), args.mode),
            }
        )
    qkv_test = []
    for b in range(args.test):
        dense = tests["qkv"][b * layer_count + layer_index].reshape(-1, 3 * hidden)
        q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
        qkv_test.append(
            {
                "q": nvfp4_encode(q_dense, args.mode),
                "k": nvfp4_encode(to_gqa_kv(k_dense, q_heads, kv_heads, head_dim), args.mode),
                "v": nvfp4_encode(to_gqa_kv(v_dense, q_heads, kv_heads, head_dim), args.mode),
            }
        )
    attn_arms = run_attention_arms(
        solution,
        {"calib": qkv_calib, "test": qkv_test},
        q_heads,
        kv_heads,
        head_dim,
        args.mode,
    )

    print("\nper-component arms (A=both_player / B=w_ref_act_player / "
          "C=w_player_act_ref / D=both_ref):")
    for name, scores in comp_scores.items():
        a = sum(scores["A"]) / len(scores["A"])
        b = sum(scores["B"]) / len(scores["B"])
        c = sum(scores["C"]) / len(scores["C"])
        d = sum(scores["D"]) / len(scores["D"])
        print(f"  {name:5s} A={a:+.4f} B={b:+.4f} C={c:+.4f} D={d:+.4f}")
        if name == "q":
            q_arms = (a, b, c, d)
    linear_mean_a = sum(
        sum(scores["A"]) / len(scores["A"]) for scores in comp_scores.values()
    ) / len(comp_scores)
    linear_mean_b = sum(
        sum(scores["B"]) / len(scores["B"]) for scores in comp_scores.values()
    ) / len(comp_scores)
    linear_mean_c = sum(
        sum(scores["C"]) / len(scores["C"]) for scores in comp_scores.values()
    ) / len(comp_scores)
    print(
        f"\nLinear mean: A(now)={linear_mean_a:+.4f} "
        f"B(w-lossless)={linear_mean_b:+.4f} "
        f"C(act-lossless)={linear_mean_c:+.4f}"
    )
    print(
        f"Attention[causal]: A={attn_arms['A']:+.4f} "
        f"B(v-lossless)={attn_arms['B']:+.4f} C={attn_arms['C']:+.4f} "
        f"D={attn_arms['D']:+.4f}"
    )


if __name__ == "__main__":
    main()