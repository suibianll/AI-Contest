"""Evaluator-side diagnostic: diff attention states between two solutions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluator"))

import torch

from real_data_eval import collect_real_data, load_solution, nvfp4_encode


def main() -> None:
    a1 = load_solution(Path("solution.py"))
    b0 = load_solution(Path("solution_b0_tmp.py"))
    model, weights, calibration, tests, q_heads, head_dim = collect_real_data(
        "models/gpt2", 12, 128, 2, 2
    )
    hidden = int(model.config.n_embd)
    layer_count = len(weights)

    print(
        "layer | q_mult max|diff  k_mult max|diff  center  perm(q/k) changed"
    )
    for layer in range(layer_count):
        qkv_calibration = []
        for batch in range(2):
            dense = calibration["qkv"][batch * layer_count + layer].reshape(
                -1, 3 * hidden
            )
            q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
            qkv_calibration.append(
                {
                    "q": nvfp4_encode(q_dense, "amax6"),
                    "k": nvfp4_encode(k_dense, "amax6"),
                    "v": nvfp4_encode(v_dense, "amax6"),
                }
            )
        sa = a1.hif4_calibration_attention(
            qkv_calibration, q_heads, q_heads, head_dim
        )
        sb = b0.hif4_calibration_attention(
            qkv_calibration, q_heads, q_heads, head_dim
        )
        q_diff = (
            sa["q_state"]["multiplier"] - sb["q_state"]["multiplier"]
        ).abs().max()
        k_diff = (
            sa["k_state"]["multiplier"] - sb["k_state"]["multiplier"]
        ).abs().max()
        center_a = int(sa["k_state"]["center_mode"])
        center_b = int(sb["k_state"]["center_mode"])
        q_perm_a = sa["q_state"]["permutation"]
        q_perm_b = sb["q_state"]["permutation"]
        k_perm_a = sa["k_state"]["permutation"]
        k_perm_b = sb["k_state"]["permutation"]
        q_perm_same = (
            q_perm_a is None and q_perm_b is None
        ) or (
            q_perm_a is not None
            and q_perm_b is not None
            and torch.equal(q_perm_a, q_perm_b)
        )
        k_perm_same = (
            k_perm_a is None and k_perm_b is None
        ) or (
            k_perm_a is not None
            and k_perm_b is not None
            and torch.equal(k_perm_a, k_perm_b)
        )
        print(
            f"{layer:5d} | {float(q_diff):13.4f} {float(k_diff):13.4f} "
            f"  {center_b}->{center_a}   q={'same' if q_perm_same else 'DIFF'}"
            f" k={'same' if k_perm_same else 'DIFF'}"
        )


if __name__ == "__main__":
    main()
