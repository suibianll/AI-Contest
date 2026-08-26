"""Dev diagnostic: inspect the rotation gate inputs on GQA L1/L6/L7.

Run: .venv/Scripts/python.exe artifacts/diag_rotation.py
"""

import importlib.util

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load_module("evaluator/real_data_eval.py", "ev")
sol = load_module("solution.py", "sol")

LAYER_COUNT = 12
HIDDEN = 768
MODE = "amax6"
DEVICE = "cuda"
KV_HEADS = 6

model, weights, calibration, tests, q_heads, head_dim = ev.collect_real_data(
    "models/gpt2", LAYER_COUNT, 128, 2, 2, device=DEVICE
)
kv_heads = KV_HEADS
group_size = q_heads // kv_heads


def qkv_sample(store, batch, layer_index, gqa):
    dense = store["qkv"][batch * LAYER_COUNT + layer_index].reshape(
        -1, 3 * HIDDEN
    )
    q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
    if gqa:
        k_dense = ev.to_gqa_kv(k_dense, q_heads, kv_heads, head_dim)
        v_dense = ev.to_gqa_kv(v_dense, q_heads, kv_heads, head_dim)
    return {
        "q": ev.nvfp4_encode(q_dense, MODE),
        "k": ev.nvfp4_encode(k_dense, MODE),
        "v": ev.nvfp4_encode(v_dense, MODE),
    }


for layer_index in (1, 6, 7):
    print(f"=== layer {layer_index} (GQA kv={kv_heads}) ===")
    calib = [qkv_sample(calibration, b, layer_index, True) for b in range(2)]
    states = sol.hif4_calibration_attention(
        calib, q_heads, kv_heads, head_dim
    )
    q_rot = states["q_state"].get("rotation")
    print(
        f"  adopted rotation: {q_rot is not None}",
        f"seed-signs sum={int(q_rot.sum().item())}" if q_rot is not None else "",
    )
    # Recompute the gate inputs: base (no rotation) vs rotation, on calib.
    # Emulate via the deployed MSE helper with both state variants.
    q_pairs = [(s["q"][0][:256], s["q"][1][:256]) for s in calib]
    k_pairs = [(s["k"][0][:256], s["k"][1][:256]) for s in calib]
    v_pairs = [(s["v"][0][:256], s["v"][1][:256]) for s in calib]

    # refs: exact float attention on the calib prefix
    refs = []
    for (q_q, q_s), (k_q, k_s), (v_q, v_s) in zip(q_pairs, k_pairs, v_pairs):
        q_dense = sol._dequantize_nvfp4_float32(q_q, q_s).to(torch.float32)
        k_dense = sol._dequantize_nvfp4_float32(k_q, k_s).to(torch.float32)
        v_dense = sol._dequantize_nvfp4_float32(v_q, v_s).to(torch.float32)
        out_c = sol._attention_forward(
            q_dense, k_dense, v_dense, q_heads, kv_heads, head_dim, True
        )
        out_n = sol._attention_forward(
            q_dense, k_dense, v_dense, q_heads, kv_heads, head_dim, False
        )
        refs.append((out_c, out_n))

    v_hats = [
        sol._dequantize_hif4(
            sol.hif4_dynamic_quantize_v(
                v_q, v_s, kv_heads, head_dim, states["v_state"]
            )
        ).to(torch.float32)
        for v_q, v_s in v_pairs
    ]
    base_states = {k: v for k, v in states.items()}
    base_c, base_s = sol._attention_deployed_mse(
        q_pairs, k_pairs, v_hats, refs,
        {**states["q_state"], "rotation": None},
        {**states["k_state"], "rotation": None},
        q_heads, kv_heads, head_dim,
    )
    rot_c, rot_s = sol._attention_deployed_mse(
        q_pairs, k_pairs, v_hats, refs,
        states["q_state"], states["k_state"],
        q_heads, kv_heads, head_dim,
    )
    print(f"  calib causal : base={['%.3e' % v for v in base_c]} rot={['%.3e' % v for v in rot_c]}")
    print(f"  calib safety : base={['%.3e' % v for v in base_s]} rot={['%.3e' % v for v in rot_s]}")
    print(
        f"  mean ratios  : causal={sum(rot_c)/sum(base_c):.4f} "
        f"safety={sum(rot_s)/sum(base_s):.4f}"
    )
