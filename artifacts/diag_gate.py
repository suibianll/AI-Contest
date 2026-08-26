"""Dev diagnostic: inspect A1 final-gate decisions and deployed-path scores
for the regressed layers (L8/L10/L11) — gated winner vs identity vs B0 proxy.

Run: .venv/Scripts/python.exe artifacts/diag_gate.py
"""

import importlib.util


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load_module("evaluator/real_data_eval.py", "ev")
sol = load_module("solution.py", "sol")
b0 = load_module("solution_b0_tmp.py", "b0")

LAYER_COUNT = 12
HIDDEN = 768
MODE = "amax6"
DEVICE = "cuda"

model, weights, calibration, tests, q_heads, head_dim = ev.collect_real_data(
    "models/gpt2", LAYER_COUNT, 128, 2, 2, device=DEVICE
)
kv_heads = q_heads


def qkv_sample(store, batch, layer_index):
    dense = store["qkv"][batch * LAYER_COUNT + layer_index].reshape(
        -1, 3 * HIDDEN
    )
    q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
    return {
        "q": ev.nvfp4_encode(q_dense, MODE),
        "k": ev.nvfp4_encode(k_dense, MODE),
        "v": ev.nvfp4_encode(v_dense, MODE),
    }


def qkv_test_pairs(store, batch, layer_index):
    dense = store["qkv"][batch * LAYER_COUNT + layer_index].reshape(
        -1, 3 * HIDDEN
    )
    q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
    return (
        ev.nvfp4_encode(q_dense, MODE),
        ev.nvfp4_encode(k_dense, MODE),
        ev.nvfp4_encode(v_dense, MODE),
    )


for layer_index in (8, 10, 11):
    print(f"=== layer {layer_index} ===")
    calib = [qkv_sample(calibration, b, layer_index) for b in range(2)]
    test_pairs = [qkv_test_pairs(tests, b, layer_index) for b in range(2)]
    calib_pairs = [(s["q"], s["k"], s["v"]) for s in calib]

    gate_log = []
    original_gate = sol._a1_gate_passes

    def spy_gate(wc, ws, ic, isf):
        result = original_gate(wc, ws, ic, isf)
        gate_log.append((wc, ws, ic, isf, result))
        return result

    sol._a1_gate_passes = spy_gate
    winner_states = sol.hif4_calibration_attention(
        calib, q_heads, kv_heads, head_dim
    )
    sol._a1_gate_passes = original_gate

    if not gate_log:
        print("  gate: not called (a1 winner == proxy winner)")
    for wc, ws, ic, isf, result in gate_log:
        ratios_c = [w / max(i, 1.0e-12) for w, i in zip(wc, ic)]
        ratios_n = [w / max(i, 1.0e-12) for w, i in zip(ws, isf)]
        print(f"  gate: winner_causal={['%.3e' % v for v in wc]}")
        print(f"        proxy_causal={['%.3e' % v for v in ic]}")
        print(f"        causal ratios={['%.3f' % r for r in ratios_c]}")
        print(
            f"        safety ratios={['%.3f' % r for r in ratios_n]} "
            f"-> keep winner={result}"
        )

    # fallback states: force the gate to reject (yields B0 proxy selection)
    sol._a1_gate_passes = lambda wc, ws, ic, isf: False
    fallback_states = sol.hif4_calibration_attention(
        calib, q_heads, kv_heads, head_dim
    )
    sol._a1_gate_passes = original_gate

    b0_states = b0.hif4_calibration_attention(calib, q_heads, kv_heads, head_dim)

    for label, states in (
        ("winner(gated)", winner_states),
        ("fallback(reject)", fallback_states),
        ("b0(proxy)", b0_states),
    ):
        test_scores = ev.score_attention(
            sol,
            test_pairs,
            states["q_state"],
            states["k_state"],
            states["v_state"],
            q_heads,
            kv_heads,
            head_dim,
            masks=("causal", "non-causal"),
        )
        calib_scores = ev.score_attention(
            sol,
            calib_pairs,
            states["q_state"],
            states["k_state"],
            states["v_state"],
            q_heads,
            kv_heads,
            head_dim,
            masks=("causal", "non-causal"),
        )
        print(
            f"  {label:14s} test: causal={test_scores['causal']:.4f} "
            f"non-causal={test_scores['non-causal']:.4f} | "
            f"calib: causal={calib_scores['causal']:.4f} "
            f"non-causal={calib_scores['non-causal']:.4f}"
        )
