"""A-TG1 prerequisite: how much of the gate is redundant, and is loss_identity 1.0?

Two claims are checked here BEFORE any edit, because both are load-bearing:

  COST      how the ~5 s/layer of hif4_calibration_attention splits between
            _a2_train_rotation (the 32-step Adam loop) and the two
            _a2_true_path_gate_loss calls.  A-TF1 already cut the trainer; if the
            gate dominates, that is where the next seconds are.

  IDENTITY  _a2_true_path_gate_loss(rotation=None) builds
            dict(q_state, learned_rotation=None) / dict(k_state, learned_rotation=None)
            and runs the SAME expression as its own standard arm.  If the dynamic
            quantizers treat an explicit None exactly like a missing key, then
            player_mse == standard_mse bit-for-bit and the return value is exactly
            x/max(x,1e-12) -- i.e. 1.0 for every non-degenerate window.

            This is asserted, not assumed: the probe reports the exact float and
            whether it is bit-equal to 1.0, on every real attention layer.

    ATG1_DEVICE=cuda .venv/Scripts/python.exe atg1_probe.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root = load("atg1_root", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device(os.environ.get("ATG1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))]
    print(f"device={dev} q_heads={qh} kv_heads={kvh} head_dim={hd} splits={splits}", flush=True)
    print(f"attention layers: {layers}", flush=True)

    def calib_list(layer):
        return [
            {r: v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
             for i, r in enumerate(("q", "k", "v"))}
            for s in range(splits)
        ]

    windows = [  # token counts per calibration split, on layer 0
        int(v2._pair(pack["calibration_qkv"][s][0][0].to(torch.float32))[0].shape[0])
        for s in range(splits)
    ]
    print(f"calibration window token counts (layer 0): {windows}", flush=True)
    print(f"gate window = split {splits - 1} -> {windows[-1]} tokens", flush=True)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # --- COST: split the calibration call into trainer / gate-identity / gate-rotation
    print(f"\n{'layer':>5} | {'trainer':>9} {'gate_id':>9} {'gate_rot':>9} | {'total':>9} | gate share", flush=True)
    shares = []
    for layer in layers:
        cl = calib_list(layer)
        win = [
            {r: root._dequantize_nvfp4_float32(*item[r]).to(torch.float32) for r in ("q", "k", "v")}
            for item in cl
        ]
        states = root._V189_CALIBRATION_ATTENTION(cl, qh, kvh, hd)

        def t(fn):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - t0, out

        dt_train, (rot, info, ctr) = t(lambda: root._a2_train_rotation(
            win[:-1], qh, kvh, hd, dev))
        dt_gid, loss_id = t(lambda: root._a2_true_path_gate_loss(
            cl[-1], qh, kvh, hd, states, None, dev))
        dt_grot, loss_rot = t(lambda: root._a2_true_path_gate_loss(
            cl[-1], qh, kvh, hd, states, rot, dev, ctr))
        tot = dt_train + dt_gid + dt_grot
        share = (dt_gid + dt_grot) / tot
        shares.append(share)
        print(f"{layer:>5} | {dt_train:>9.3f} {dt_gid:>9.3f} {dt_grot:>9.3f} | {tot:>9.3f} | {share:>6.1%}", flush=True)
        print(f"        loss_identity={loss_id!r}  bit==1.0: {loss_id == 1.0}"
              f"   loss_rotation={loss_rot!r}  arm={'rotation' if loss_rot < loss_id else 'identity'}", flush=True)
    print(f"\ngate share of calibration: median {statistics.median(shares):.1%}", flush=True)

    # --- IDENTITY: is the identity arm's player_mse bit-equal to its own standard arm?
    print("\nidentity-arm bit check (player_mse vs standard_mse):", flush=True)
    for layer in layers:
        cl = calib_list(layer)
        states = root._V189_CALIBRATION_ATTENTION(cl, qh, kvh, hd)
        gw = cl[-1]

        def run(state_q, state_k):
            q_quant, q_scale = gw["q"]
            k_quant, k_scale = gw["k"]
            v_quant, v_scale = gw["v"]
            q_ref = root._dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
            k_ref = root._dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
            v_ref = root._dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)
            qp = root.hif4_dynamic_quantize_q(q_quant, q_scale, qh, hd, state_q)
            kp = root.hif4_dynamic_quantize_k(k_quant, k_scale, kvh, hd, state_k)
            vp = root.hif4_dynamic_quantize_v(v_quant, v_scale, kvh, hd, states["v_state"])
            q_hat = root._dequantize_hif4(qp).to(torch.float32)
            k_hat = root._dequantize_hif4(kp).to(torch.float32)
            v_hat = root._dequantize_hif4(vp).to(torch.float32)
            ref = root._a2_attention_forward(q_ref[None], k_ref[None], v_ref[None], qh, kvh, hd)[0]
            ply = root._a2_attention_forward(q_hat[None], k_hat[None], v_hat[None], qh, kvh, hd)[0]
            return float((ply - ref).square().mean())

        pk = dict(states["k_state"], learned_rotation=None)
        p_arm = run(dict(states["q_state"], learned_rotation=None), pk)
        s_arm = run(states["q_state"], states["k_state"])
        print(f"  layer {layer:>3}: identity-player={p_arm!r}  standard={s_arm!r}  "
              f"bit-equal={p_arm == s_arm}", flush=True)

    print("\nIf the identity-player and standard arms are bit-equal on every layer, then")
    print("loss_identity is exactly x/max(x,1e-12) and the identity player arm is redundant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
