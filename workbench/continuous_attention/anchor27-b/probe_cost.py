"""A27-B pre-registration cost probe: measure true-readout slice encode costs.

Times, for one FA layer of the 4B proxy cache, under the shared GPU lock:
  - base stack calibration (_V189_CALIBRATION_ATTENTION equivalent = module B)
  - one full-window true gate loss (_a21_gate_loss)
  - per-group slice encodes: _dense_to_hif4 on (T,1024) q-slice and (T,256) k-slice
  - the state refine parameters that the FD readout must replicate

Output: probe_cost.json (numbers frozen into mechanism.md config before R-4).
"""
from pathlib import Path
import importlib.util
import json
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
SRC = HERE.parent / "anchor26-a1/solution.py"
LAYER = 0


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load(SRC, "a27_probe")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor27-b-costprobe") == 0
    try:
        calib = [
            {role: tuple(t.cuda() for t in ev._pair(dense))
             for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][LAYER])}
            for f in range(5)
        ]
        with torch.no_grad():
            t0 = time.perf_counter()
            base = mod._V189_CALIBRATION_ATTENTION(calib, 16, 4, 256)
            torch.cuda.synchronize()
            t_base = time.perf_counter() - t0

            qs = base["q_state"]
            refine = {
                "offsets_len": len(qs["offsets"]) if qs.get("offsets") is not None else 0,
                "error_threshold": float(qs["error_threshold"]),
                "accept_margin": float(qs["accept_margin"]),
                "max_refine_ratio": float(qs["max_refine_ratio"]),
                "max_refine_blocks": qs.get("max_refine_blocks"),
                "importance_len": int(qs["importance"].numel()) if qs.get("importance") is not None else 0,
            }

            t0 = time.perf_counter()
            gate_loss = mod._a21_gate_loss(calib[-1], base, 16, 4, 256)
            torch.cuda.synchronize()
            t_gate = time.perf_counter() - t0

            # slice readout timing at c=0 (identity rotation), training folds only
            dev = torch.device("cuda")
            timings = {"q_slice": [], "k_slice": [], "attn": []}
            for f in range(4):
                dense_q = mod._dequantize_nvfp4_float32(*calib[f]["q"]).to(dev, torch.float32)
                dense_k = mod._dequantize_nvfp4_float32(*calib[f]["k"]).to(dev, torch.float32)
                u_q = mod._a1_stack_transform(dense_q, 16, 256, qs, False)
                u_k = mod._a1_stack_transform(dense_k, 4, 256, base["k_state"], True)
                eye = torch.eye(256, device=dev).expand(4, 256, 256)
                q_rot = mod._a2_apply_group_rotation(u_q, 16, eye).reshape(u_q.shape[0], 4, 4, 256)[:, 0].reshape(u_q.shape[0], -1)
                k_rot = mod._a2_apply_group_rotation(u_k, 4, eye).reshape(u_k.shape[0], 4, 256)[:, 0].reshape(u_k.shape[0], -1)
                imp_q = qs["importance"].to(dev)[:1024] if qs.get("importance") is not None else None
                imp_k = base["k_state"]["importance"].to(dev)[:256] if base["k_state"].get("importance") is not None else None
                for _ in range(2):
                    print(f"DEBUG fold{f}: q_rot {tuple(q_rot.shape)} imp_q {None if imp_q is None else tuple(imp_q.shape)}; "
                          f"k_rot {tuple(k_rot.shape)} imp_k {None if imp_k is None else tuple(imp_k.shape)}", flush=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pq = mod._dense_to_hif4(
                        q_rot, importance=imp_q, search_offsets=qs["offsets"],
                        error_threshold=float(qs["error_threshold"]),
                        accept_margin=float(qs["accept_margin"]),
                        max_refine_ratio=float(qs["max_refine_ratio"]),
                        max_refine_blocks=qs.get("max_refine_blocks"),
                    )
                    torch.cuda.synchronize()
                    timings["q_slice"].append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    pk = mod._dense_to_hif4(
                        k_rot, importance=imp_k, search_offsets=base["k_state"]["offsets"],
                        error_threshold=float(base["k_state"]["error_threshold"]),
                        accept_margin=float(base["k_state"]["accept_margin"]),
                        max_refine_ratio=float(base["k_state"]["max_refine_ratio"]),
                        max_refine_blocks=base["k_state"].get("max_refine_blocks"),
                    )
                    torch.cuda.synchronize()
                    timings["k_slice"].append(time.perf_counter() - t0)
                qh = mod._dequantize_hif4(pq).float().reshape(u_q.shape[0], 4, 256)
                kh = mod._dequantize_hif4(pk).float().reshape(u_k.shape[0], 1, 256)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = torch.softmax(qh @ kh.transpose(-1, -2) / 16.0, dim=-1) @ torch.zeros(u_k.shape[0], 1, 256, device=dev)
                torch.cuda.synchronize()
                timings["attn"].append(time.perf_counter() - t0)

        result = {
            "layer": LAYER,
            "base_calib_s": t_base,
            "gate_loss_s": t_gate,
            "gate_loss_value": gate_loss,
            "refine_params": refine,
            "q_slice_s_mean": sum(timings["q_slice"]) / len(timings["q_slice"]),
            "k_slice_s_mean": sum(timings["k_slice"]) / len(timings["k_slice"]),
            "attn_s_mean": sum(timings["attn"]) / len(timings["attn"]),
            "tokens_per_training_foldset": 10 + 128 + 512 + 1024,
        }
        (HERE / "probe_cost.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        gpu_lock.release("A", "anchor27-b-costprobe")


if __name__ == "__main__":
    main()
