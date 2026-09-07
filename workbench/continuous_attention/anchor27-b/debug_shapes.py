"""Debug: print exact shapes inside the A27-B group_loss pipeline (fold 0, g=0)."""
from pathlib import Path
import importlib.util
import math
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load(HERE / "solution.py", "a27_debug")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor27-b-debug") == 0
    try:
        calib = [
            {role: tuple(t.cuda() for t in ev._pair(dense))
             for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][0])}
            for f in range(5)
        ]
        dev = torch.device("cuda")
        with torch.no_grad():
            base = mod._V189_CALIBRATION_ATTENTION(calib, 16, 4, 256)
            qs, ks = base["q_state"], base["k_state"]
            qh, kh, dim = 16, 4, 256
            per_group, bands, cap = qh // kh, 8, 256
            q_width = per_group * dim
            item = calib[0]
            dense_q = mod._dequantize_nvfp4_float32(*item["q"]).to(dev, torch.float32)
            dense_k = mod._dequantize_nvfp4_float32(*item["k"]).to(dev, torch.float32)
            u_q = mod._a1_stack_transform(dense_q, qh, dim, qs, False)
            u_k = mod._a1_stack_transform(dense_k, kh, dim, ks, True)
            print("u_q", tuple(u_q.shape), "u_k", tuple(u_k.shape))
            idx = mod._a2_even_indices(int(u_k.shape[0]), cap, dev)
            keep = int(idx.numel())
            print("keep", keep)
            u_q = u_q.index_select(0, idx)
            u_k = u_k.index_select(0, idx)
            v_params = mod.hif4_dynamic_quantize_v(*item["v"], kh, dim, base["v_state"])
            v_hat = mod._dequantize_hif4(v_params).to(torch.float32).index_select(0, idx)
            print("v_hat", tuple(v_hat.shape))
            g = 0
            u_q_g = u_q.reshape(keep, kh, q_width)[:, g]
            u_k_g = u_k.reshape(keep, kh, dim)[:, g]
            print("u_q_g", tuple(u_q_g.shape), "u_k_g", tuple(u_k_g.shape))
            u_mat = mod._a27_hadamard(dim, dev)
            c0 = torch.zeros(bands, device=dev)
            d_g = mod._a27_band_expand(c0.unsqueeze(0), dim, bands)[0]
            print("d_g", tuple(d_g.shape))
            e_q = (u_mat @ torch.diag_embed(d_g.exp()) @ u_mat.t()).unsqueeze(0)
            print("e_q", tuple(e_q.shape))
            q_rot = mod._a2_apply_group_rotation(u_q_g, per_group, e_q)
            k_rot = mod._a2_apply_group_rotation(u_k_g, 1, e_q)
            print("q_rot", tuple(q_rot.shape), "k_rot", tuple(k_rot.shape))
            imp_q = qs["importance"].detach().to(dev, torch.float32).reshape(-1)[g * q_width:(g + 1) * q_width]
            pq = mod._dense_to_hif4(
                q_rot, importance=imp_q, search_offsets=qs["offsets"],
                error_threshold=float(qs["error_threshold"]), accept_margin=float(qs["accept_margin"]),
                max_refine_ratio=float(qs["max_refine_ratio"]), max_refine_blocks=qs.get("max_refine_blocks"))
            print("pq keys", list(pq.keys()))
            for key, value in pq.items():
                print("  ", key, tuple(value.shape) if torch.is_tensor(value) else value)
            q_hat = mod._dequantize_hif4(pq).to(torch.float32)
            print("q_hat", tuple(q_hat.shape))
            q_hat = q_hat.reshape(-1, per_group, dim)
            pk = mod._dense_to_hif4(
                k_rot, importance=ks["importance"].detach().to(dev, torch.float32).reshape(-1)[g * dim:(g + 1) * dim],
                search_offsets=ks["offsets"], error_threshold=float(ks["error_threshold"]),
                accept_margin=float(ks["accept_margin"]), max_refine_ratio=float(ks["max_refine_ratio"]),
                max_refine_blocks=ks.get("max_refine_blocks"))
            k_hat = mod._dequantize_hif4(pk).to(torch.float32).reshape(-1, 1, dim)
            print("q_hat", tuple(q_hat.shape), "k_hat", tuple(k_hat.shape))
            logits = q_hat @ k_hat.transpose(-1, -2)
            print("logits", tuple(logits.shape))
            v_g = v_hat.reshape(keep, kh, dim)[:, g]
            print("v_g", tuple(v_g.shape))
            out = torch.softmax(logits, dim=-1) @ v_g
            print("out", tuple(out.shape))
    finally:
        gpu_lock.release("A", "anchor27-b-debug")


if __name__ == "__main__":
    main()
