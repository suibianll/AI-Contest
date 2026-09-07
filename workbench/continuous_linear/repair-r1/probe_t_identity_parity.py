"""L-R2 关键核对：deployment_forward(T=I) 是否逐位等于父真实 API 输出。

工作包 L-R2 第 5 点："T=I 应恢复父路径；不能恢复时先查 rank/状态依赖，
不能把父和新目标之间的结构差异归为优化收益。"

上一轮 probe_l2_direction 的 deployment_forward 只重建了 smooth/perm/
hadamard + weight GPTQ + activation GPTQ，但**未包含父的 rank-2 残差
(state residual_u/v) 与 static-actorder hdiag 块序、以及 gptq_block_order
重排**。若 T=I 时该简化前向 != 父真实输出，则 3/3 退化结论不成立（结构差异
混入），需要修正后再判定。

LOCAL diagnostic only。
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

CACHE = Path(r"d:\工作内容\AI竞赛\artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt")
L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
EVAL_DIR = Path(r"d:\工作内容\AI竞赛\evaluator")

sys.path.insert(0, str(EVAL_DIR))
import proxy_v3_eval as pv3  # noqa: E402
import official_eval as v2  # noqa: E402

spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BLOCK = 64
print(f"device={device.type}")


def check(layer: int, role: str):
    raw = v2.load_pack(CACHE)
    shard = layer % pv3.SHARD_COUNT
    pack = pv3.prepare_shard(raw, shard, scenario="linear", ood=False)
    calib_pairs = [
        v2._move_pair(pack.linear_calibration_activations[role][sample][layer], device)
        for sample in pack.metadata["linear_calibration_indices"]
    ]
    weight_pair = v2._move_pair(pack.weights[layer][role], device)
    result = sol.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calib_pairs
    )
    astate = result["activation_state"]
    wp = result["weight_params"]
    ref_w = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32).to(device)
    player_w = v2.dequantize_hif4(dict(wp), ref_w.shape).to(torch.float32).to(device)

    for case in pack.linear_cases:
        if (case.layer, case.role) != (layer, role):
            continue
        act_pair = v2._move_pair(
            pack.test_activations[role][case.test_window][layer], device
        )
        # 父真实动态 API（含 actorder + rank）
        act_params = sol.hif4_dynamic_quantize_activation(
            act_pair[0], act_pair[1], astate
        )
        ref_x = v2.dequantize_nvfp4(
            *pack.test_activations[role][case.test_window][layer]
        ).to(torch.float32).to(device)
        player_x = v2.dequantize_hif4(
            v2._cpu_params(act_params), ref_x.shape
        ).to(torch.float32).to(device)
        ref_out = ref_x @ ref_w.T
        parent = player_x @ player_w.T
        parent_mse = float((parent - ref_out).square().mean())

        # T=I 简化前向（probe_l2_direction 的 deployment_forward 语义）
        # 重建部署坐标（smooth/perm/hadamard），不含 rank/actorder
        d_inv = (
            astate["smooth_inv"].to(device=device, dtype=torch.float32)
            if astate.get("smooth_inv") is not None else torch.ones(ref_w.shape[-1], device=device)
        )
        perm = astate["permutation"]
        if perm is not None:
            perm = perm.to(device=device, dtype=torch.int64)
        else:
            perm = torch.arange(ref_w.shape[-1], dtype=torch.int64, device=device)
        bs = int(astate.get("block_smooth_size", 0))
        bss = int(astate.get("block_smooth_seed", 0))
        W_s = sol._linear_pair_transform(
            ref_w, d_inv.reciprocal(), perm, bs, bss, weight_side=True
        )
        x = ref_x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        if bs:
            x = sol._block_hadamard_transform(x, bs, bss)
        X = x
        # 简化 weight GPTQ（无 rank gram、无 importance 正规）
        gram_full = (X.t() @ X) / max(float(X.shape[0]), 1e-9)
        H = gram_full.clone()
        H.diagonal().add_(0.2 * float(H.diagonal().mean()))
        blocks = int(W_s.shape[1]) // _BLOCK
        wgram = sol._flat_group_gram(gram_full, int(W_s.shape[1])).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(int(W_s.shape[0]), blocks, 8, 2, 4, 4).contiguous()
        wp_s = sol._gptq_quantize_weight(
            W_s, H,
            importance=X.detach().square().mean(dim=0).clamp_min(1e-9),
            group_gram=wgram,
            search_offsets=sol._WEIGHT_OFFSETS,
            error_threshold=float(sol._WEIGHT_REFINE_ERROR_THRESHOLD),
            accept_margin=float(sol._WEIGHT_REFINE_ACCEPT_MARGIN),
            max_refine_ratio=1.0,
            max_refine_blocks=int(sol._WEIGHT_REFINE_MAX_BLOCKS),
            full_sweep_top_k=0, regularization=0.2,
        )
        w_s = v2.dequantize_hif4(dict(wp_s), ref_w.shape).to(torch.float32).to(device)
        wh = sol._dequantize_hif4(wp_s).to(torch.float32)
        wout_gram = wh.t() @ wh
        H_a = wout_gram.clone()
        H_a.diagonal().add_(0.2 * float(H_a.diagonal().mean()))
        L_a = torch.linalg.cholesky(H_a)
        a_gram = sol._flat_group_gram(wout_gram, int(X.shape[1])).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(int(X.shape[0]), blocks, 8, 2, 4, 4).contiguous()
        ap = sol._activation_gptq_quantize(
            X, torch.cholesky_inverse(L_a),
            importance=X.detach().square().mean(dim=0).clamp_min(1e-9),
            group_gram=a_gram,
            search_offsets=sol._DYNAMIC_OFFSETS,
            error_threshold=float(sol._ACTIVATION_REFINE_ERROR_THRESHOLD),
            accept_margin=float(sol._ACTIVATION_REFINE_ACCEPT_MARGIN),
            max_refine_ratio=float(sol._ACTIVATION_REFINE_MAX_RATIO),
            max_refine_blocks=int(sol._ACTIVATION_REFINE_MAX_BLOCKS),
        )
        ah = sol._dequantize_hif4(ap).to(torch.float32)
        simplified = ah @ wh.t()
        simp_mse = float((simplified - ref_out).square().mean())
        rel = simp_mse / parent_mse if parent_mse > 0 else float("inf")
        print(
            f"[L{layer}-{role} w{case.test_window}] parent_mse={parent_mse:.3e} "
            f"T=I_simplified_mse={simp_mse:.3e} ratio={rel:.3f} "
            f"{'MATCH' if rel < 1.05 else 'MISMATCH'}"
        )


for layer, role in ((0, "o"), (11, "proj"), (0, "fc_up")):
    try:
        check(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[{role}-L{layer}] ERROR {type(exc).__name__}: {exc}")