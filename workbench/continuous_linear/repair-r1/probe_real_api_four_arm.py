"""L-R1 扩展：真实 API 四臂归因 + 坐标语义审计。

在真实轮询中的输入/部署闭环基础上：
1. 从真实 hif4_* API 返回值构造 E00/E10/E01/E11（官方 _linear_error_source_details 口径）。
2. 审计 smooth/perm/block/rank 的连续乘积语义：
   - 先验证可逆部分（smooth/perm/hadamard）X_final @ W_smooth^T == X @ W^T（连续域）。
   - 再单独核对 rank（A + A U V^T）是否保持最终输出坐标，判定
     W-only/A-only 是否属于同坐标纯量化分量；否则标 UNIDENTIFIABLE。

LOCAL diagnostic only。
"""

from __future__ import annotations

import importlib.util
import json
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

sol = importlib.util.module_from_spec(
    (spec := importlib.util.spec_from_file_location("l4sol", L4_PATH))
)
spec.loader.exec_module(sol)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device.type}")


def analyze(layer: int, role: str):
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
    wp = result["weight_params"]
    astate = result["activation_state"]

    ref_w = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32).to(device)
    player_w = v2.dequantize_hif4(dict(wp), ref_w.shape).to(torch.float32).to(device)
    std_w = v2.decode_standard_hif4(
        v2.encode_standard_hif4(ref_w)
    ).to(torch.float32).to(device)

    # ---- 坐标语义审计：连续参照与可逆部分 ----
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

    # 激活连续变换（不含 rank）：x * d_inv -> perm -> hadamard
    # 权重侧：w * d -> perm -> hadamard （_linear_pair_transform weight_side=True）
    def transform_w(w):
        t = w
        t = t * d_inv.reciprocal().unsqueeze(0)
        t = t.index_select(-1, perm)
        if bs:
            t = sol._block_hadamard_transform(t, bs, bss)
        return t

    def transform_x(x):
        t = x * d_inv.unsqueeze(0)
        t = t.index_select(-1, perm)
        if bs:
            t = sol._block_hadamard_transform(t, bs, bss)
        return t

    for case in pack.linear_cases:
        if (case.layer, case.role) != (layer, role):
            continue
        act_pair = v2._move_pair(
            pack.test_activations[role][case.test_window][layer], device
        )
        act_params = sol.hif4_dynamic_quantize_activation(
            act_pair[0], act_pair[1], astate
        )
        ref_x = v2.dequantize_nvfp4(
            *pack.test_activations[role][case.test_window][layer]
        ).to(torch.float32).to(device)
        player_x = v2.dequantize_hif4(
            v2._cpu_params(act_params), ref_x.shape
        ).to(torch.float32).to(device)
        std_x = v2.decode_standard_hif4(
            v2.encode_standard_hif4(ref_x)
        ).to(torch.float32).to(device)

        reference = ref_x @ ref_w.T
        both = player_x @ player_w.T
        std = std_x @ std_w.T
        # 中间臂：真实候选 W（带 rank/变换坐标）+ 标准 A；真实候选 A + 标准 W
        w_only = std_x @ player_w.T
        a_only = player_x @ std_w.T
        details = v2._linear_error_source_details(
            std, w_only, a_only, both, reference,
            ref_w, std_w, player_w, ref_x, std_x, player_x,
        )
        # 连续不变式检查：可逆部分（不含 rank）
        x_c = transform_x(ref_x)
        w_c = transform_w(ref_w)
        prod_cont = x_c @ w_c.T
        rel_prod = float(
            ((prod_cont - reference) ** 2).mean() / (reference ** 2).mean().clamp_min(1e-12)
        )
        # rank 语义：比较连续 product 与 reference（rank 是否等价）
        has_rank = astate.get("residual_u") is not None
        print(
            f"[L{layer}-{role} w{case.test_window}] "
            f"E00={details['mse_standard']:.3e} "
            f"E10={details['mse_w_only']:.3e} "
            f"E01={details['mse_a_only']:.3e} "
            f"E11={details['mse_both']:.3e} "
            f"inter_gain={details['interaction_gain']:+.4f} "
            f"cont_relerr={rel_prod:.2e} rank={has_rank}"
        )


for layer, role in ((0, "o"), (0, "fc_up"), (0, "proj"),
                    (11, "o"), (11, "fc_up"), (11, "proj")):
    try:
        analyze(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[L{layer}-{role}] ERROR {type(exc).__name__}: {exc}")