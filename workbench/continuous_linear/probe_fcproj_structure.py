"""fc/proj 桶解析机制前置诊断：部署坐标下逐 64 块的误差能量与可解析特征。

官方 P3：Linear 官方增益 100% 落 fc+proj 大形状桶（W2=fc 1818、W3=proj 1767）。
本探针在真实部署闭环（与 L-R1 一致的输入/API）内，统计 fc/proj 各 state 的：
1. 逐 64 块 player 输出误差能量分布（块级，非 W/A 伪归因；与 L-R1 四臂
   UNIDENTIFIABLE 不冲突——这里只观察部署坐标的块级误差定位）。
2. weight 五字段在块内的 scale/lv2/lv3 使用率、mantissa 分布（解析可优化的
   结构特征：如固定 scale 的块数、lv2/lv3 激活率）。
3. 激活侧 gram/h_inv 的块对角结构（wide 块数多时是否有解析可利用的层间耦合）。

目的：为"fc/proj 解析求解机制"卡提供事实依据；不改变父、不训练、不扫参。
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

spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BLOCK = 64
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
    w_blocks = ref_w.shape[-1] // _BLOCK

    # weight 五字段结构
    sf = wp["scale_factor"].to(torch.float32)   # [out, blocks]
    lv2 = wp["scale_lv2"].to(torch.float32)
    lv3 = wp["scale_lv3"].to(torch.float32)
    mant = wp["mant"].to(torch.float32)
    # 统计 lv2/lv3 激活率 & mantissa 分布（合法编码结构特征）
    lv2_rate = float((lv2 > 1.0).float().mean())
    lv3_rate = float((lv3 > 1.0).float().mean())
    mant_hist = torch.bincount(mant.round().clamp(0, 7).to(torch.int64).reshape(-1),
                               minlength=8).tolist()
    nz = float((mant != 0).float().mean())

    # 第一个 test case 的块级误差
    case = next(c for c in pack.linear_cases
                if c.layer == layer and c.role == role)
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
    reference = ref_x @ ref_w.T
    delta = (player_x @ player_w.T - reference)
    # 输出误差按 out 块（每 64 个输出通道一组）归并
    out_blocks = delta.shape[-1] // _BLOCK
    block_mse = delta.square().reshape(delta.shape[0], out_blocks, _BLOCK).mean(
        dim=(0, 2)
    )
    # 激活侧按 in 块（每 64 通道一组）归并 player-activation 误差
    a_delta = player_x - ref_x
    a_blocks = ref_x.shape[-1] // _BLOCK
    a_block_mse = a_delta.square().reshape(
        a_delta.shape[0], a_blocks, _BLOCK
    ).mean(dim=(0, 2))
    # gram/h_inv 块对角能量
    h_inv = astate.get("h_inv")
    h_block = None
    if h_inv is not None:
        hb = h_inv.to(torch.float32).square()
        hb = hb.reshape(a_blocks, _BLOCK, a_blocks, _BLOCK)
        diag = torch.einsum("iaja->ij", hb).diagonal()
        off = torch.einsum("iaja->ij", hb).sum() - diag.sum()
        h_block = {"diag_share": float(diag.sum() / hb.sum().clamp_min(1e-12)),
                   "off_diag_share": float(off / hb.sum().clamp_min(1e-12))}

    top_out = int(torch.topk(block_mse, k=min(5, out_blocks)).indices.min().item())
    bot_out = int(torch.topk(block_mse, k=min(5, out_blocks), largest=False).indices.min().item())
    out_range = float(block_mse.max() / block_mse.min().clamp_min(1e-12))
    print(
        f"[{role}-L{layer}] w={tuple(ref_w.shape)} blocks={w_blocks} "
        f"lv2_rate={lv2_rate:.3f} lv3_rate={lv3_rate:.3f} mant_nz={nz:.3f} "
        f"mant_hist={mant_hist}"
    )
    print(
        f"    out_block_mse range={out_range:.1f}x top_idx_min={top_out} "
        f"bot_idx_min={bot_out} a_lv2={lv2_rate} "
        f"a_block_mse top/bot={float(a_block_mse.max()/a_block_mse.min().clamp_min(1e-12)):.1f}x"
    )
    if h_block:
        print(f"    h_inv: diag_share={h_block['diag_share']:.3f} "
              f"off_diag_share={h_block['off_diag_share']:.3f}")


for layer, role in ((0, "fc_up"), (0, "fc_gate"), (11, "proj"), (0, "proj")):
    try:
        analyze(layer, role)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()