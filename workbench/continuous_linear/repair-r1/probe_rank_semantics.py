"""L-R1 rank 语义审计：完整部署连续坐标（含 rank）与 reference 的关系。

检查 L4 的 rank-2 补偿（residual_u/residual_v）是否是无误差的可逆配对：
激活部署链 dense -> smooth -> perm -> hadamard -> rank(A + A U V^T)；
权重在 smooth/perm/hadamard 坐标编码（无显式 R^-T，rank 通过 gram rebind）。
若含 rank 的连续激活 @ 连续权重 != XW^T 且偏差可测，则 rank 属
non-equivalent compensation，单独列 reference 偏差；W-only/A-only 归因
应标 UNIDENTIFIABLE。

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
out_rows = []


def audit(layer: int, role: str):
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
    u = (
        astate["residual_u"].to(device=device, dtype=torch.float32)
        if astate.get("residual_u") is not None else None
    )
    vv = (
        astate["residual_v"].to(device=device, dtype=torch.float32)
        if astate.get("residual_v") is not None else None
    )

    def w_coord(w):
        t = w * d_inv.reciprocal().unsqueeze(0)
        t = t.index_select(-1, perm)
        if bs:
            t = sol._block_hadamard_transform(t, bs, bss)
        return t

    def x_coord(x, apply_rank: bool):
        t = x * d_inv.unsqueeze(0)
        t = t.index_select(-1, perm)
        if bs:
            t = sol._block_hadamard_transform(t, bs, bss)
        if apply_rank and u is not None and vv is not None:
            t = t + (t @ u) @ vv.transpose(0, 1)
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
        reference = ref_x @ ref_w.T

        # 连续参照：带 rank 的 X_rank @ W_coord^T vs reference
        x_rank = x_coord(ref_x, apply_rank=True)
        w_c = w_coord(ref_w)
        prod_rank = x_rank @ w_c.T
        rel_rank = float(
            ((prod_rank - reference) ** 2).mean() / (reference ** 2).mean().clamp_min(1e-12)
        )
        # 不含 rank
        x_nr = x_coord(ref_x, apply_rank=False)
        prod_nr = x_nr @ w_c.T
        rel_nr = float(
            ((prod_nr - reference) ** 2).mean() / (reference ** 2).mean().clamp_min(1e-12)
        )
        # 完整部署（候选 A + 候选 W）的最终误差（应等于 evaluator）
        mse_full = float(((player_x @ player_w.T) - reference).square().mean())
        row = {
            "layer": layer, "role": role, "test_window": case.test_window,
            "split": pack.test_windows[case.test_window].split,
            "rank_present": u is not None and vv is not None,
            "cont_with_rank_relerr": rel_rank,
            "cont_without_rank_relerr": rel_nr,
            "final_mse": mse_full,
        }
        out_rows.append(row)
        print(
            f"[L{layer}-{role} w{case.test_window}] rank={row['rank_present']} "
            f"with_rank_relerr={rel_rank:.2e} without_rank_relerr={rel_nr:.2e} "
            f"final_mse={mse_full:.3e}"
        )


for layer, role in ((0, "o"), (0, "fc_up"), (0, "proj"),
                    (11, "o"), (11, "fc_up"), (11, "proj")):
    try:
        audit(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[L{layer}-{role}] ERROR {type(exc).__name__}: {exc}")

out_path = Path(
    r"d:\工作内容\AI竞赛\artifacts\proxy_v3\continuous\linear\repair-r1\rank-semantics.json"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"rows": out_rows}, indent=1), encoding="utf-8")
print("wrote", out_path)