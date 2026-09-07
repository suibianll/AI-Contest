"""L-R1: 真实输入/部署闭环（修复证据链第一优先）。

目标（evidence-repair-next-cycle.md L-R1）：
- 通过 evaluator 的 NVFP4 prepared cache 路径（sprepare_shard/_pair/nvfp4_encode）
  获取与 official 完全一致的 carrier/scale，禁止 round(x/scale) 自制输入。
- 调用父（L4）实际 hif4_calibration_and_quantize_weight 解码返回 weight_params，
  激活调用真实 hif4_dynamic_quantize_activation（不能简化编码器）。
- 以原 NVFP4 解码 XW^T 为最终 reference，核对真实 API 输出与同一 case
  evaluator 的最终 MSE（L4 id shard 产物）。

固定小面板：层 0、11，role o/fc_up/proj（每个 state 两个 in-dist holdout
window，offsets 与 prepare_shard 完全一致），不把训练窗口当 holdout。

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
L4_EVAL_DIR = Path(
    r"d:\工作内容\AI竞赛\artifacts\proxy_v3\v162-independent\linear\l4-v189-linear-exact\id\v162-linear-l4-v189-linear-exact"
)

sys.path.insert(0, str(EVAL_DIR))
import proxy_v3_eval as pv3  # noqa: E402
import official_eval as v2  # noqa: E402

spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device.type} nvfp4_mode={v2.NVFP4_MODE}")


def load_evaluator_mse(target: tuple[int, str]):
    """从 L4 id shard 产物提取同 case 的 mse_player/mse_standard/gain。"""
    layer, role = target
    shard = layer % pv3.SHARD_COUNT
    path = L4_EVAL_DIR / f"candidate-linear-shard{shard}.json"
    m = json.load(open(path, encoding="utf-8"))
    out = []
    for result in m["results"]:
        for it in result["case_scores"]["linear"]:
            if it["layer"] == layer and it["role"] == role:
                out.append({
                    "case_id": it["case_id"],
                    "test_window": it["test_window"],
                    "test_split": it["test_split"],
                    "mse_standard": it["mse_standard"],
                    "mse_player": it["mse_player"],
                    "gain": it["gain"],
                    "calibration_indices": it["calibration_indices"],
                })
    return shard, out


def main():
    raw = v2.load_pack(CACHE)
    for layer, role in ((0, "o"), (0, "fc_up"), (0, "proj"),
                        (11, "o"), (11, "fc_up"), (11, "proj")):
        shard = layer % pv3.SHARD_COUNT
        # 用与官方完全一致的 prepare_shard 构造 (含 _pair 编码与 case 选择)
        pack = pv3.prepare_shard(raw, shard, scenario="linear", ood=False)
        # 取该 layer/role 的校准输入（folds 0,1）
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
        v2.validate_hif4_params(wp, v2.dequantize_nvfp4(*pack.weights[layer][role]).shape)
        v2.validate_state(astate)

        # evaluator 同 case 的期望
        _, expected = load_evaluator_mse((layer, role))
        exp_by_window = {it["test_window"]: it for it in expected}
        mismatches = []
        for case in pack.linear_cases:
            if (case.layer, case.role) != (layer, role):
                continue
            act_pair = v2._move_pair(
                pack.test_activations[role][case.test_window][layer], device
            )
            act_params = sol.hif4_dynamic_quantize_activation(
                act_pair[0], act_pair[1], astate
            )
            ref_act = v2.dequantize_nvfp4(
                *pack.test_activations[role][case.test_window][layer]
            ).to(torch.float32).to(device)
            ref_w = v2.dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32).to(device)
            player_act = v2.dequantize_hif4(
                v2._cpu_params(act_params), ref_act.shape
            ).to(torch.float32).to(device)
            player_w = v2.dequantize_hif4(dict(wp), ref_w.shape).to(torch.float32).to(device)
            reference = ref_act @ ref_w.T
            player = player_act @ player_w.T
            mse_player = float((player - reference).square().mean())
            exp = exp_by_window.get(case.test_window)
            if exp is None:
                mismatches.append(("no-eval-case", case.test_window))
                continue
            diff = abs(mse_player - exp["mse_player"])
            rel = diff / max(abs(exp["mse_player"]), 1e-12)
            if rel > 1e-6:
                mismatches.append(
                    (f"mse-diff rel={rel:.3e}", case.test_window, mse_player, exp["mse_player"])
                )
            print(
                f"[L{layer}-{role}] w{case.test_window} "
                f"split={pack.test_windows[case.test_window].split} "
                f"mse_player={mse_player:.6e} expected={exp['mse_player']:.6e} "
                f"rel_diff={rel:.2e} ok={rel <= 1e-6}"
            )
        if mismatches:
            print(f"  !! L{layer}-{role} mismatches: {mismatches}")


if __name__ == "__main__":
    main()