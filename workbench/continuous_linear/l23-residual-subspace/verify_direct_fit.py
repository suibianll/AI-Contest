"""L23 直接拟合修正验证（用户指令 2026-09-07）。

检查项：
1. 全部校准行参与求解（无奇偶拆分；_l23_block_solve 收到全部行）。
2. 增量残差与直接重算一致：最终解码 W 的直接重算损失 == 接受判定的最终损失。
3. 最终五字段解码后的实际拟合误差与接受判定一致（同一 ω 加权口径）。
4. 冻结激活路径：L23 on/off 的 activation_state 逐位一致；动态激活输出一致。
5. Attention control：标准 attention 校准 state 合法且为空（L4 冻结侧）。
6. 仅接受块 sign/mant 改变；scale/lv2/lv3 与父逐位一致。

用法（CPU）：
.venv/Scripts/python.exe workbench/continuous_linear/l23-residual-subspace/verify_direct_fit.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import reference_hif4 as ref  # noqa: E402

CAND = os.path.join(HERE, "candidate", "solution.py")
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def load(name):
    spec = importlib.util.spec_from_file_location(name, CAND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pair(tokens: int, channels: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(tokens, channels, generator=g)
    blocks = channels // 16
    grouped = dense.unflatten(-1, (blocks, 16))
    scale = (grouped.abs().amax(-1) / 6.0).clamp(min=1e-6, max=1e6)
    normalized = grouped / scale[..., None]
    sign = normalized.sign()
    index = torch.argmin((normalized.abs()[..., None] - FP4).abs(), dim=-1)
    quant = (sign * FP4[index]).flatten(-2, -1)
    return quant, scale


def state_tensors_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(state_tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def run() -> int:
    failures: list[str] = []
    torch.manual_seed(0)
    out_f, in_f = 48, 512
    n_blocks = in_f // 64
    wq, ws = make_pair(out_f, in_f, 11)
    act_pairs = [make_pair(64, in_f, 100 + k) for k in range(3)]
    total_rows = 64 * 3  # 全部校准行（3 fold × 64）

    module = load("l23_verify")

    # 0) 奇偶拆分彻底删除
    if hasattr(module, "_l23_fold_split"):
        failures.append("code: _l23_fold_split still present")

    # 1) 全部行参与求解
    seen_rows: list[int] = []
    orig_solve = module._l23_block_solve

    def spy_solve(Xb, R, lam):
        seen_rows.append(int(Xb.shape[0]))
        return orig_solve(Xb, R, lam)

    module._l23_block_solve = spy_solve
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
    module._l23_block_solve = orig_solve
    log = buf.getvalue()
    if not seen_rows:
        failures.append("rows: _l23_block_solve never called (L23 unreachable)")
    else:
        bad = sorted({n for n in seen_rows if n != total_rows})
        if bad:
            failures.append(
                f"rows: solve used row counts {bad}, expected {total_rows} (all rows)"
            )
    m = re.search(r"\[L23\] accepted=(\d+)/(\d+) L_all ([0-9.eE+-]+) → ([0-9.eE+-]+)", log)
    if not m:
        failures.append("log: final [L23] line missing")
        accepted = -1
        L_final = float("nan")
    else:
        accepted = int(m.group(1))
        L_final = float(m.group(4))
        if int(m.group(2)) != n_blocks:
            failures.append(f"log: block count {m.group(2)} != {n_blocks}")
    if accepted == 0:
        failures.append("reach: no block accepted (no-op)")

    ref.validate_hif4_params(result["weight_params"], (out_f, in_f))

    # 2)+3) 最终解码误差 == 接受判定最终损失（同 ω 口径直接重算）
    w_orig = module._dequantize_nvfp4_float32(wq, ws).to(torch.float32)
    W_dec = module._dequantize_hif4(result["weight_params"]).to(torch.float32)
    F = len(act_pairs)
    xs, ys = [], []
    for f, pair in enumerate(act_pairs):
        x = module._dequantize_nvfp4_float32(pair[0], pair[1]).to(torch.float32)
        actp = module.hif4_dynamic_quantize_activation(
            pair[0], pair[1], result["activation_state"]
        )
        xh = module._dequantize_hif4(actp).to(torch.float32)
        y = x @ w_orig.T
        omega = 1.0 / (F * max(float((y ** 2).sum()), 1e-12))
        sw = math.sqrt(omega)
        xs.append(xh * sw)
        ys.append(y * sw)
    xh_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    L_decode = float(((xh_all @ W_dec.t() - y_all) ** 2).mean())
    if not math.isfinite(L_final) or not math.isfinite(L_decode):
        failures.append("loss: non-finite comparison")
    elif abs(L_decode - L_final) > 1e-3 * max(L_final, 1e-30):
        failures.append(
            f"loss: decode {L_decode:.6e} vs tracked final {L_final:.6e} "
            f"(rel {abs(L_decode - L_final) / max(L_final, 1e-30):.3e})"
        )
    else:
        print(f"loss: decode {L_decode:.6e} == tracked final {L_final:.6e} "
              f"(rel {abs(L_decode - L_final) / max(L_final, 1e-30):.3e})")

    # 4) 冻结激活路径 + 6) 仅接受块改变
    module._L23_RESIDUAL_SUBSPACE = False
    with contextlib.redirect_stdout(io.StringIO()):
        parent = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
    module._L23_RESIDUAL_SUBSPACE = True
    if not state_tensors_equal(result["activation_state"], parent["activation_state"]):
        failures.append("freeze: activation_state differs between L23 on/off")
    if not state_tensors_equal(
        module.hif4_dynamic_quantize_activation(
            act_pairs[0][0], act_pairs[0][1], result["activation_state"]
        ),
        module.hif4_dynamic_quantize_activation(
            act_pairs[0][0], act_pairs[0][1], parent["activation_state"]
        ),
    ):
        failures.append("freeze: dynamic activation output differs")
    for key in ("scale_factor", "scale_lv2", "scale_lv3"):
        if not torch.equal(
            result["weight_params"][key], parent["weight_params"][key]
        ):
            failures.append(f"freeze: {key} changed by L23")
    changed = 0
    for b in range(n_blocks):
        s = not torch.equal(result["weight_params"]["sign"][:, b], parent["weight_params"]["sign"][:, b])
        t = not torch.equal(result["weight_params"]["mant"][:, b], parent["weight_params"]["mant"][:, b])
        if s != t:
            failures.append(f"writeback: block {b} sign/mant change mismatch")
            break
        changed += int(s)
    if accepted >= 0 and changed != accepted:
        failures.append(f"writeback: changed blocks {changed} != accepted {accepted}")

    # 5) Attention control（L4 冻结标准侧：state 为空且合法）
    def attn_pair(tokens, channels, seed):
        q, s = make_pair(tokens, channels, seed)
        return q, s

    windows = [
        {
            "q": attn_pair(10, 14 * 64, 7),
            "k": attn_pair(10, 2 * 64, 8),
            "v": attn_pair(10, 2 * 64, 9),
        },
        {
            "q": attn_pair(128, 14 * 64, 10),
            "k": attn_pair(128, 2 * 64, 11),
            "v": attn_pair(128, 2 * 64, 12),
        },
    ]
    states = module.hif4_calibration_attention(windows, 14, 2, 64)
    for name in ("q_state", "k_state", "v_state"):
        ref.validate_state(states[name])
        if states[name] != {}:
            failures.append(f"attention: {name} not the frozen standard empty state")

    if failures:
        print(f"--- {len(failures)} failure(s)")
        for item in failures:
            print("   ", item)
        return 1
    print(f"VERIFY PASS: rows={total_rows} accepted={accepted}/{n_blocks} "
          f"L0→L_final tracked, decode==tracked, states frozen, attention standard")
    return 0


if __name__ == "__main__":
    sys.exit(run())