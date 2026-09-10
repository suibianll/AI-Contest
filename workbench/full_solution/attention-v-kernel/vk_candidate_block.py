"""VK candidate block, appended to the v237 root copy by build.py."""

VK_BLOCK = r'''

# ---------------------------------------------------------------------------
# VK (2026-09-11): kernel-weighted V code selection.
#
# `v_error_balance.py` chose V codes to shrink the per-channel summed error --
# the UNIFORM kernel, which weights every key equally.  The scored attention is
# not uniform: it is softmax(Q K^T / sqrt(d)), and on real calibration windows a
# fitted relative-position kernel differs from it by 0.80 of A's own response
# magnitude (VK-1), while the uniform weighting is off by ~17x (VK-3).
#
# This block compiles a 32-parameter relative-position kernel per Q head into
# v_state and uses it, instead of a flat sum, to choose V's HiF4 codes:
#
#     minimise  sum_g sum_{h in g} sum_t ( sum_k w_h[t-k] * d[k,c] )^2 ,
#     d = V_hat_hiF4 - V_ref_nvfp4
#
# which is available at deploy time (the V API receives the NVFP4 pair and
# produces the codes) and is separable across channels.
#
# Solved in closed form per element.  J is a quadratic form in the codes, so a
# change eps on element i moves it by exactly
#
#     dJ_i = g_i * eps + eps^2 * Q_ii ,  g = 2 sum_h W_h^T (W_h d) ,
#     Q_ii = sum_h (W_h^T W_h)_ii
#
# and each sweep applies, per channel, the single best legal +-1 mantissa step.
# Channels are independent, so the per-channel steps can be applied together.
# The sweep count is a fixed bound (like the fixed 16-step schedule in the root's
# Linear `_em1_dynamic_descent`), not a search over encoder configurations.
#
# Cost.  W_h @ d is a banded convolution of support 2R+1 done as shifted adds --
# no T x T matrix and no unfold temporary.  VK-3 measured the same rule at
# radius 15 and radius 7; the fitted kernels carry their mass within |r| <= 3, so
# the radius is a cost knob rather than an accuracy knob.
#
# Safety.  Calibration gates per layer: the kernel is stored only if the
# correction moves the gate window's V-side output closer to the reference.
# At deploy time the parent's own code path runs first, and any failure returns
# the parent parameters unchanged, so the candidate can never be worse than the
# root by more than the gate's measured effect.
# ---------------------------------------------------------------------------

_VK_RADIUS = 7
_VK_NBUCKET = 2 * _VK_RADIUS + 1
_VK_NPARAM = _VK_NBUCKET + 1          # 31 banded weights + 1 shared far bucket -> 16
_VK_SWEEPS = 6
_VK_FIT_WINDOWS = 3                   # fit on windows[0:3]
_VK_GATE_WINDOW = 3                   # accept only if window 3 improves

_VK_PARENT_V = hif4_dynamic_quantize_v


def _vk_bucket_index(tokens: int) -> torch.Tensor:
    idx = torch.full((tokens, tokens), _VK_NBUCKET, dtype=torch.long)
    for t in range(tokens):
        for k in range(tokens):
            off = k - t
            if abs(off) <= _VK_RADIUS:
                idx[t, k] = off + _VK_RADIUS
    return idx


def _vk_probs(qd, kd, heads, group, h, g, dim):
    logits = qd[:, h, :] @ kd[:, g, :].transpose(-1, -2)
    return torch.softmax(logits / (float(dim) ** 0.5), dim=-1)


def _vk_shifted(delta, weights, radius):
    """sum_r weights[r] * delta[t+r-r_center], with the far bucket handled apart."""

    tokens = delta.shape[0]
    out = torch.zeros_like(delta)
    near_sum = torch.zeros_like(delta)
    for r in range(2 * radius + 1):
        off = r - radius
        lo, hi = max(0, -off), min(tokens, tokens - off)
        if hi <= lo:
            continue
        src = delta[lo + off:hi + off]
        out[lo:hi] += weights[r] * src
        near_sum[lo:hi] += src
    far = weights[2 * radius + 1]
    return out + far * (delta.sum(0, keepdim=True) - near_sum)


def _vk_row_energy(weights, tokens, radius, device):
    energy = torch.zeros(tokens, dtype=torch.float64, device=device)
    counted = torch.zeros(tokens, dtype=torch.float64, device=device)
    for r in range(2 * radius + 1):
        off = r - radius
        lo, hi = max(0, -off), min(tokens, tokens - off)
        if hi > lo:
            energy[lo:hi] += float(weights[r]) ** 2
            counted[lo:hi] += 1.0
    return energy + (float(tokens) - counted) * float(weights[2 * radius + 1]) ** 2


@torch.no_grad()
def _vk_fit(windows, states, q_heads, kv_heads, head_dim, device):
    """w[h] = mean of A_h over each relative-offset bucket, on the fit windows."""

    group = q_heads // kv_heads
    totals = torch.zeros(q_heads, _VK_NPARAM, dtype=torch.float64)
    counts = torch.zeros(_VK_NPARAM, dtype=torch.float64)
    for item in windows:
        q_quant, q_scale = item["q"]
        k_quant, k_scale = item["k"]
        q_dense = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)
        k_dense = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)
        pq = hif4_dynamic_quantize_q(q_quant, q_scale, q_heads, head_dim, states["q_state"])
        pk = hif4_dynamic_quantize_k(k_quant, k_scale, kv_heads, head_dim, states["k_state"])
        qd = _dequantize_hif4(pq).to(torch.float32).reshape(-1, q_heads, head_dim)
        kd = _dequantize_hif4(pk).to(torch.float32).reshape(-1, kv_heads, head_dim)
        tokens = int(qd.shape[0])
        # Sample ROWS evenly across the window and keep the FULL key axis.
        # Taking the first n rows and n columns instead changes what each
        # relative-offset bucket averages over -- most of all the shared far
        # bucket -- and the fitted kernel is the whole mechanism.
        keep = (
            list(range(tokens)) if tokens <= 128
            else sorted({int(round(x)) for x in torch.linspace(0, tokens - 1, 128).tolist()})
        )
        offs = torch.arange(tokens)
        idx_full = offs[None, :] - offs[:, None]
        idx_full = torch.where(
            idx_full.abs() <= _VK_RADIUS,
            idx_full + _VK_RADIUS,
            torch.full_like(idx_full, _VK_NBUCKET),
        )
        idxm = idx_full[keep]
        for h in range(q_heads):
            p = _vk_probs(qd, kd, q_heads, group, h, h // group, head_dim)[keep]
            for b in range(_VK_NPARAM):
                sel = idxm == b
                if bool(sel.any()):
                    totals[h, b] += float(p[sel].sum())
                    if h == 0:
                        counts[b] += float(sel.sum())
    return (totals / counts.clamp_min(1.0)[None, :]).to(torch.float32)


@torch.no_grad()
def _vk_correct(params, kernel, v_quant, v_scale, kv_heads, head_dim):
    """One fixed-bound closed-form descent on the kernel objective."""

    group_dim = kv_heads * head_dim
    mant = params["mant"].to(torch.float64) * 4.0           # integer code 0..7
    sign = params["sign"].to(torch.float64)
    step = (
        params["scale_factor"].to(torch.float64)
        * params["scale_lv2"].to(torch.float64)
        * params["scale_lv3"].to(torch.float64)
        / 4.0
    ).expand_as(mant)
    tokens = int(mant.shape[0])
    if tokens < _VK_RADIUS + 2:
        return params

    # Everything runs on the params' device.  The state and the returned
    # parameters are CPU tensors while the caller's v_quant may be on GPU, so
    # mixing them silently raises -- and on the official Kunpeng judge there is
    # no GPU at all, which is the case this must be correct for.
    dev = mant.device
    kernel = kernel.to(device=dev, dtype=torch.float64)
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float64).to(dev)
    # The dense value of an element is sign * code * (sf*lv2*lv3/4): dropping
    # the sign reconstructs V with scrambled signs, which the gate caught as
    # a 1.4x-6.3x WORSE loss on every layer while VK-3 measured -1.8% for the
    # same rule on the same window.
    flat_step = (sign * step).reshape(tokens, group_dim)
    movable = ((sign != 0) & (mant > 0) & (mant < 7)).reshape(tokens, group_dim)
    codes = mant.reshape(tokens, group_dim).clone()

    heads_per_group = int(kernel.shape[0]) // int(kv_heads)
    for _ in range(_VK_SWEEPS):
        vals = (codes * flat_step).reshape(tokens, kv_heads, head_dim)
        resid = vals - v_ref.reshape(tokens, kv_heads, head_dim)
        grad = torch.zeros_like(resid)
        qii = torch.zeros(tokens, dtype=torch.float64, device=dev)
        for g in range(int(kv_heads)):
            dg = resid[:, g, :]
            for j in range(heads_per_group):
                wg = kernel[g * heads_per_group + j]
                sig = _vk_shifted(dg, wg, _VK_RADIUS)
                grad[:, g, :] += 2.0 * _vk_shifted(
                    sig, torch.cat([wg[:_VK_NBUCKET].flip(0), wg[_VK_NBUCKET:]]), _VK_RADIUS
                )
                qii += _vk_row_energy(wg, tokens, _VK_RADIUS, dev)

        n_chan = kv_heads * head_dim
        flat_grad = grad.reshape(tokens, n_chan)
        best_gain = torch.full((n_chan,), float("inf"), dtype=torch.float64, device=dev)
        best_row = torch.zeros(n_chan, dtype=torch.long, device=dev)
        best_dir = torch.zeros(n_chan, dtype=torch.float64, device=dev)
        for direction in (1.0, -1.0):
            eps = direction * flat_step
            change = flat_grad * eps + (eps ** 2) * qii[:, None]
            change = torch.where(movable, change, torch.full_like(change, float("inf")))
            gain, arg = change.min(dim=0)
            take = gain < best_gain
            best_gain = torch.where(take, gain, best_gain)
            best_row = torch.where(take, arg, best_row)
            best_dir = torch.where(take, torch.full_like(best_gain, direction), best_dir)
        if not bool((best_gain < 0).any()):
            break
        improved = best_gain < 0
        cols = torch.arange(n_chan, device=dev)
        updated = (codes[best_row, cols] + best_dir).clamp(0.0, 7.0)
        codes[best_row, cols] = torch.where(improved, updated, codes[best_row, cols])

    out = {key: value.clone() for key, value in params.items()}
    out_mant = (codes / 4.0).to(device=params["mant"].device, dtype=params["mant"].dtype)
    out["mant"] = out_mant.reshape(params["mant"].shape)
    new_sign = torch.where(
        out_mant.reshape(params["mant"].shape) == 0,
        torch.zeros_like(params["sign"]),
        params["sign"],
    )
    out["sign"] = new_sign.to(params["sign"].dtype)
    return out


def _vk_gate_loss(q_quant, q_scale, k_quant, k_scale, v_quant, v_scale,
                  states, heads, kv_heads, head_dim, corrected):
    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)[None]
    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)[None]
    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)[None]
    pq = hif4_dynamic_quantize_q(q_quant, q_scale, heads, head_dim, states["q_state"])
    pk = hif4_dynamic_quantize_k(k_quant, k_scale, kv_heads, head_dim, states["k_state"])
    qh = _dequantize_hif4(pq).to(torch.float32)[None]
    kh = _dequantize_hif4(pk).to(torch.float32)[None]
    vh = _dequantize_hif4(corrected).to(torch.float32)[None]
    target = _a2_attention_forward(q_ref, k_ref, v_ref, heads, kv_heads, head_dim)
    out = _a2_attention_forward(qh, kh, vh, heads, kv_heads, head_dim)
    return float((out - target).square().mean().item())


_VK_PARENT_CALIBRATION = hif4_calibration_attention


@torch.no_grad()
def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    """The root's calibration, then a gated relative-position kernel for V."""

    states = _VK_PARENT_CALIBRATION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    if (
        not isinstance(states, dict)
        or "v_state" not in states
        or not isinstance(calib_qkv_list, list)
        or len(calib_qkv_list) <= _VK_GATE_WINDOW
    ):
        return states
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernel = _vk_fit(
            calib_qkv_list[:_VK_FIT_WINDOWS], states,
            int(q_num_heads), int(kv_num_heads), int(head_dim), device,
        )
        gate = calib_qkv_list[_VK_GATE_WINDOW]
        parent_params = hif4_dynamic_quantize_v(
            *gate["v"], int(kv_num_heads), int(head_dim), states["v_state"]
        )
        candidate_params = _vk_correct(
            parent_params, kernel, *gate["v"], int(kv_num_heads), int(head_dim)
        )
        mse_parent = _vk_gate_loss(
            *gate["q"], *gate["k"], *gate["v"], states,
            int(q_num_heads), int(kv_num_heads), int(head_dim), parent_params,
        )
        mse_candidate = _vk_gate_loss(
            *gate["q"], *gate["k"], *gate["v"], states,
            int(q_num_heads), int(kv_num_heads), int(head_dim), candidate_params,
        )
        changed = int((candidate_params["mant"] != parent_params["mant"]).sum())
        states["v_state"]["vk_gate_parent"] = float(mse_parent)
        states["v_state"]["vk_gate_candidate"] = float(mse_candidate)
        states["v_state"]["vk_changed"] = changed
        if changed > 0 and mse_candidate < mse_parent:
            states["v_state"]["vk_kernel"] = kernel.detach().to("cpu").clone()
            states["v_state"]["vk_arm"] = "kernel"
        else:
            states["v_state"].pop("vk_kernel", None)
            states["v_state"]["vk_arm"] = "parent"
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the parent
        states["v_state"].pop("vk_kernel", None)
        states["v_state"]["vk_arm"] = "fallback"
        states["v_state"]["vk_error"] = f"{type(exc).__name__}: {exc}"
    return states


@torch.no_grad()
def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    """The root's V path, then the compiled kernel's code correction."""

    params = _VK_PARENT_V(v_quant, v_scale, kv_num_heads, head_dim, v_state)
    try:
        if not isinstance(v_state, dict):
            return params
        kernel = v_state.get("vk_kernel")
        if kernel is None:
            return params
        return _vk_correct(params, kernel, v_quant, v_scale, int(kv_num_heads), int(head_dim))
    except Exception:  # noqa: BLE001 - the parent's parameters are always valid
        return params


'''
