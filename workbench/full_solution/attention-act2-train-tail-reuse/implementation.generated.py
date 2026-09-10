# ---------------------------------------------------------------------------
# A-CT2 -- the training tail carries its per-role sums instead of walking the
# folds a second time.
#
# The A-GR1 tail walks every `prepared` fold twice.  The `final_loss` loop
# evaluates
#
#     _agr1_scale_loss_grad(
#         _a2_apply_group_rotation(fold[i][0], int(fold[i][2]), m or p), fold[i][1]
#     )[0]
#
# once with `m` for the q role and once with `p` for the k role, and keeps the
# sum.  The two `agr1_*_scale_ratio2` entries of the `info` dict then evaluate
# exactly the same expression on exactly the same folds, keeping only the scalar
# the first walk discarded:
#
#   A-GR1  ... for fold in prepared:
#              for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
#                  value, _ = _agr1_scale_loss_grad(...)
#                  final_loss += float(value.item()) / float(len(prepared))
#          ...
#          "agr1_q_scale_ratio2": <walk every fold again with m>,
#          "agr1_k_scale_ratio2": <walk every fold again with p>,
#
#   A-CT2  ... for fold in prepared:
#              (q_coordinate, q_denominator, q_heads), (k_...) = fold
#              q_value, _ = _agr1_scale_loss_grad(... m ...)
#              final_loss += float(q_value.item()) / float(len(prepared))
#              k_value, _ = _agr1_scale_loss_grad(... p ...)
#              final_loss += float(k_value.item()) / float(len(prepared))
#              q_scale_sum += float(q_value.item())
#              k_scale_sum += float(k_value.item())
#          ...
#          "agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
#          "agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),
#
# Measured on every full-attention layer of the panel: 204
# `_agr1_scale_loss_grad` calls per `_agr1_train` (32 steps x 2 roles x 3 folds
# = 192 in training, 6 in the final_loss loop, 6 in the ratios), and all 6 ratio
# calls repeat a final_loss call with bit-identical inputs and return a
# bit-identical scalar.  The second walk is therefore pure recomputation, and
# deleting it takes the call count to 198.
#
# The two accumulation orders reproduce the parent bit for bit: `final_loss`
# still adds q-then-k per fold with each term divided by len(prepared) first,
# and each ratio sum still accumulates in fold order and divides once at the
# end, which is what the parent's `sum(... for fold in prepared) /
# float(len(prepared))` does.  audit.py checked the latter directly by
# rebuilding both ratios from the loop's own scalars -- exact on all six layers.
#
# The unroll is required, not stylistic: a float is immutable, so passing
# `(q_scale_sum, k_scale_sum)` through `zip` would rebind a local and drop the
# sum.
#
# Nothing else moves: the 32-step training loop and its gradient, _agr1_project,
# _agr1_scale_loss_grad, _a2_apply_group_rotation, the gate, the window and
# candidate configuration, the acceptance rule and the final state compilation
# are the A-GR1 code, byte for byte.
#
# The definition below shadows the A-GR1 one, which stays in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _agr1_train(
    windows: list,
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    force_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Train one general M = I + N per KV group in the frozen parent frame."""

    if int(head_dim) <= 0 or int(head_dim) % 64 != 0:
        raise ValueError("Residual training requires a 64-divisible head dimension")
    parent_q = states["q_state"].get("learned_rotation")
    parent_k = states["k_state"].get("learned_rotation")
    parent_center = states["k_state"].get("learned_center")
    if parent_q is None:
        rq = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rq = parent_q.to(device=device, dtype=torch.float32).clone()
    if parent_k is None:
        rk = torch.eye(int(head_dim), device=device).expand(
            int(kv_heads), int(head_dim), int(head_dim)
        ).clone()
    else:
        rk = parent_k.to(device=device, dtype=torch.float32).clone()
    has_center = parent_center is not None
    if has_center:
        center = parent_center.to(device=device, dtype=torch.float32).clone()
    else:
        center = torch.zeros(int(kv_heads), int(head_dim), device=device)

    n = torch.zeros(
        int(kv_heads), int(head_dim), int(head_dim),
        device=device, dtype=torch.float32,
    )
    eye = torch.eye(int(head_dim), device=device, dtype=torch.float32)
    prepared = []
    for item in windows:
        fold = []
        for role, heads in (("q", q_heads), ("k", kv_heads)):
            raw = _dequantize_nvfp4_float32(*item[role]).to(
                device=device, dtype=torch.float32
            )
            coordinate = _agr1_parent_coordinate(
                raw, states[role + "_state"], heads, head_dim, role == "k"
            )
            denominator = coordinate.reshape(
                -1, int(coordinate.shape[-1]) // 64, 64
            ).abs().amax(-1, keepdim=True)
            fold.append((coordinate, denominator, heads))
        prepared.append(fold)
    if not prepared:
        raise ValueError("Residual training has no calibration windows")

    exp_avg = torch.zeros_like(n)
    exp_avg_sq = torch.zeros_like(n)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, _AGR1_TRAIN_STEPS + 1):
        m = eye + n
        m_inv = torch.linalg.inv(m)
        p = m_inv.transpose(-1, -2)
        grad_m = torch.zeros_like(n)
        grad_p = torch.zeros_like(n)
        loss = n.square().mean() * _AGR1_REG_WEIGHT
        for fold in prepared:
            for (coordinate, denominator, heads), matrix, grad_matrix in zip(
                fold, (m, p), (grad_m, grad_p)
            ):
                transformed = _a2_apply_group_rotation(
                    coordinate, int(heads), matrix
                )
                value, dx = _agr1_scale_loss_grad(transformed, denominator)
                loss = loss + value / float(len(prepared))
                grad_matrix.add_(
                    _agr1_matrix_grad(coordinate, dx, int(heads), int(kv_heads))
                    / float(len(prepared))
                )
        if step == 1:
            initial_loss = float(loss.item())
        if force_zero:
            continue
        gradient = (
            grad_m
            - p @ grad_p.transpose(-1, -2) @ p
            + 2.0 * _AGR1_REG_WEIGHT * n / float(n.numel())
        )
        norm = float(gradient.norm().item())
        if norm > _AGR1_TRAIN_CLIP:
            gradient = gradient * (_AGR1_TRAIN_CLIP / norm)
        exp_avg.mul_(_AGR1_BETA1).add_(gradient, alpha=1.0 - _AGR1_BETA1)
        exp_avg_sq.mul_(_AGR1_BETA2).addcmul_(
            gradient, gradient, value=1.0 - _AGR1_BETA2
        )
        bias1 = 1.0 - _AGR1_BETA1 ** step
        bias2 = 1.0 - _AGR1_BETA2 ** step
        update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1.0e-8)
        n = _agr1_project(eye + n - _AGR1_TRAIN_LR * update) - eye

    if force_zero:
        m = eye.expand(int(kv_heads), int(head_dim), int(head_dim))
        p = m
    else:
        m = eye + n
        p = torch.linalg.inv(m).transpose(-1, -2)
        final_loss = float(n.square().mean().item()) * _AGR1_REG_WEIGHT
        q_scale_sum = 0.0
        k_scale_sum = 0.0
        for fold in prepared:
            (
                (q_coordinate, q_denominator, q_heads),
                (k_coordinate, k_denominator, k_heads),
            ) = fold
            q_value, _ = _agr1_scale_loss_grad(
                _a2_apply_group_rotation(q_coordinate, int(q_heads), m),
                q_denominator,
            )
            final_loss += float(q_value.item()) / float(len(prepared))
            k_value, _ = _agr1_scale_loss_grad(
                _a2_apply_group_rotation(k_coordinate, int(k_heads), p),
                k_denominator,
            )
            final_loss += float(k_value.item()) / float(len(prepared))
            q_scale_sum += float(q_value.item())
            k_scale_sum += float(k_value.item())

    # Compile into the existing root fields, exactly like v192: the K center
    # follows the same M^{-T} transform so the additive K term stays in the
    # same reciprocal coordinate frame and logits are exactly preserved.
    transformed_q = rq @ m
    transformed_k = rk @ p
    center_new = (
        (center.unsqueeze(-2) @ p).squeeze(-2)
        if has_center else None
    )
    inverse_error = float(
        (
            transformed_q @ transformed_k.transpose(-1, -2)
            - rq @ rk.transpose(-1, -2)
        ).abs().max().item()
    )
    singular = torch.linalg.svdvals(m)
    info = {
        "agr1_steps": _AGR1_TRAIN_STEPS,
        "agr1_fit_windows": len(prepared),
        "agr1_attempted_groups": int(kv_heads),
        "agr1_initial_loss": initial_loss,
        "agr1_final_loss": final_loss,
        "agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
        "agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),
        "agr1_inverse_error": inverse_error,
        "agr1_n_norm": float(n.norm().item()),
        "agr1_n_max_abs": float(n.abs().max().item()),
        "agr1_singular_min": float(singular.min().item()),
        "agr1_singular_max": float(singular.max().item()),
        "agr1_parent_arm": "rotation" if parent_q is not None else "identity",
        "agr1_center_compiled": bool(has_center),
    }
    return (
        transformed_q.detach().to(device="cpu").contiguous(),
        transformed_k.detach().to(device="cpu").contiguous(),
        None if center_new is None else center_new.detach().to(device="cpu").contiguous(),
        info,
    )
