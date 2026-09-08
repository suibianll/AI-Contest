# ---------------------------------------------------------------------------
# v202: compile the existing sample-energy block order from the first decode.
#
# The parent calibration already materializes every calibration activation and
# later materializes the final deployment-coordinate samples for its residual
# solve.  Keep the full first-decode tensors alive and apply the already
# selected final transform once, then store the exact order used by the
# combined Linear wrapper.  This removes the post-calibration NVFP4 decode and
# state reconstruction while leaving all selected parameters and dynamic
# outputs unchanged.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _v202_sample_energy_block_order_from_calibration(
    activation_full_samples: list[torch.Tensor],
    smooth_inv: Optional[torch.Tensor],
    permutation: Optional[torch.Tensor],
    block_smooth_size: int,
    block_smooth_seed: int,
    residual_u: Optional[torch.Tensor],
    residual_v: Optional[torch.Tensor],
    rank1_u: torch.Tensor,
    rank1_v: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not activation_full_samples or not torch.is_tensor(importance):
        return None
    channels = int(importance.numel())
    if channels == 0 or channels % _HIF4_BLOCK_SIZE != 0:
        return None
    energy_sum = None
    window_count = 0
    for raw in activation_full_samples:
        dense = raw
        if smooth_inv is not None:
            dense = dense * smooth_inv.reshape(1, -1)
        if permutation is not None:
            dense = dense.index_select(-1, permutation)
        if int(block_smooth_size) != 0:
            dense = _block_hadamard_transform(
                dense, int(block_smooth_size), int(block_smooth_seed)
            )
        if residual_u is not None and residual_v is not None:
            dense = dense + (dense @ residual_u) @ residual_v.transpose(0, 1)
        else:
            dense = dense + (dense @ rank1_u).unsqueeze(-1) * rank1_v
        if dense.ndim != 2 or int(dense.shape[-1]) != channels:
            return None
        window_energy = dense.to(torch.float32).square().mean(dim=0)
        energy_sum = (
            window_energy if energy_sum is None else energy_sum + window_energy
        )
        window_count += 1
    if energy_sum is None or window_count == 0:
        return None
    block_count = channels // _HIF4_BLOCK_SIZE
    block_scores = (
        (energy_sum / float(window_count)) * importance.to(energy_sum.device)
    ).reshape(block_count, _HIF4_BLOCK_SIZE).sum(dim=1)
    return torch.argsort(
        torch.nan_to_num(block_scores, nan=0.0, posinf=0.0, neginf=0.0),
        descending=True,
    ).to(device="cpu", dtype=torch.int16).contiguous()
