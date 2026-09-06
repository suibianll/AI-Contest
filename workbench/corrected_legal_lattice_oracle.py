"""R1 output-aware legal-lattice oracle for the corrected codec plan.

This is a research-only oracle.  It does not write candidate state and is not
used by the six submission APIs.  For a sampled weight row/block it enumerates
all finite E6M2 scale codes, both legal lv2 choices, both legal lv3 choices,
and the fixed 16 floor/ceil mantissa patterns.  Selection is made against the
actual calibrated output residual with the parent dynamic activation held
fixed.  The resulting block deltas are evaluated on a held-out calibration
fold and on proxy-v2 test windows.

The oracle is deliberately distinct from the retired legal-codec R1: that
probe minimized operand SSE in a fixed coordinate.  This probe asks whether
the legal scale/hierarchy lattice has output-target headroom under the real
NVFP4 activation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "evaluator", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import reference_hif4 as ref  # noqa: E402
import solution as sol  # noqa: E402
import legal_codec_output_probe as legal  # noqa: E402


E6M2_CODES = tuple(range(255))
HIF4_BLOCK = 64
GROUP_WIDTH = 8
SUBGROUP_WIDTH = 4
PATTERN_BITS = ((torch.arange(16)[:, None] >> torch.arange(4)) & 1).to(torch.bool)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _uniform_indices(total: int, count: int) -> tuple[int, ...]:
    if total <= 0 or count <= 0:
        return ()
    if total == 1 or count == 1:
        return (0,)
    return tuple(sorted({int(math.floor(k * (total - 1) / (count - 1))) for k in range(count)}))


def _move_pair(pair: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    moved = v2._move_pair(v2._pair(pair), device)
    return moved[0], moved[1]


def _sample_tokens(value: torch.Tensor, limit: int) -> torch.Tensor:
    if int(value.shape[0]) <= limit:
        return value.to(torch.float32)
    indices = torch.linspace(0, int(value.shape[0]) - 1, limit, device=value.device).round().to(torch.int64)
    return value.index_select(0, indices).to(torch.float32)


def _fold(
    raw: Any, layer: int, role: str, window: int, state: Mapping[str, Any],
    continuous_weight: torch.Tensor, parent_weight: torch.Tensor,
    device: torch.device, token_limit: int,
) -> dict[str, Any]:
    pair = _move_pair(raw.calibration_activations[role][window][layer], device)
    reference = v2.dequantize_nvfp4(*pair).to(torch.float32)
    reference_shape = tuple(reference.shape)
    reference = _sample_tokens(reference, token_limit)
    transformed = legal._state_activation_transform(reference, state)
    activation_params = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
    quantized = ref.dequantize_hif4(
        activation_params, reference_shape
    ).to(torch.float32)
    quantized = _sample_tokens(quantized, token_limit)
    teacher = transformed.mm(continuous_weight.transpose(0, 1))
    parent_output = quantized.mm(parent_weight.transpose(0, 1))
    return {
        "window": int(window),
        "reference": reference,
        "transformed_reference": transformed,
        "quantized": quantized,
        "teacher": teacher,
        "parent_output": parent_output,
        "residual": teacher - parent_output,
        "split": str(raw.test_windows[window].split) if window < len(raw.test_windows) else "train",
        "length": int(len(raw.test_windows[window].input_ids)) if window < len(raw.test_windows) else 0,
    }


def _candidate_subgroup(
    target: torch.Tensor, scale: torch.Tensor, e2: int, e3: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [16,4] legal values, mantissas and integer codes."""

    denominator = scale * float(1 << (e2 + e3))
    raw_code = target.abs().to(torch.float32) * (4.0 / denominator)
    floor_code = torch.floor(raw_code).clamp(0.0, 7.0).to(torch.int64)
    ceil_code = torch.ceil(raw_code).clamp(0.0, 7.0).to(torch.int64)
    bits = PATTERN_BITS.to(device=device)
    codes = torch.where(bits, ceil_code.unsqueeze(0), floor_code.unsqueeze(0))
    mant = codes.to(torch.float32) * 0.25
    values = torch.sign(target).to(torch.float32).unsqueeze(0) * mant * denominator
    return values, mant, codes


def _score_delta(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    start: int, delta: torch.Tensor,
) -> torch.Tensor:
    """Score one [candidate, width] delta tensor against fixed residuals."""

    scores = []
    for fold_index in train_indices:
        fold = folds[fold_index]
        z = fold["quantized"][:, start:start + int(delta.shape[-1])]
        residual = fold["residual"][:, 0]
        prediction = z.mm(delta.transpose(0, 1))
        scores.append((residual[:, None] - prediction).square().sum(dim=0))
    return torch.stack(scores, dim=0).sum(dim=0)


def _select_row_block(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    row: int, block: int, target_block: torch.Tensor, parent_block: torch.Tensor,
    residuals: list[torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    """One deterministic Gauss-Seidel block oracle for one output row."""

    block_start = int(block) * HIF4_BLOCK
    target = target_block.reshape(HIF4_BLOCK)
    parent = parent_block.reshape(HIF4_BLOCK)
    scale_values = ref.e6m2_decode(
        torch.arange(255, dtype=torch.int64, device=device)
    ).to(torch.float32)

    selected_delta = torch.zeros(HIF4_BLOCK, dtype=torch.float32, device=device)
    selected_values = torch.zeros(HIF4_BLOCK, dtype=torch.float32, device=device)
    selected_mant = torch.zeros(8, 2, 4, dtype=torch.float32, device=device)
    selected_e3 = torch.ones(8, 2, dtype=torch.float32, device=device)
    selected_e2 = torch.ones(8, dtype=torch.float32, device=device)
    selected_code = 0
    current_score = torch.stack([
        residuals[index][:, int(row)].square().sum() for index in train_indices
    ]).sum()
    attempted = 0

    def score_full_group(group_start: int, group_delta: torch.Tensor) -> torch.Tensor:
        scores = []
        for fold_index in train_indices:
            z = folds[fold_index]["quantized"][:, block_start + group_start:block_start + group_start + 8]
            residual = residuals[fold_index][:, int(row)]
            prediction = z.mv(group_delta)
            scores.append((residual - prediction).square().sum())
        return torch.stack(scores).sum()

    for code, scale in zip(E6M2_CODES, scale_values):
        group_choices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]] = []
        for group in range(8):
            group_start = group * GROUP_WIDTH
            group_target = target[group_start:group_start + GROUP_WIDTH]
            group_parent = parent[group_start:group_start + GROUP_WIDTH]
            e2_choices = []
            for e2 in (0, 1):
                subgroup_values = []
                subgroup_mants = []
                subgroup_e3 = []
                for subgroup in range(2):
                    sub_start = subgroup * SUBGROUP_WIDTH
                    values_by_e3 = []
                    scores_by_e3 = []
                    mants_by_e3 = []
                    codes_by_e3 = []
                    for e3 in (0, 1):
                        values, mant, codes = _candidate_subgroup(
                            group_target[sub_start:sub_start + SUBGROUP_WIDTH],
                            scale, e2, e3, device,
                        )
                        delta = values - group_parent[sub_start:sub_start + SUBGROUP_WIDTH].unsqueeze(0)
                        scores = []
                        for fold_index in train_indices:
                            z = folds[fold_index]["quantized"][:, block_start + group_start + sub_start:block_start + group_start + sub_start + SUBGROUP_WIDTH]
                            residual = residuals[fold_index][:, int(row)]
                            scores.append((residual[:, None] - z.mm(delta.transpose(0, 1))).square().sum(dim=0))
                        scores_by_e3.append(torch.stack(scores, dim=0).sum(dim=0))
                        values_by_e3.append(values)
                        mants_by_e3.append(mant)
                        codes_by_e3.append(codes)
                        attempted += int(values.shape[0])
                    best_e3 = int(torch.argmin(torch.stack(scores_by_e3).reshape(-1)) % 16)
                    best_e3_choice = int(torch.argmin(torch.stack(scores_by_e3).amin(dim=1)))
                    best_score = torch.stack(scores_by_e3)[best_e3_choice, best_e3]
                    subgroup_values.append(values_by_e3[best_e3_choice][best_e3])
                    subgroup_mants.append(mants_by_e3[best_e3_choice][best_e3])
                    subgroup_e3.append(best_e3_choice)
                    codes_by_e3[best_e3_choice][best_e3]
                    e2_choices.append(best_score)
                group_values = torch.cat(subgroup_values)
                group_delta = group_values - group_parent
                group_score = score_full_group(group_start, group_delta)
                e2_choices[-1] = group_score
                group_choices.append((group_values, torch.cat(subgroup_mants), torch.tensor(subgroup_e3, device=device), e2, float(group_score)))
            best_group_index = min(range(len(group_choices)), key=lambda index: group_choices[index][4])
            values, mant, e3_values, e2_value, group_score_value = group_choices[best_group_index]
            group_delta = values - group_parent
            current_group_start = block_start + group_start
            # Only accept an actual improvement over the current residual; the
            # parent is always a legal fallback and is the tie winner.
            old_group_score = score_full_group(group_start, torch.zeros_like(group_delta))
            if group_score_value < float(old_group_score):
                selected_values[group_start:group_start + GROUP_WIDTH] = values
                selected_delta[group_start:group_start + GROUP_WIDTH] = group_delta
                selected_mant[group] = mant.reshape(2, 4)
                selected_e3[group] = e3_values
                selected_e2[group] = float(1 + e2_value)
                selected_code = int(code)
                for fold_index in train_indices:
                    z = folds[fold_index]["quantized"][:, current_group_start:current_group_start + GROUP_WIDTH]
                    residuals[fold_index][:, int(row)] -= z.mv(group_delta)
                current_score = current_score - old_group_score + torch.as_tensor(group_score_value, device=device)

    # Build a canonical legal one-block witness for validation and reporting.
    signs = torch.sign(target).reshape(8, 2, 4)
    params = {
        "scale_factor": ref.e6m2_decode(torch.tensor([selected_code], device=device)).reshape(1, 1, 1, 1, 1),
        "scale_lv2": selected_e2.reshape(1, 1, 8, 1, 1),
        "scale_lv3": selected_e3.reshape(1, 1, 8, 2, 1),
        "sign": torch.where(selected_mant == 0.0, torch.zeros_like(signs), signs).reshape(1, 1, 8, 2, 4),
        "mant": selected_mant.reshape(1, 1, 8, 2, 4),
    }
    ref.validate_hif4_params(params, (1, HIF4_BLOCK))
    return {
        "delta": selected_delta,
        "values": selected_values,
        "params": params,
        "scale_code": int(selected_code),
        "changed_values": int((selected_delta != 0.0).sum()),
        "attempted_patterns": int(attempted),
        "final_score": float(current_score),
    }


def _select_row_block_v2(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    row: int, block: int, target_block: torch.Tensor, parent_block: torch.Tensor,
    residuals: list[torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    """Legal output oracle with one shared E6M2 scale per 64-block."""

    block_start = int(block) * HIF4_BLOCK
    target = target_block.reshape(HIF4_BLOCK)
    parent = parent_block.reshape(HIF4_BLOCK)
    scale_values = ref.e6m2_decode(
        torch.arange(255, dtype=torch.int64, device=device)
    ).to(torch.float32)
    base_residuals = [residuals[index][:, int(row)].clone() for index in train_indices]
    base_score = torch.stack([value.square().sum() for value in base_residuals]).sum()
    best_score = base_score
    best_delta = torch.zeros(HIF4_BLOCK, dtype=torch.float32, device=device)
    best_values = parent.clone()
    best_mant = torch.zeros(8, 2, 4, dtype=torch.float32, device=device)
    best_e3 = torch.ones(8, 2, dtype=torch.float32, device=device)
    best_e2 = torch.ones(8, dtype=torch.float32, device=device)
    best_code: int | None = None
    attempted = 0

    for code, scale in zip(E6M2_CODES, scale_values):
        trial_residuals = [value.clone() for value in base_residuals]
        trial_delta = torch.zeros(HIF4_BLOCK, dtype=torch.float32, device=device)
        trial_values = parent.clone()
        trial_mant = torch.zeros(8, 2, 4, dtype=torch.float32, device=device)
        trial_e3 = torch.ones(8, 2, dtype=torch.float32, device=device)
        trial_e2 = torch.ones(8, dtype=torch.float32, device=device)

        for group in range(8):
            group_start = group * GROUP_WIDTH
            group_target = target[group_start:group_start + GROUP_WIDTH]
            group_parent = parent[group_start:group_start + GROUP_WIDTH]
            group_choices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]] = []

            for e2 in (0, 1):
                subgroup_values = []
                subgroup_mants = []
                subgroup_e3 = []
                for subgroup in range(2):
                    sub_start = subgroup * SUBGROUP_WIDTH
                    scores_by_e3 = []
                    values_by_e3 = []
                    mants_by_e3 = []
                    for e3 in (0, 1):
                        values, mant, _ = _candidate_subgroup(
                            group_target[sub_start:sub_start + SUBGROUP_WIDTH],
                            scale, e2, e3, device,
                        )
                        delta = values - group_parent[sub_start:sub_start + SUBGROUP_WIDTH].unsqueeze(0)
                        fold_scores = []
                        for fold_position, fold_index in enumerate(train_indices):
                            z = folds[fold_index]["quantized"][:, block_start + group_start + sub_start:block_start + group_start + sub_start + SUBGROUP_WIDTH]
                            prediction = z.mm(delta.transpose(0, 1))
                            fold_scores.append((trial_residuals[fold_position][:, None] - prediction).square().sum(dim=0))
                        scores_by_e3.append(torch.stack(fold_scores, dim=0).sum(dim=0))
                        values_by_e3.append(values)
                        mants_by_e3.append(mant)
                        attempted += int(values.shape[0])
                    table = torch.stack(scores_by_e3, dim=0)
                    flat_best = int(torch.argmin(table.reshape(-1)))
                    best_e3_choice, best_pattern = divmod(flat_best, 16)
                    subgroup_values.append(values_by_e3[best_e3_choice][best_pattern])
                    subgroup_mants.append(mants_by_e3[best_e3_choice][best_pattern])
                    subgroup_e3.append(best_e3_choice)

                group_values = torch.cat(subgroup_values)
                group_delta = group_values - group_parent
                group_scores = []
                for fold_position, fold_index in enumerate(train_indices):
                    z = folds[fold_index]["quantized"][:, block_start + group_start:block_start + group_start + GROUP_WIDTH]
                    group_scores.append((trial_residuals[fold_position] - z.mv(group_delta)).square().sum())
                group_choices.append((
                    group_values, torch.cat(subgroup_mants),
                    torch.tensor(subgroup_e3, device=device), e2,
                    float(torch.stack(group_scores).sum()),
                ))

            choice = min(range(len(group_choices)), key=lambda index: group_choices[index][4])
            values, mant, e3_values, e2_value, _ = group_choices[choice]
            group_delta = values - group_parent
            trial_values[group_start:group_start + GROUP_WIDTH] = values
            trial_delta[group_start:group_start + GROUP_WIDTH] = group_delta
            trial_mant[group] = mant.reshape(2, 4)
            trial_e3[group] = e3_values + 1.0
            trial_e2[group] = float(1 + e2_value)
            for fold_position, fold_index in enumerate(train_indices):
                z = folds[fold_index]["quantized"][:, block_start + group_start:block_start + group_start + GROUP_WIDTH]
                trial_residuals[fold_position] -= z.mv(group_delta)

        block_scores = []
        for fold_position, fold_index in enumerate(train_indices):
            z = folds[fold_index]["quantized"][:, block_start:block_start + HIF4_BLOCK]
            block_scores.append((base_residuals[fold_position] - z.mv(trial_delta)).square().sum())
        trial_score = torch.stack(block_scores).sum()
        if trial_score < best_score:
            best_score = trial_score
            best_delta = trial_delta
            best_values = trial_values
            best_mant = trial_mant
            best_e3 = trial_e3
            best_e2 = trial_e2
            best_code = int(code)

    accepted = best_code is not None and bool(best_score < base_score)
    if accepted:
        for fold_index in train_indices:
            z = folds[fold_index]["quantized"][:, block_start:block_start + HIF4_BLOCK]
            residuals[fold_index][:, int(row)] -= z.mv(best_delta)

    params = None
    if accepted:
        signs = torch.sign(target).reshape(8, 2, 4)
        params = {
            "scale_factor": ref.e6m2_decode(torch.tensor([best_code], device=device)).reshape(1, 1, 1, 1, 1),
            "scale_lv2": best_e2.reshape(1, 1, 8, 1, 1),
            "scale_lv3": best_e3.reshape(1, 1, 8, 2, 1),
            "sign": torch.where(best_mant == 0.0, torch.zeros_like(signs), signs).reshape(1, 1, 8, 2, 4),
            "mant": best_mant.reshape(1, 1, 8, 2, 4),
        }
        ref.validate_hif4_params(params, (1, HIF4_BLOCK))
    return {
        "delta": best_delta if accepted else torch.zeros_like(best_delta),
        "values": best_values if accepted else parent,
        "params": params,
        "scale_code": int(best_code) if accepted else None,
        "changed_values": int((best_delta != 0.0).sum()) if accepted else 0,
        "attempted_patterns": int(attempted),
        "final_score": float(best_score),
    }


def _select_row_block_v3(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    row: int, block: int, target_block: torch.Tensor, parent_block: torch.Tensor,
    residuals: list[torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    """Vectorized legal output oracle with one shared scale per block.

    The residual objective is represented by ``delta.T @ G @ delta - 2 b.T
    delta``.  This is exactly the squared output objective for the selected
    training folds, but avoids a Python loop over candidate predictions.  For
    each shared scale, each group chooses its two legal lv3/mantissa branches;
    the complete 64-column block then chooses the best shared scale.  All
    decisions use the block-entry residual, so the oracle is deterministic and
    does not pretend that independently selected group scales are legal.
    """

    block_start = int(block) * HIF4_BLOCK
    target = target_block.reshape(HIF4_BLOCK).to(torch.float32)
    parent = parent_block.reshape(HIF4_BLOCK).to(torch.float32)
    scale_values = ref.e6m2_decode(
        torch.arange(255, dtype=torch.int64, device=device)
    ).to(torch.float32)
    base_residuals = [residuals[index][:, int(row)].clone() for index in train_indices]
    gram64 = torch.zeros((HIF4_BLOCK, HIF4_BLOCK), dtype=torch.float32, device=device)
    linear64 = torch.zeros(HIF4_BLOCK, dtype=torch.float32, device=device)
    for fold_position, fold_index in enumerate(train_indices):
        z = folds[fold_index]["quantized"][:, block_start:block_start + HIF4_BLOCK]
        r = base_residuals[fold_position]
        gram64 += z.transpose(0, 1).mm(z)
        linear64 += z.transpose(0, 1).mv(r)

    def quadratic(delta: torch.Tensor, gram: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
        return (delta.mm(gram) * delta).sum(dim=-1) - 2.0 * delta.mv(linear)

    values_by_group: list[torch.Tensor] = []
    mants_by_group: list[torch.Tensor] = []
    e3_by_group: list[torch.Tensor] = []
    scores_by_group: list[torch.Tensor] = []
    attempted = 0

    for group in range(8):
        group_start = group * GROUP_WIDTH
        group_target = target[group_start:group_start + GROUP_WIDTH]
        group_parent = parent[group_start:group_start + GROUP_WIDTH]
        group_values_e2 = []
        group_mants_e2 = []
        group_e3_e2 = []
        group_scores_e2 = []
        gram8 = gram64[group_start:group_start + GROUP_WIDTH, group_start:group_start + GROUP_WIDTH]
        linear8 = linear64[group_start:group_start + GROUP_WIDTH]
        for e2 in (0, 1):
            subgroup_values = []
            subgroup_mants = []
            subgroup_e3 = []
            for subgroup in range(2):
                sub_start = subgroup * SUBGROUP_WIDTH
                target4 = group_target[sub_start:sub_start + SUBGROUP_WIDTH]
                parent4 = group_parent[sub_start:sub_start + SUBGROUP_WIDTH]
                exponent = torch.tensor([e2, e2 + 1], dtype=torch.int64, device=device)
                denominator = scale_values[:, None] * (2.0 ** exponent.to(torch.float32))[None, :]
                raw_code = target4.abs()[None, None, :] * (4.0 / denominator[:, :, None])
                floor_code = torch.floor(raw_code).clamp(0.0, 7.0).to(torch.int64)
                ceil_code = torch.ceil(raw_code).clamp(0.0, 7.0).to(torch.int64)
                bits = PATTERN_BITS.to(device=device)
                codes = torch.where(
                    bits[None, None, :, :],
                    ceil_code[:, :, None, :],
                    floor_code[:, :, None, :],
                )
                values = torch.sign(target4)[None, None, None, :] * codes.to(torch.float32) * 0.25
                values = values * denominator[:, :, None, None]
                delta = values - parent4[None, None, None, :]
                delta_flat = delta.reshape(-1, SUBGROUP_WIDTH)
                gram4 = gram64[group_start + sub_start:group_start + sub_start + SUBGROUP_WIDTH,
                               group_start + sub_start:group_start + sub_start + SUBGROUP_WIDTH]
                linear4 = linear64[group_start + sub_start:group_start + sub_start + SUBGROUP_WIDTH]
                scores = quadratic(delta_flat, gram4, linear4).reshape(255, 2, 16)
                pattern_score, pattern_index = scores.min(dim=-1)
                e3_index = pattern_score.argmin(dim=-1)
                scale_index = torch.arange(255, device=device)
                chosen_values = values[scale_index, e3_index, pattern_index[scale_index, e3_index]]
                chosen_mants = codes[scale_index, e3_index, pattern_index[scale_index, e3_index]].to(torch.float32) * 0.25
                subgroup_values.append(chosen_values)
                subgroup_mants.append(chosen_mants)
                subgroup_e3.append(e3_index)
                attempted += int(values.shape[0] * values.shape[1] * values.shape[2])

            group_values = torch.cat(subgroup_values, dim=-1)
            group_delta = group_values - group_parent[None, :]
            group_score_value = quadratic(group_delta, gram8, linear8)
            group_values_e2.append(group_values)
            group_mants_e2.append(torch.cat(subgroup_mants, dim=-1).reshape(255, 2, 4))
            group_e3_e2.append(torch.stack(subgroup_e3, dim=-1))
            group_scores_e2.append(group_score_value)

        values_by_group.append(torch.stack(group_values_e2, dim=1))
        mants_by_group.append(torch.stack(group_mants_e2, dim=1))
        e3_by_group.append(torch.stack(group_e3_e2, dim=1))
        scores_by_group.append(torch.stack(group_scores_e2, dim=1))

    values_all = torch.stack(values_by_group, dim=2)  # [255,2,8,8]
    mants_all = torch.stack(mants_by_group, dim=2)    # [255,2,8,2,4]
    e3_all = torch.stack(e3_by_group, dim=2)          # [255,2,8,2]
    scores_all = torch.stack(scores_by_group, dim=2)  # [255,2,8]
    e2_index = scores_all.argmin(dim=1)               # [255,8]
    value_index = e2_index[:, None, :, None].expand(-1, 1, -1, GROUP_WIDTH)
    selected_values = values_all.gather(1, value_index).squeeze(1).reshape(255, HIF4_BLOCK)
    mant_index = e2_index[:, None, :, None, None].expand(-1, 1, -1, 2, 4)
    selected_mants = mants_all.gather(1, mant_index).squeeze(1)
    e3_index = e2_index[:, None, :, None].expand(-1, 1, -1, 2)
    selected_e3 = e3_all.gather(1, e3_index).squeeze(1).to(torch.float32) + 1.0
    selected_e2 = e2_index.to(torch.float32) + 1.0
    block_delta = selected_values - parent[None, :]
    block_scores = quadratic(block_delta, gram64, linear64)
    best_scale_index = int(torch.argmin(block_scores))
    accepted = bool(block_scores[best_scale_index] < 0.0)
    best_delta = block_delta[best_scale_index] if accepted else torch.zeros_like(parent)
    if accepted:
        for fold_index in train_indices:
            z = folds[fold_index]["quantized"][:, block_start:block_start + HIF4_BLOCK]
            residuals[fold_index][:, int(row)] -= z.mv(best_delta)
        best_code = best_scale_index
        params = {
            "scale_factor": scale_values[best_scale_index].reshape(1, 1, 1, 1, 1),
            "scale_lv2": selected_e2[best_scale_index].reshape(1, 1, 8, 1, 1),
            "scale_lv3": selected_e3[best_scale_index].reshape(1, 1, 8, 2, 1),
            "sign": torch.where(
                selected_mants[best_scale_index] == 0.0,
                torch.zeros_like(selected_mants[best_scale_index]),
                torch.sign(target).reshape(8, 2, 4),
            ).reshape(1, 1, 8, 2, 4),
            "mant": selected_mants[best_scale_index].reshape(1, 1, 8, 2, 4),
        }
        ref.validate_hif4_params(params, (1, HIF4_BLOCK))
    else:
        best_code = None
        params = None
    return {
        "delta": best_delta,
        "values": selected_values[best_scale_index] if accepted else parent,
        "params": params,
        "scale_code": best_code,
        "changed_values": int((best_delta != 0.0).sum()) if accepted else 0,
        "attempted_patterns": int(attempted),
        "final_score": float(block_scores[best_scale_index]),
    }


def _select_sample(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    rows: Sequence[int], blocks: Sequence[int], target_weight: torch.Tensor,
    parent_weight: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    residuals = [fold["residual"].clone() for fold in folds]
    deltas = torch.zeros((len(rows), len(blocks), HIF4_BLOCK), dtype=torch.float32, device=device)
    records: list[dict[str, Any]] = []
    for row_position, row in enumerate(rows):
        for block_position, block in enumerate(blocks):
            selected = _select_row_block_v3(
                folds, train_indices, int(row), int(block),
                target_weight[int(row), int(block) * HIF4_BLOCK:(int(block) + 1) * HIF4_BLOCK],
                parent_weight[int(row), int(block) * HIF4_BLOCK:(int(block) + 1) * HIF4_BLOCK],
                residuals, device,
            )
            deltas[row_position, block_position] = selected["delta"]
            records.append({
                "row": int(row), "block": int(block),
                "scale_code": selected["scale_code"],
                "changed_values": selected["changed_values"],
                "attempted_patterns": selected["attempted_patterns"],
                "params": (
                    {name: value.detach().cpu().tolist() for name, value in selected["params"].items()}
                    if selected["params"] is not None else None
                ),
            })
    return {"deltas": deltas, "records": records, "residuals": residuals}


def _apply_deltas(
    fold: Mapping[str, Any], selection: Mapping[str, Any],
    rows: Sequence[int], blocks: Sequence[int],
) -> torch.Tensor:
    residual = fold["residual"].clone()
    for row_position, row in enumerate(rows):
        for block_position, block in enumerate(blocks):
            delta = selection["deltas"][row_position, block_position]
            if bool((delta == 0.0).all()):
                continue
            lo = int(block) * HIF4_BLOCK
            prediction = fold["quantized"][:, lo:lo + HIF4_BLOCK].mv(delta)
            residual[:, int(row)] -= prediction
    return residual


def _gain(fold: Mapping[str, Any], candidate_residual: torch.Tensor) -> float:
    parent = float(fold["residual"].square().mean())
    candidate = float(candidate_residual.square().mean())
    return (parent - candidate) / max(parent, 1.0e-30)


def _test_fold(
    raw: Any, layer: int, role: str, window: int, state: Mapping[str, Any],
    continuous_weight: torch.Tensor, parent_weight: torch.Tensor,
    device: torch.device, token_limit: int,
) -> dict[str, Any]:
    pair = _move_pair(raw.test_activations[role][window][layer], device)
    reference_full = v2.dequantize_nvfp4(*pair).to(torch.float32)
    reference_shape = tuple(reference_full.shape)
    reference = _sample_tokens(reference_full, token_limit)
    transformed = legal._state_activation_transform(reference, state)
    params = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
    quantized = _sample_tokens(ref.dequantize_hif4(params, reference_shape).to(torch.float32), token_limit)
    teacher = transformed.mm(continuous_weight.transpose(0, 1))
    parent_output = quantized.mm(parent_weight.transpose(0, 1))
    return {
        "window": int(window), "quantized": quantized,
        "teacher": teacher, "parent_output": parent_output,
        "residual": teacher - parent_output,
        "split": str(raw.test_windows[window].split),
        "length": int(len(raw.test_windows[window].input_ids)),
    }


def _load_state(
    raw: Any, parent_path: Path, layer: int, role: str, device: torch.device,
    cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> tuple[Any, dict[str, torch.Tensor], str]:
    return legal._load_parent_state(raw, parent_path, layer, role, str(device), cache)


def _state_run(
    raw: Any, parent_path: Path, layer: int, role: str, args: argparse.Namespace,
    state_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> dict[str, Any]:
    device = torch.device(args.device)
    weight_pair = _move_pair(raw.weights[layer][role], device)
    raw_weight = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
    state, parent_params, artifact = _load_state(raw, parent_path, layer, role, device, state_cache)
    parent_params = {name: value.to(device=device) for name, value in parent_params.items()}
    parent_weight = ref.dequantize_hif4(parent_params, tuple(raw_weight.shape)).to(torch.float32)
    continuous_weight = legal._state_parent_transform(raw_weight, state)
    rows = _uniform_indices(int(raw_weight.shape[0]), int(args.rows))
    blocks = _uniform_indices(int(raw_weight.shape[1]) // HIF4_BLOCK, int(args.blocks))
    calibration_windows = [int(value) for value in args.calibration_windows]
    folds = [
        _fold(raw, layer, role, window, state, continuous_weight, parent_weight, device, int(args.token_limit))
        for window in calibration_windows
    ]
    loo_records = []
    for held_out in range(len(folds)):
        train = [index for index in range(len(folds)) if index != held_out]
        selection = _select_sample(folds, train, rows, blocks, continuous_weight, parent_weight, device)
        held_out_residual = _apply_deltas(folds[held_out], selection, rows, blocks)
        loo_records.append({
            "held_out_window": int(folds[held_out]["window"]),
            "train_windows": [int(folds[index]["window"]) for index in train],
            "gain": _gain(folds[held_out], held_out_residual),
            "selection": {"records": selection["records"]},
        })
    full_selection = _select_sample(
        folds, list(range(len(folds))), rows, blocks,
        continuous_weight, parent_weight, device,
    )
    holdout_records = []
    for window in args.holdout_windows:
        test = _test_fold(
            raw, layer, role, int(window), state, continuous_weight, parent_weight,
            device, int(args.token_limit),
        )
        candidate_residual = _apply_deltas(test, full_selection, rows, blocks)
        holdout_records.append({
            "window": int(window), "split": test["split"], "length": test["length"],
            "gain": _gain(test, candidate_residual),
        })
    return {
        "layer": int(layer), "role": str(role), "shape": list(raw_weight.shape),
        "rows": list(rows), "blocks": list(blocks),
        "parent_calibration_artifact": artifact,
        "loo": loo_records,
        "holdout": holdout_records,
        "full_selection": {
            "records": full_selection["records"],
            "changed_values": sum(int(item["changed_values"]) for item in full_selection["records"]),
            "attempted_patterns": sum(int(item["attempted_patterns"]) for item in full_selection["records"]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    parent_path = Path(args.parent).resolve()
    cache_path = Path(args.cache).resolve()
    raw = v2.load_pack(cache_path)
    layers = [int(value) for value in args.layers]
    roles = [str(value) for value in args.roles]
    state_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]] = {}
    records = []
    for layer in layers:
        for role in roles:
            records.append(_state_run(raw, parent_path, layer, role, args, state_cache))
    loo_gains = [float(item["gain"]) for record in records for item in record["loo"]]
    holdout_gains = [float(item["gain"]) for record in records for item in record["holdout"]]
    layer_values = []
    for layer in layers:
        values = [
            float(item["gain"])
            for record in records if record["layer"] == layer
            for item in record["holdout"]
        ]
        layer_values.append({
            "layer": int(layer),
            "mean_gain": sum(values) / max(1, len(values)),
            "median_gain": float(torch.median(torch.tensor(values, dtype=torch.float64))) if values else 0.0,
            "positive_fraction": sum(value > 0.0 for value in values) / max(1, len(values)),
        })
    metrics = {
        "state_count": len(records),
        "loo_mean_gain": sum(loo_gains) / max(1, len(loo_gains)),
        "loo_median_gain": float(torch.median(torch.tensor(loo_gains, dtype=torch.float64))) if loo_gains else 0.0,
        "loo_worst_gain": min(loo_gains) if loo_gains else 0.0,
        "holdout_mean_gain": sum(holdout_gains) / max(1, len(holdout_gains)),
        "holdout_median_gain": float(torch.median(torch.tensor(holdout_gains, dtype=torch.float64))) if holdout_gains else 0.0,
        "holdout_l1_mean_abs_gain": sum(abs(value) for value in holdout_gains) / max(1, len(holdout_gains)),
        "holdout_positive_cases": sum(value > 0.0 for value in holdout_gains),
        "holdout_case_count": len(holdout_gains),
        "layer_summary": layer_values,
        "changed_values": sum(int(record["full_selection"]["changed_values"]) for record in records),
        "attempted_patterns": sum(int(record["full_selection"]["attempted_patterns"]) for record in records),
        "elapsed_s": time.perf_counter() - started,
    }
    # R1 is an oracle gate, not a deployment claim.  Requiring a positive
    # holdout median, positive median LOO, and three positive layers prevents
    # a calibration-only oracle witness from opening R2.
    positive_layers = sum(item["median_gain"] > 0.0 for item in layer_values)
    material = (
        metrics["loo_median_gain"] > 0.0
        and metrics["holdout_mean_gain"] > 0.0
        and metrics["holdout_median_gain"] > 0.0
        and metrics["holdout_l1_mean_abs_gain"] < 0.02
        and positive_layers >= 3
    )
    result = {
        "stage": "r1-corrected-legal-lattice-output-oracle",
        "status": "MATERIAL_HEADROOM" if material else "NO_SUPPORTED_MECHANISM",
        "hypothesis": "实际 NVFP4 激活输出目标下，联合枚举合法 E6M2 scale、lv2/lv3 与固定 mantissa 邻接模式可跨折改善 v186 权重输出。",
        "parent": {
            "path": str(parent_path), "sha256": _sha256_file(parent_path),
            "version": "v186",
        },
        "cache": {"path": str(cache_path), "sha256": _sha256_file(cache_path), "protocol": "proxy-v2"},
        "fixed_config": {
            "layers": layers, "roles": roles, "rows": int(args.rows), "blocks": int(args.blocks),
            "calibration_windows": [int(value) for value in args.calibration_windows],
            "holdout_windows": [int(value) for value in args.holdout_windows],
            "token_limit": int(args.token_limit),
            "scale_codes": 255, "lv2_choices": 2, "lv3_choices": 2, "mantissa_patterns": 16,
            "selection": "ascending block,row,group; parent wins ties; train-fold output residual",
        },
        "metrics": metrics,
        "records": records,
        "execution": {
            "git_head": _git_head(), "device": str(args.device),
            "torch": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()),
        },
        "gate": "R1_MATERIAL_HEADROOM_TO_R2" if material else "R1_NO_MATERIAL_LEGAL_OUTPUT_HEADROOM",
        "next_step": "r2" if material else "close_plan",
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="solution.py")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 8, 15, 23])
    parser.add_argument("--roles", nargs="+", default=["q", "k", "v", "o", "fc_gate", "fc_up", "proj"])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--calibration-windows", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--holdout-windows", nargs="+", type=int, default=[1, 2, 6, 7])
    parser.add_argument("--token-limit", type=int, default=128)
    return parser


if __name__ == "__main__":
    payload = run(_parser().parse_args())
    print(json.dumps({
        "stage": payload["stage"], "status": payload["status"],
        "metrics": payload["metrics"], "gate": payload["gate"],
        "next_step": payload["next_step"],
    }, ensure_ascii=False), flush=True)
