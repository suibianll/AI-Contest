from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import TierConfig
from .formats import quantize_to_nvfp4


Pair = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LinearCase:
    name: str
    weight: Pair
    calibration: tuple[Pair, ...]
    tests: tuple[Pair, ...]


@dataclass(frozen=True)
class AttentionCase:
    name: str
    q_num_heads: int
    kv_num_heads: int
    head_dim: int
    calibration: tuple[dict[str, Pair], ...]
    tests: tuple[dict[str, Pair], ...]


@dataclass(frozen=True)
class EvaluationSuite:
    linear: tuple[LinearCase, ...]
    attention: tuple[AttentionCase, ...]


def _profile(width: int, mode: str, generator: torch.Generator) -> torch.Tensor:
    if mode == "balanced":
        return torch.ones(width)
    if mode == "hierarchy":
        return torch.logspace(-1.0, 1.0, width)
    if mode == "outlier":
        result = torch.ones(width)
        result[::max(1, width // 16)] = 8.0
        return result
    if mode == "heavy_tail":
        return torch.exp(torch.randn(width, generator=generator) * 0.7).clamp(0.05, 20.0)
    raise ValueError(f"unsupported profile mode: {mode}")


def _matrix(
    rows: int,
    profile: torch.Tensor,
    generator: torch.Generator,
    heavy: bool = False,
    mean: Optional[torch.Tensor] = None,
    correlated: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    value = torch.randn(rows, int(profile.numel()), generator=generator) * profile
    if heavy:
        outlier_mask = torch.rand(value.shape, generator=generator) < 0.01
        value = torch.where(outlier_mask, value * 8.0, value)
    if mean is not None:
        value = value + mean
    if correlated:
        grouped = value.reshape(rows, -1, 64)
        shared = grouped.mean(dim=-1, keepdim=True)
        value = (0.55 * grouped + 0.45 * shared).reshape(rows, -1)
    if sparse:
        mask = torch.rand(value.shape, generator=generator) < 0.70
        value = torch.where(mask, torch.zeros_like(value), value)
    return value


def _pair(value: torch.Tensor, device: torch.device) -> Pair:
    quant, scale = quantize_to_nvfp4(value)
    return quant.to(device), scale.to(device)


def build_suite(seed: int, tier: TierConfig, device: torch.device) -> EvaluationSuite:
    """Build deterministic, Torch-only synthetic cases for one evaluation seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    smoke = tier.dev_seeds == 1
    soak = tier.dev_seeds >= 8
    channels, out_features, tokens, seq = (
        (128, 48, 17, 24) if smoke else (512, 256, 97, 80) if soak else (256, 128, 49, 48)
    )
    linear_modes = (
        ("balanced", "outlier", "calib_test_shift")
        if smoke
        else ("balanced", "hierarchy", "outlier", "heavy_tail", "calib_test_shift", "correlated", "sparse", "mean_shift")
    )
    attention_modes = (
        ("balanced", "k_shift", "saturated_logits")
        if smoke
        else ("balanced", "qk_imbalance", "k_shift", "heavy_tail", "saturated_logits", "v_outlier", "qk_correlation", "mean_shift")
    )
    linear_cases: list[LinearCase] = []
    for mode in linear_modes:
        profile_mode = mode if mode in {"balanced", "hierarchy", "outlier", "heavy_tail"} else "hierarchy"
        calib_profile = _profile(channels, profile_mode, generator)
        test_profile = calib_profile
        if mode == "calib_test_shift":
            drift = torch.logspace(-0.65, 0.65, channels)
            test_profile = calib_profile * drift[torch.randperm(channels, generator=generator)]
        mean = 0.75 * torch.randn(channels, generator=generator) if mode == "mean_shift" else None
        weight_profile = calib_profile.reciprocal().clamp(0.05, 20.0)
        weight = _pair(_matrix(out_features, weight_profile, generator, mode == "heavy_tail", correlated=mode == "correlated", sparse=mode == "sparse"), device)
        calibration = tuple(
            _pair(_matrix(tokens, calib_profile, generator, mode == "heavy_tail", mean, mode == "correlated", mode == "sparse"), device)
            for _ in range(tier.calibration_samples)
        )
        tests = tuple(
            _pair(_matrix(tokens + (index % 2) * 16, test_profile, generator, mode in {"heavy_tail", "calib_test_shift"}, mean, mode == "correlated", mode == "sparse"), device)
            for index in range(tier.test_samples)
        )
        linear_cases.append(LinearCase(mode, weight, calibration, tests))

    head_specs = [(4, 2, 64)]
    if not smoke:
        head_specs += [(8, 2, 64), (8, 8, 64)]
    if soak:
        head_specs += [(4, 1, 128)]
    attention_cases: list[AttentionCase] = []
    for mode_index, mode in enumerate(attention_modes):
        q_heads, kv_heads, head_dim = head_specs[mode_index % len(head_specs)]
        group_size = q_heads // kv_heads
        kv_channels = kv_heads * head_dim
        q_channels = q_heads * head_dim
        base_mode = "hierarchy" if mode in {"qk_imbalance", "qk_correlation"} else "heavy_tail" if mode in {"heavy_tail", "v_outlier"} else "balanced"
        k_profile = _profile(kv_channels, base_mode, generator)
        q_profile_kv = k_profile.reciprocal().clamp(0.05, 20.0)
        q_profile = q_profile_kv.reshape(kv_heads, head_dim).repeat_interleave(group_size, dim=0).reshape(q_channels)
        shift = 2.5 * torch.randn(kv_channels, generator=generator) if mode in {"k_shift", "mean_shift"} else None

        def sample(index: int, calibration: bool) -> dict[str, Pair]:
            length = seq if calibration else seq + (index % 2) * 16
            q = _matrix(length, q_profile, generator, mode == "heavy_tail")
            k = _matrix(length, k_profile, generator, mode == "heavy_tail", shift, correlated=mode == "qk_correlation")
            if mode == "qk_correlation":
                shared = torch.randn(length, kv_heads, head_dim, generator=generator)
                k = 0.65 * k.reshape(length, kv_heads, head_dim) + 0.35 * shared
                q_shared = shared.repeat_interleave(group_size, dim=1)
                q = 0.65 * q.reshape(length, q_heads, head_dim) + 0.35 * q_shared
                q, k = q.reshape(length, q_channels), k.reshape(length, kv_channels)
            if mode == "saturated_logits":
                factor = 2.5 if calibration else 3.25
                q, k = q * factor, k * factor
            v = _matrix(length, torch.ones(kv_channels), generator, mode in {"heavy_tail", "v_outlier"})
            if mode == "v_outlier" and not calibration:
                outlier_mask = torch.rand(v.shape, generator=generator) < 0.01
                v = torch.where(outlier_mask, v * 10.0, v)
            return {"q": _pair(q, device), "k": _pair(k, device), "v": _pair(v, device)}

        attention_cases.append(
            AttentionCase(
                f"{mode}_h{q_heads}_kv{kv_heads}_d{head_dim}",
                q_heads,
                kv_heads,
                head_dim,
                tuple(sample(index, True) for index in range(tier.calibration_samples)),
                tuple(sample(index, False) for index in range(tier.test_samples)),
            )
        )
    return EvaluationSuite(tuple(linear_cases), tuple(attention_cases))
