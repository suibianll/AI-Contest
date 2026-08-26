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


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _profile(
    width: int,
    mode: str,
    generator: torch.Generator,
    *,
    holdout: bool,
) -> torch.Tensor:
    if mode == "balanced":
        sigma = _uniform(generator, 0.0, 0.10 if not holdout else 0.16)
        return torch.exp(torch.randn(width, generator=generator) * sigma)
    if mode == "hierarchy":
        span = _uniform(generator, 0.65, 1.15 if not holdout else 1.35)
        profile = torch.logspace(-span, span, width)
        return profile[torch.randperm(width, generator=generator)]
    if mode == "outlier":
        result = torch.ones(width)
        count = max(1, width // int(_uniform(generator, 12.0, 24.0)))
        indices = torch.randperm(width, generator=generator)[:count]
        result[indices] = _uniform(generator, 5.0, 11.0 if not holdout else 15.0)
        return result
    if mode == "heavy_tail":
        sigma = _uniform(generator, 0.45, 0.90 if not holdout else 1.10)
        return torch.exp(torch.randn(width, generator=generator) * sigma).clamp(
            0.025 if holdout else 0.05, 32.0 if holdout else 20.0
        )
    raise ValueError(f"unsupported profile mode: {mode}")


def _matrix(
    rows: int,
    profile: torch.Tensor,
    generator: torch.Generator,
    *,
    heavy: bool = False,
    mean: Optional[torch.Tensor] = None,
    correlated: bool = False,
    sparse: bool = False,
    holdout: bool = False,
) -> torch.Tensor:
    value = torch.randn(rows, int(profile.numel()), generator=generator) * profile
    if heavy:
        probability = _uniform(generator, 0.005, 0.02 if not holdout else 0.03)
        multiplier = _uniform(generator, 5.0, 11.0 if not holdout else 15.0)
        outlier_mask = torch.rand(value.shape, generator=generator) < probability
        value = torch.where(outlier_mask, value * multiplier, value)
    if mean is not None:
        value = value + mean
    if correlated:
        strength = _uniform(generator, 0.25, 0.70 if not holdout else 0.82)
        grouped = value.reshape(rows, -1, 64)
        shared = grouped.mean(dim=-1, keepdim=True)
        value = ((1.0 - strength) * grouped + strength * shared).reshape(rows, -1)
    if sparse:
        zero_rate = _uniform(generator, 0.50, 0.82 if not holdout else 0.90)
        mask = torch.rand(value.shape, generator=generator) < zero_rate
        value = torch.where(mask, torch.zeros_like(value), value)
    return value


def _pair(value: torch.Tensor, device: torch.device) -> Pair:
    quant, scale = quantize_to_nvfp4(value)
    return quant.to(device), scale.to(device)


def _linear_shapes(tier_name: str) -> tuple[tuple[int, int, int], ...]:
    if tier_name == "smoke":
        return ((128, 48, 17),)
    if tier_name == "standard":
        return (
            (128, 96, 33),
            (192, 128, 49),
            (256, 160, 65),
            (384, 192, 81),
        )
    return (
        (256, 192, 65),
        (384, 224, 81),
        (512, 256, 97),
        (640, 320, 113),
    )


def _attention_specs(tier_name: str) -> tuple[tuple[int, int, int], ...]:
    if tier_name == "smoke":
        return ((4, 2, 64),)
    return (
        (4, 2, 64),
        (8, 2, 64),
        (8, 8, 64),
        (4, 1, 128),
        (4, 4, 128),
        (16, 4, 64),
    )


def _sequence_lengths(tier_name: str) -> tuple[int, ...]:
    if tier_name == "smoke":
        return (24,)
    if tier_name == "standard":
        return (32, 48, 64, 80)
    return (48, 64, 80, 96, 112)


def build_suite(
    seed: int,
    tier: TierConfig,
    device: torch.device,
    *,
    tier_name: str,
    split: str,
) -> EvaluationSuite:
    """Build a deterministic Torch-only suite without scenario/shape confounding."""
    if tier_name not in {"smoke", "standard", "soak"}:
        raise ValueError("tier_name must be smoke, standard, or soak")
    if split not in {"dev", "holdout"}:
        raise ValueError("split must be dev or holdout")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + (0 if split == "dev" else 1_000_000_007))
    smoke = tier_name == "smoke"
    holdout = split == "holdout"

    linear_modes = (
        ("balanced", "outlier", "calib_test_shift")
        if smoke
        else (
            "balanced", "hierarchy", "outlier", "heavy_tail",
            "calib_test_shift", "correlated", "sparse", "mean_shift",
        )
    )
    shapes = _linear_shapes(tier_name)
    shape_order = torch.randperm(len(shapes), generator=generator).tolist()
    linear_cases: list[LinearCase] = []
    for mode_index, mode in enumerate(linear_modes):
        channels, out_features, tokens = shapes[
            shape_order[mode_index % len(shape_order)]
        ]
        profile_mode = (
            mode
            if mode in {"balanced", "hierarchy", "outlier", "heavy_tail"}
            else "hierarchy"
        )
        calib_profile = _profile(
            channels, profile_mode, generator, holdout=holdout
        )
        test_profile = calib_profile
        if mode == "calib_test_shift":
            drift_span = _uniform(generator, 0.30, 0.85 if not holdout else 1.10)
            drift = torch.logspace(-drift_span, drift_span, channels)
            test_profile = calib_profile * drift[
                torch.randperm(channels, generator=generator)
            ]
        calibration_mean = None
        test_mean = None
        if mode == "mean_shift":
            amplitude = _uniform(generator, 0.35, 1.0 if not holdout else 1.35)
            calibration_mean = amplitude * torch.randn(
                channels, generator=generator
            )
            test_mean = -_uniform(generator, 0.25, 0.80) * calibration_mean
        weight_profile = calib_profile.reciprocal().clamp(0.025, 32.0)
        weight = _pair(
            _matrix(
                out_features,
                weight_profile,
                generator,
                heavy=mode == "heavy_tail",
                correlated=mode == "correlated",
                sparse=mode == "sparse",
                holdout=holdout,
            ),
            device,
        )
        calibration = tuple(
            _pair(
                _matrix(
                    tokens,
                    calib_profile,
                    generator,
                    heavy=mode == "heavy_tail",
                    mean=calibration_mean,
                    correlated=mode == "correlated",
                    sparse=mode == "sparse",
                    holdout=holdout,
                ),
                device,
            )
            for _ in range(tier.calibration_samples)
        )
        tests = tuple(
            _pair(
                _matrix(
                    tokens + (index % 2) * 16,
                    test_profile,
                    generator,
                    heavy=mode in {"heavy_tail", "calib_test_shift"},
                    mean=test_mean,
                    correlated=mode == "correlated",
                    sparse=mode == "sparse",
                    holdout=holdout,
                ),
                device,
            )
            for index in range(tier.test_samples)
        )
        linear_cases.append(
            LinearCase(
                f"{mode}_c{channels}_o{out_features}_t{tokens}",
                weight,
                calibration,
                tests,
            )
        )

    attention_modes = (
        ("balanced", "k_shift", "saturated_logits")
        if smoke
        else (
            "balanced", "qk_imbalance", "k_shift", "heavy_tail",
            "saturated_logits", "v_outlier", "qk_correlation", "mean_shift",
        )
    )
    head_specs = _attention_specs(tier_name)
    seq_lengths = _sequence_lengths(tier_name)
    spec_order = torch.randperm(len(head_specs), generator=generator).tolist()
    seq_order = torch.randperm(len(seq_lengths), generator=generator).tolist()
    attention_cases: list[AttentionCase] = []
    for mode_index, mode in enumerate(attention_modes):
        q_heads, kv_heads, head_dim = head_specs[
            spec_order[mode_index % len(spec_order)]
        ]
        seq = seq_lengths[seq_order[mode_index % len(seq_order)]]
        group_size = q_heads // kv_heads
        kv_channels = kv_heads * head_dim
        q_channels = q_heads * head_dim
        base_mode = (
            "hierarchy"
            if mode in {"qk_imbalance", "qk_correlation"}
            else "heavy_tail"
            if mode in {"heavy_tail", "v_outlier"}
            else "balanced"
        )
        k_profile = _profile(
            kv_channels, base_mode, generator, holdout=holdout
        )
        q_profile_kv = k_profile.reciprocal().clamp(0.025, 32.0)
        q_profile = (
            q_profile_kv.reshape(kv_heads, head_dim)
            .repeat_interleave(group_size, dim=0)
            .reshape(q_channels)
        )
        shift_amplitude = _uniform(
            generator, 1.25, 3.5 if not holdout else 4.75
        )
        shift = (
            shift_amplitude * torch.randn(kv_channels, generator=generator)
            if mode in {"k_shift", "mean_shift"}
            else None
        )
        saturated_calibration = _uniform(
            generator, 1.8, 3.0 if not holdout else 3.5
        )
        saturated_test_ratio = _uniform(
            generator, 1.10, 1.50 if not holdout else 1.75
        )

        def sample(index: int, calibration: bool) -> dict[str, Pair]:
            length = seq if calibration else seq + (index % 2) * 16
            q = _matrix(
                length,
                q_profile,
                generator,
                heavy=mode == "heavy_tail",
                holdout=holdout,
            )
            current_shift = shift
            if mode == "mean_shift" and not calibration and shift is not None:
                current_shift = -_uniform(generator, 0.25, 0.90) * shift
            k = _matrix(
                length,
                k_profile,
                generator,
                heavy=mode == "heavy_tail",
                mean=current_shift,
                correlated=mode == "qk_correlation",
                holdout=holdout,
            )
            if mode == "qk_correlation":
                strength = _uniform(
                    generator, 0.25, 0.70 if not holdout else 0.82
                )
                shared = torch.randn(
                    length, kv_heads, head_dim, generator=generator
                )
                k = (
                    (1.0 - strength)
                    * k.reshape(length, kv_heads, head_dim)
                    + strength * shared
                )
                q_shared = shared.repeat_interleave(group_size, dim=1)
                q = (
                    (1.0 - strength)
                    * q.reshape(length, q_heads, head_dim)
                    + strength * q_shared
                )
                q, k = (
                    q.reshape(length, q_channels),
                    k.reshape(length, kv_channels),
                )
            if mode == "saturated_logits":
                factor = saturated_calibration * (
                    1.0 if calibration else saturated_test_ratio
                )
                q, k = q * factor, k * factor
            v = _matrix(
                length,
                torch.ones(kv_channels),
                generator,
                heavy=mode in {"heavy_tail", "v_outlier"},
                holdout=holdout,
            )
            if mode == "v_outlier" and not calibration:
                probability = _uniform(
                    generator, 0.005, 0.02 if not holdout else 0.03
                )
                multiplier = _uniform(
                    generator, 7.0, 13.0 if not holdout else 17.0
                )
                outlier_mask = (
                    torch.rand(v.shape, generator=generator) < probability
                )
                v = torch.where(outlier_mask, v * multiplier, v)
            return {
                "q": _pair(q, device),
                "k": _pair(k, device),
                "v": _pair(v, device),
            }

        attention_cases.append(
            AttentionCase(
                f"{mode}_h{q_heads}_kv{kv_heads}_d{head_dim}_s{seq}",
                q_heads,
                kv_heads,
                head_dim,
                tuple(
                    sample(index, True)
                    for index in range(tier.calibration_samples)
                ),
                tuple(
                    sample(index, False)
                    for index in range(tier.test_samples)
                ),
            )
        )
    return EvaluationSuite(tuple(linear_cases), tuple(attention_cases))
