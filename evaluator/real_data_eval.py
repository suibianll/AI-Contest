"""Evaluate a HiF4 solution on real GPT-2 weights and activations."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
from reference_hif4 import (  # noqa: E402
    dequantize_hif4,
    dequantize_nvfp4,
    decode_standard_hif4,
    encode_standard_hif4,
    validate_state,
)


TEXT = (
    "The history of science is a history of measurement and precision. "
    "Modern language models learn to predict the next word from vast amounts of text. "
    "Attention mechanisms allow a model to focus on the most relevant parts of a sequence. "
    "Quantization reduces numerical precision to accelerate inference on specialized hardware. "
    "A four bit format stores each value using only four bits, trading range for speed. "
    "The transformer architecture has become the backbone of modern natural language processing. "
    "Neural networks are trained with gradient descent and backpropagation. "
    "The embedding layer maps discrete tokens into continuous vector representations. "
    "Layer normalization stabilizes training by normalizing activations across hidden units. "
    "Residual connections help gradients flow through deep networks during optimization. "
    "Softmax converts raw logits into a probability distribution over the vocabulary. "
    "Matrix multiplication is the fundamental operation in deep learning inference. "
    "Huawei's Ascend hardware provides efficient support for low precision computation. "
    "The calibration phase collects statistics from representative input data. "
    "Online quantization must be fast because it runs for every token during inference. "
    "A block scale is shared by a group of values to reduce metadata overhead. "
    "Outliers can dominate the dynamic range and reduce the accuracy of block quantization. "
    "Smoothing redistributes magnitude between activations and weights to improve precision. "
    "Permuting channels keeps exactly equivalent operations while improving quantization. "
    "The exact solver enumerates all valid exponent combinations for a given scale. "
    "Testing on held out data measures how well the quantization generalizes. "
    "The competition evaluates both linear layers and attention projections separately. "
    "Time limits require the quantization algorithm to be efficient as well as accurate. "
    "Small language models can still exhibit the same outlier patterns as large ones. "
    "Understanding the distribution of values is the key to effective quantization. "
    "Every experiment should be reproducible with a fixed random seed. "
    "The final score is the relative improvement over a standard baseline. "
    "Careful numerical analysis reveals why some scale choices outperform others. "
    "This paragraph provides diverse natural text for capturing real activation statistics. "
    "The quick brown fox jumps over the lazy dog while machines learn to reason. "
)

REQUIRED_FUNCTIONS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)

# Revised official panel limit (2026-08-29).  Historical reports may still
# mention the former 300-second limit, but active timing checks use 420s.
OFFICIAL_RUNTIME_LIMIT_SECONDS = 420.0


def load_solution(path: Path) -> ModuleType:
    source = path.resolve()
    spec = importlib.util.spec_from_file_location("_hif4_evaluated_solution", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = [
        name for name in REQUIRED_FUNCTIONS if not callable(getattr(module, name, None))
    ]
    if missing:
        raise AttributeError(f"solution is missing functions: {', '.join(missing)}")
    return module


def std_hif4(solution: ModuleType, dense: torch.Tensor) -> torch.Tensor:
    # The standard denominator must come from the frozen reference codec,
    # never from the candidate's `_dense_to_hif4`.
    del solution  # unused; kept for call-site compatibility
    params = encode_standard_hif4(dense)
    return decode_standard_hif4(params).to(torch.float32)


def causal_attention(
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    causal: bool = True,
) -> torch.Tensor:
    batch, tokens, _ = q_dense.shape
    q = q_dense.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    k = (
        k_dense.reshape(batch, tokens, kv_heads, head_dim)
        .transpose(1, 2)
        .repeat_interleave(group, dim=1)
    )
    v = (
        v_dense.reshape(batch, tokens, kv_heads, head_dim)
        .transpose(1, 2)
        .repeat_interleave(group, dim=1)
    )
    logits = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    if causal:
        mask = torch.triu(
            torch.full((tokens, tokens), float("-inf"), device=logits.device),
            diagonal=1,
        )
        logits = logits + mask
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities @ v).transpose(1, 2).reshape(
        batch, tokens, q_heads * head_dim
    )


def to_gqa_kv(
    dense: torch.Tensor, q_heads: int, kv_heads: int, head_dim: int
) -> torch.Tensor:
    group = q_heads // kv_heads
    return dense.reshape(-1, kv_heads, group, head_dim).mean(dim=2).reshape(
        -1, kv_heads * head_dim
    )


def collect_real_data(
    model_name_or_path: str,
    layers: int,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    device: str = "cpu",
    token_offset: int = 0,
    text: str | None = None,
):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model.eval().to(device)

    ids = tokenizer(
        TEXT if text is None else text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )["input_ids"][0]
    if ids.numel() == 0:
        raise ValueError("evaluation text tokenized to an empty sequence")
    offset = int(token_offset) % int(ids.numel())
    if offset:
        ids = torch.cat((ids[offset:], ids[:offset]))
    required_tokens = (calibration_samples + test_samples) * sequence_length
    if required_tokens > ids.numel():
        repeats = (required_tokens + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeats)
    ids = ids[:required_tokens]

    hidden = int(model.config.n_embd)
    q_heads = int(model.config.n_head)
    head_dim = hidden // q_heads
    blocks = model.transformer.h[:layers]
    if len(blocks) != layers:
        raise ValueError(
            f"model has {len(model.transformer.h)} layers; cannot evaluate {layers}"
        )

    weights = []
    for layer in blocks:
        attention_weight = layer.attn.c_attn.weight.detach().t().float()
        weights.append(
            {
                "q": attention_weight[:hidden].clone(),
                "k": attention_weight[hidden : 2 * hidden].clone(),
                "v": attention_weight[2 * hidden :].clone(),
                "o": layer.attn.c_proj.weight.detach().t().float(),
                "fc": layer.mlp.c_fc.weight.detach().t().float(),
                "proj": layer.mlp.c_proj.weight.detach().t().float(),
            }
        )

    def empty_capture():
        return {
            "act": {name: [] for name in ("q", "k", "v", "o", "fc", "proj")},
            "qkv": [],
        }

    calibration = empty_capture()
    tests = empty_capture()

    def capture(store, begin: int, end: int) -> None:
        captured = {index: {} for index in range(len(blocks))}
        handles = []

        def input_hook(index: int, key: str):
            def hook(_module, inputs, _output):
                captured[index][key] = inputs[0].detach().float()

            return hook

        def qkv_hook(index: int):
            def hook(_module, inputs, output):
                captured[index]["attn_in"] = inputs[0].detach().float()
                captured[index]["attn_raw"] = output.detach().float()

            return hook

        for index, layer in enumerate(blocks):
            handles.append(layer.attn.c_attn.register_forward_hook(qkv_hook(index)))
            handles.append(
                layer.attn.c_proj.register_forward_hook(
                    input_hook(index, "attn_proj_in")
                )
            )
            handles.append(
                layer.mlp.c_fc.register_forward_hook(input_hook(index, "fc_in"))
            )
            handles.append(
                layer.mlp.c_proj.register_forward_hook(input_hook(index, "proj_in"))
            )

        try:
            with torch.no_grad():
                for batch_index in range(begin, end):
                    start = batch_index * sequence_length
                    model(ids[start : start + sequence_length][None].to(device))
                    for index in range(len(blocks)):
                        flat = lambda value: value.reshape(-1, value.shape[-1])
                        attention_input = flat(captured[index]["attn_in"])
                        store["act"]["q"].append(attention_input)
                        store["act"]["k"].append(attention_input)
                        store["act"]["v"].append(attention_input)
                        store["act"]["o"].append(
                            flat(captured[index]["attn_proj_in"])
                        )
                        store["act"]["fc"].append(flat(captured[index]["fc_in"]))
                        store["act"]["proj"].append(
                            flat(captured[index]["proj_in"])
                        )
                        store["qkv"].append(flat(captured[index]["attn_raw"]))
        finally:
            for handle in handles:
                handle.remove()

    capture(calibration, 0, calibration_samples)
    capture(
        tests,
        calibration_samples,
        calibration_samples + test_samples,
    )
    return model, weights, calibration, tests, q_heads, head_dim


def score_linear(
    solution, weight_pair, activation_pairs, activation_state, weight_params
):
    scores = []
    weight_reference = dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_standard = std_hif4(solution, weight_reference)
    weight_player = dequantize_hif4(weight_params, weight_reference.shape)
    for activation_pair in activation_pairs:
        activation_reference = dequantize_nvfp4(*activation_pair).to(torch.float32)
        reference = activation_reference @ weight_reference.T
        standard = std_hif4(solution, activation_reference) @ weight_standard.T
        player_params = solution.hif4_dynamic_quantize_activation(
            *activation_pair, activation_state
        )
        player_activation = dequantize_hif4(
            player_params, activation_reference.shape
        )
        player = player_activation @ weight_player.T
        mse_standard = float((standard - reference).square().mean())
        mse_player = float((player - reference).square().mean())
        scores.append((mse_standard - mse_player) / mse_standard)
    return sum(scores) / len(scores)


def score_attention(
    solution,
    qkv_pairs,
    q_state,
    k_state,
    v_state,
    q_heads,
    kv_heads,
    head_dim,
    masks=("non-causal",),
):
    scores = {mask: [] for mask in masks}
    for q_pair, k_pair, v_pair in qkv_pairs:
        q_reference = dequantize_nvfp4(*q_pair).to(torch.float32)
        k_reference = dequantize_nvfp4(*k_pair).to(torch.float32)
        v_reference = dequantize_nvfp4(*v_pair).to(torch.float32)
        q_standard = std_hif4(solution, q_reference)
        k_standard = std_hif4(solution, k_reference)
        v_standard = std_hif4(solution, v_reference)
        q_params = solution.hif4_dynamic_quantize_q(
            *q_pair, q_heads, head_dim, q_state
        )
        k_params = solution.hif4_dynamic_quantize_k(
            *k_pair, kv_heads, head_dim, k_state
        )
        v_params = solution.hif4_dynamic_quantize_v(
            *v_pair, kv_heads, head_dim, v_state
        )
        q_player = dequantize_hif4(q_params, q_reference.shape)
        k_player = dequantize_hif4(k_params, k_reference.shape)
        v_player = dequantize_hif4(v_params, v_reference.shape)
        for mask in masks:
            causal = mask == "causal"
            reference = causal_attention(
                q_reference[None],
                k_reference[None],
                v_reference[None],
                q_heads,
                kv_heads,
                head_dim,
                causal,
            )
            standard = causal_attention(
                q_standard[None],
                k_standard[None],
                v_standard[None],
                q_heads,
                kv_heads,
                head_dim,
                causal,
            )
            player = causal_attention(
                q_player[None],
                k_player[None],
                v_player[None],
                q_heads,
                kv_heads,
                head_dim,
                causal,
            )
            mse_standard = float((standard - reference).square().mean())
            mse_player = float((player - reference).square().mean())
            scores[mask].append((mse_standard - mse_player) / mse_standard)
    return {
        mask: sum(values) / len(values) for mask, values in scores.items()
    }


API_NAMES = (
    "hif4_calibration_and_quantize_weight",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_activation",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


def instrument_solution(solution: ModuleType) -> dict:
    """Evaluator-side timing wrapper around the six formal APIs.

    Calibration may call a public dynamic API internally.  Only top-level
    evaluator calls contribute to the calibration/dynamic split; nested calls
    are recorded separately so their time is not counted twice.
    """

    stats = {
        "first_start": None,
        "last_end": None,
        "calibration": 0.0,
        "dynamic": 0.0,
        "calls": {name: 0 for name in API_NAMES},
        "nested_calls": {name: 0 for name in API_NAMES},
        "depth": 0,
    }
    for name in API_NAMES:
        original = getattr(solution, name)

        def wrapped(*args, _original=original, _name=name, **kwargs):
            top_level = stats["depth"] == 0
            start = time.perf_counter()
            if top_level and stats["first_start"] is None:
                stats["first_start"] = start
            stats["depth"] += 1
            try:
                return _original(*args, **kwargs)
            finally:
                end = time.perf_counter()
                stats["depth"] -= 1
                if top_level:
                    stats["last_end"] = end
                    if _name.startswith("hif4_calibration"):
                        stats["calibration"] += end - start
                    else:
                        stats["dynamic"] += end - start
                    stats["calls"][_name] += 1
                else:
                    stats["nested_calls"][_name] += 1

        setattr(solution, name, wrapped)
    return stats


def evaluate(args: argparse.Namespace) -> None:
    solution = load_solution(args.solution)
    stats = instrument_solution(solution)
    model, weights, calibration, tests, q_heads, head_dim = collect_real_data(
        args.model,
        args.layers,
        args.seq,
        args.calib,
        args.test,
        device=args.device,
        token_offset=args.token_offset,
    )
    layer_count = len(weights)
    hidden = int(model.config.n_embd)
    kv_heads = args.kv_heads if args.kv_heads is not None else q_heads
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(
            f"--kv-heads {kv_heads} must be a positive divisor of q_heads {q_heads}"
        )
    use_gqa = kv_heads != q_heads

    linear_scores = {
        name: [] for name in ("q", "k", "v", "o", "fc", "proj")
    }
    masks = tuple(
        {
            "causal": ("causal",),
            "non-causal": ("non-causal",),
            "both": ("causal", "non-causal"),
        }[args.attn_mask]
    )
    attention_scores = {mask: [] for mask in masks}
    features: list[tuple[str, bool, bool, bool, bool]] = []
    for layer_index in range(layer_count):
        for name in linear_scores:
            weight_pair = nvfp4_encode(weights[layer_index][name], args.mode)
            calibration_pairs = [
                nvfp4_encode(
                    calibration["act"][name][batch * layer_count + layer_index],
                    args.mode,
                )
                for batch in range(args.calib)
            ]
            calibrated = solution.hif4_calibration_and_quantize_weight(
                *weight_pair, calibration_pairs
            )
            if not isinstance(calibrated, dict) or not {
                "weight_params",
                "activation_state",
            }.issubset(calibrated):
                raise ValueError(
                    "hif4_calibration_and_quantize_weight must return "
                    "weight_params and activation_state"
                )
            validate_state(calibrated["activation_state"])
            test_pairs = [
                nvfp4_encode(
                    tests["act"][name][batch * layer_count + layer_index],
                    args.mode,
                )
                for batch in range(args.test)
            ]
            linear_scores[name].append(
                score_linear(
                    solution,
                    weight_pair,
                    test_pairs,
                    calibrated["activation_state"],
                    calibrated["weight_params"],
                )
            )

        qkv_calibration = []
        for batch in range(args.calib):
            dense = calibration["qkv"][batch * layer_count + layer_index].reshape(
                -1, 3 * hidden
            )
            q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
            if use_gqa:
                k_dense = to_gqa_kv(k_dense, q_heads, kv_heads, head_dim)
                v_dense = to_gqa_kv(v_dense, q_heads, kv_heads, head_dim)
            qkv_calibration.append(
                {
                    "q": nvfp4_encode(q_dense, args.mode),
                    "k": nvfp4_encode(k_dense, args.mode),
                    "v": nvfp4_encode(v_dense, args.mode),
                }
            )
        states = solution.hif4_calibration_attention(
            qkv_calibration, q_heads, kv_heads, head_dim
        )
        if not isinstance(states, dict) or not {
            "q_state",
            "k_state",
            "v_state",
        }.issubset(states):
            raise ValueError(
                "hif4_calibration_attention must return q_state, k_state, and v_state"
            )
        validate_state(states["q_state"])
        validate_state(states["k_state"])
        validate_state(states["v_state"])
        if args.verbose:
            # Telemetry (§12.1): derive per-layer selection rates from the
            # read-only state fields; never touches the solution itself.
            for side in ("q_state", "k_state"):
                layer_state = states[side]
                features.append(
                    (
                        side,
                        layer_state.get("multiplier") is not None,
                        layer_state.get("permutation") is not None,
                        layer_state.get("rotation") is not None,
                        int(layer_state.get("center_mode", 0) or 0) != 0,
                    )
                )
            features.append(
                (
                    "v_state",
                    False,
                    False,
                    False,
                    states["v_state"].get("importance") is not None,
                )
            )

        qkv_tests = []
        for batch in range(args.test):
            dense = tests["qkv"][batch * layer_count + layer_index].reshape(
                -1, 3 * hidden
            )
            q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
            if use_gqa:
                k_dense = to_gqa_kv(k_dense, q_heads, kv_heads, head_dim)
                v_dense = to_gqa_kv(v_dense, q_heads, kv_heads, head_dim)
            qkv_tests.append(
                (
                    nvfp4_encode(q_dense, args.mode),
                    nvfp4_encode(k_dense, args.mode),
                    nvfp4_encode(v_dense, args.mode),
                )
            )
        per_layer_attention = score_attention(
            solution,
            qkv_tests,
            states["q_state"],
            states["k_state"],
            states["v_state"],
            q_heads,
            kv_heads,
            head_dim,
            masks=masks,
        )
        for mask, value in per_layer_attention.items():
            attention_scores[mask].append(value)

    print(
        f"GPT-2 layers={layer_count} hidden={hidden} heads={q_heads}x{head_dim} "
        f"kv_heads={kv_heads} mode={args.mode} seq={args.seq} "
        f"calib={args.calib} test={args.test} attn_mask={args.attn_mask}"
    )
    print("Linear scores (mean across layers):")
    for name, values in linear_scores.items():
        print(
            f"  {name:5s} mean={sum(values) / len(values):.4f} "
            f"min={min(values):.4f} max={max(values):.4f}"
        )
    for mask, values in attention_scores.items():
        print(
            f"Attention[{mask}] mean={sum(values) / len(values):.4f} "
            f"min={min(values):.4f} max={max(values):.4f}"
        )
    linear_official_sum = sum(
        score * args.test for values in linear_scores.values() for score in values
    )
    attention_official_sum = sum(
        score * args.test for values in attention_scores.values() for score in values
    )
    print(
        "Official-flow score sum: "
        f"linear={linear_official_sum:.6f} "
        f"attention={attention_official_sum:.6f} "
        f"total={linear_official_sum + attention_official_sum:.6f}"
    )
    if args.verbose:
        print("Per-layer attention scores:")
        for mask, values in attention_scores.items():
            print(
                f"  [{mask}] "
                + " ".join(f"{value:.4f}" for value in values)
            )
        print("Per-layer linear scores:")
        for name, values in linear_scores.items():
            print(
                f"  {name:5s} "
                + " ".join(f"{value:.4f}" for value in values)
            )
        if features:
            print("State feature selection rates (per side, across layers):")
            for side in ("q_state", "k_state", "v_state"):
                rows = [f for f in features if f[0] == side]
                rate = lambda flag: sum(1 for r in rows if r[flag]) / len(rows)  # noqa: E731
                if side == "v_state":
                    print(
                        f"  {side}: importance={rate(4):.2f}"
                    )
                else:
                    print(
                        f"  {side}: multiplier={rate(1):.2f} "
                        f"permutation={rate(2):.2f} rotation={rate(3):.2f} "
                        f"centering={rate(4):.2f}"
                    )
    if stats["first_start"] is not None:
        stage = stats["last_end"] - stats["first_start"]
        api_total = stats["calibration"] + stats["dynamic"]
        print(
            f"Timing algorithm-stage={stage:.2f}s "
            f"calibration={stats['calibration']:.2f}s "
            f"dynamic={stats['dynamic']:.2f}s api-total={api_total:.2f}s"
        )
        if api_total >= OFFICIAL_RUNTIME_LIMIT_SECONDS:
            raise TimeoutError(
                f"official API total {api_total:.3f}s is not strictly below "
                f"{OFFICIAL_RUNTIME_LIMIT_SECONDS:.0f}s"
            )
        nested_total = sum(stats["nested_calls"].values())
        if nested_total:
            nested = " ".join(
                f"{name}={count}"
                for name, count in stats["nested_calls"].items()
                if count
            )
            print(f"Timing nested-api-calls={nested_total} ({nested})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "solution.py",
        help="solution.py to evaluate",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "gpt2"),
        help="GPT-2 model name or local model directory "
        "(default: bundled models/gpt2)",
    )
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--calib", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument(
        "--mode", default="amax6", choices=("amax6", "amax4", "pow2")
    )
    parser.add_argument(
        "--attn-mask",
        default="non-causal",
        choices=("causal", "non-causal", "both"),
        help="attention mask used for scoring (official-flow default: non-causal)",
    )
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="deterministic rotation of the built-in token stream for local holdout windows",
    )
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-layer scores for diagnostics",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for evaluation, e.g. cpu or cuda",
    )
    args = parser.parse_args(argv)
    if min(args.layers, args.seq, args.calib, args.test) <= 0:
        parser.error("--layers, --seq, --calib, and --test must be positive")
    if args.token_offset < 0:
        parser.error("--token-offset must be non-negative")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
