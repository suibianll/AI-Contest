"""Dense operand capture for the Qwen3.5-4B structure proxy panel.

The official evaluation model was user-confirmed as Qwen3.5-35B-A3B (hybrid
Gated DeltaNet + gated attention MoE).  Running 35B locally is not feasible,
so this capture builds a same-architecture-family proxy from ``Qwen3.5-4B``
(dense variant: identical hybrid layout, DeltaNet head geometry, gated
attention with head_dim 256 / partial RoPE 0.25, same tokenizer).

Panel design (``qwen35-4b-panel-v1``):

- 24 of 32 model layers: blocks {0, 1, 2, 4, 5, 7} of the repeating
  [DeltaNet, DeltaNet, DeltaNet, full-attention] layout -> 18 DeltaNet +
  6 full-attention layers.
- Panel order is shard-balanced: proxy-v3 strided shards (offset 6) each
  receive exactly one full-attention layer plus three DeltaNet layers.
- Seven uniform roles per layer.  Full-attention layers use q/k/v/o_proj.
  DeltaNet layers map ``q``/``k``/``v`` to the contiguous q/k/v row slices of
  ``in_proj_qkv`` and ``o`` to ``out_proj``.  ``in_proj_z``/``in_proj_b``/
  ``in_proj_a`` and ``conv1d`` are gating/depthwise ops, not Linear cases,
  and are excluded.
- Attention scenario operands exist for full-attention layers only: the
  official Q/K/V API contract assumes one (q_heads, kv_heads, head_dim)
  geometry per case, which DeltaNet (q/k 16x128 vs v 32x128) cannot express,
  and the scenario scores softmax attention.  DeltaNet q/k/v post-conv states
  are therefore not captured.
- Full-attention Q/K operands are post q_norm/k_norm and post partial RoPE,
  matching what enters attention; v is the raw v_proj output flattened in
  head-major order.  Linear role weights are raw module weights (q uses the
  q-half rows of the double-width ``q_proj``; the sigmoid gate half is not a
  role), so Linear cases score plain ``X @ W.T`` exactly like the Qwen2.5-0.5B
  panel.
- Storage dtype is float16: the forward runs in float16/bfloat16 and the
  evaluator's ``nvfp4_encode`` upcasts to float32 before quantising, so fp16
  storage is bit-identical to the fp32-widened Qwen2.5 convention at half the
  memory.
- Windows follow the proxy-v2 schedules.  Calibration activations are stored
  for windows {0, 1} (the two Linear calibration folds); test activations and
  attention Q/K/V are stored for the proxy-v3 compact pair windows
  {1, 2, 6, 7}.  Other slots are ``None`` by design; the v3 shard flow never
  reads them.  Full-panel (12-window) default runs on this cache are out of
  scope and will fail loudly if attempted.

Run (CPU is the supported default: the fp16 4B model is ~8 GiB, which does
not fit the local 8 GiB GPU — CUDA would silently spill into WDDM sysmem
fallback and eventually die with a raw ``CUDA error: out of memory``):

    .venv/Scripts/python.exe evaluator/capture_qwen35.py \
        --model models/qwen3.5-4b \
        --output artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt

On CPU the forward runs in bfloat16; stored operands are converted to
float16 on the fly (bf16 has 7 mantissa bits, fp16 has 10, so the cast is
exact for the value range; ``nvfp4_encode`` upcasts to float32 anyway).
``--device cuda`` remains available for GPUs with enough free VRAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

EVALUATOR_DIR = Path(__file__).resolve().parent
ROOT = EVALUATOR_DIR.parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

import official_eval as v2  # noqa: E402

PANEL_SCHEMA = "qwen35-heterogeneous-v1"
PANEL_NAME = "qwen35-4b-panel-v1"
MODEL_NAME = "qwen3.5-4b"
DEFAULT_MODEL_PATH = ROOT / "models" / MODEL_NAME
DEFAULT_OUTPUT = v2.CACHE_DIR / f"{MODEL_NAME}-{v2.PROTOCOL}.pt"

# 24-layer panel over the 32-layer hybrid model.  Blocks of the repeating
# [linear, linear, linear, full] pattern: blocks 3 and 6 are dropped.
KEPT_BLOCKS = (0, 1, 2, 4, 5, 7)
FULL_ATTENTION_INTERVAL = 4
# Panel positions of the six full-attention layers.  proxy-v3 shards are
# strided by 6 over the 24-layer panel, so one FA position per residue class
# gives every shard exactly one attention layer.
FA_PANEL_POSITIONS = (0, 1, 5, 8, 15, 22)

LINEAR_CALIBRATION_WINDOWS = (0, 1)
TEST_STORED_WINDOWS = (1, 2, 6, 7)  # proxy-v3 COMPACT_WINDOW_INDICES


def _panel_layer_plan(layer_types: list[str]) -> tuple[list[int], list[str], list[int]]:
    """Return (model layers in panel order, per-panel-layer types, FA panel indices)."""
    fa_model_layers: list[int] = []
    dn_model_layers: list[int] = []
    for index, layer_type in enumerate(layer_types):
        if index // FULL_ATTENTION_INTERVAL not in KEPT_BLOCKS:
            continue
        if layer_type == "full_attention":
            fa_model_layers.append(index)
        else:
            dn_model_layers.append(index)
    if len(fa_model_layers) != len(FA_PANEL_POSITIONS):
        raise RuntimeError(
            f"panel expects {len(FA_PANEL_POSITIONS)} full-attention layers, found {len(fa_model_layers)}"
        )
    total = len(dn_model_layers) + len(fa_model_layers)
    plan: list[int] = [0 for _ in range(total)]
    types: list[str] = [""] * total
    for position, model_layer in zip(FA_PANEL_POSITIONS, fa_model_layers):
        plan[position] = model_layer
        types[position] = "full_attention"
    dn_iter = iter(dn_model_layers)
    for position in range(total):
        if types[position] == "":
            plan[position] = next(dn_iter)
            types[position] = "linear_attention"
    fa_positions = [p for p, t in enumerate(types) if t == "full_attention"]
    return plan, types, fa_positions


def _load_text_model(model_path: Path, device: torch.device) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    model = _load_conditional_model(model_path, dtype)
    model.eval().to(device)
    core = getattr(model, "model", model)
    text = getattr(core, "language_model", core)
    if not hasattr(text, "layers") or not hasattr(text, "rotary_emb"):
        raise RuntimeError("cannot locate the Qwen3.5 text model (layers/rotary_emb missing)")
    return tokenizer, model, text


def _load_conditional_model(model_path: Path, dtype: torch.dtype) -> Any:
    # Prefer the explicit multimodal wrapper: the checkpoint keys live under
    # model.language_model.*, and AutoModelForCausalLM may otherwise resolve to
    # the text-only class and silently drop every weight.
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError:
        Qwen3_5ForConditionalGeneration = None  # type: ignore[assignment]
    if Qwen3_5ForConditionalGeneration is not None:
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=dtype
        )
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=dtype)


def _cpu16(value: torch.Tensor) -> torch.Tensor:
    """Move to CPU float16 storage (bf16 -> fp16 cast is exact for the range)."""
    return value.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _flat16(value: torch.Tensor) -> torch.Tensor:
    return _cpu16(value).reshape(-1, value.shape[-1])


def _weight16(module: torch.nn.Module) -> torch.Tensor:
    return _cpu16(module.weight)


def _q_half_weight(q_proj: torch.nn.Module, num_heads: int) -> torch.Tensor:
    """Extract the query-half rows of the double-width (value+gate) q_proj."""
    weight = _cpu16(q_proj.weight)
    per_head = weight.shape[0] // num_heads
    if per_head % 2 != 0:
        raise RuntimeError("q_proj output width is not head-paired")
    half = per_head // 2
    return (
        weight.view(num_heads, per_head, -1)[:, :half, :]
        .reshape(num_heads * half, weight.shape[1])
        .contiguous()
    )


def capture_pack(model_path: Path, device_name: str, output_path: Path) -> v2.RawPack:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    device = torch.device(device_name)
    tokenizer, model, text = _load_text_model(model_path, device)
    try:
        paths = {split: v2.DATA_DIR / filename for split, filename in v2.WIKITEXT_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing pinned WikiText files: " + ", ".join(missing))
        train_rows = v2._load_rows(paths["train"])
        validation_rows = v2._load_rows(paths["validation"])
        test_rows = v2._load_rows(paths["test"])
        calibration_windows = v2._select_calibration_windows(tokenizer, train_rows)
        test_windows = v2._select_test_windows(
            tokenizer, {"validation": validation_rows, "test": test_rows}
        )

        config = text.config
        hidden = int(config.hidden_size)
        q_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(config.head_dim)
        layer_types_config = list(config.layer_types)
        plan, panel_types, fa_positions = _panel_layer_plan(layer_types_config)
        fa_set = set(fa_positions)
        layers_count = len(plan)

        model_layers = list(text.layers)
        activations: dict[str, list[list[Any]]] = {
            role: [[None for _ in range(layers_count)] for _ in calibration_windows]
            for role in v2.ROLES
        }
        test_activations: dict[str, list[list[Any]]] = {
            role: [[None for _ in range(layers_count)] for _ in test_windows]
            for role in v2.ROLES
        }
        cal_qkv: list[list[Any]] = [[None for _ in range(layers_count)] for _ in calibration_windows]
        test_qkv: list[list[Any]] = [[None for _ in range(layers_count)] for _ in test_windows]
        weights: list[dict[str, torch.Tensor]] = [dict() for _ in range(layers_count)]
        weight_shapes: dict[str, dict[str, list[int]]] = {}

        captured: dict[str, Any] = {}
        rope: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        handles: list[Any] = []

        def _pre(key: str):
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs:
                    raise RuntimeError(f"{key} received no inputs")
                # Move to CPU immediately: per-index keys keep all panel layers
                # alive simultaneously (a shared key would be overwritten by the
                # next layer during the single full-model forward), and moving
                # in-hook keeps GPU residency at forward-transient levels only.
                captured[key] = _cpu16(inputs[0])

            return hook

        def _post(key: str):
            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], value: Any) -> None:
                captured[key] = _cpu16(value)

            return hook

        def register_layer(panel_index: int, layer: torch.nn.Module) -> None:
            prefix = f"{panel_index}:"
            if panel_index in fa_set:
                attn = layer.self_attn
                handles.append(attn.q_proj.register_forward_pre_hook(_pre(prefix + "hidden_in")))
                handles.append(attn.q_norm.register_forward_hook(_post(prefix + "qn")))
                handles.append(attn.k_norm.register_forward_hook(_post(prefix + "kn")))
                handles.append(attn.v_proj.register_forward_hook(_post(prefix + "vn")))
                handles.append(attn.o_proj.register_forward_pre_hook(_pre(prefix + "o_in")))
            else:
                lin = layer.linear_attn
                handles.append(lin.in_proj_qkv.register_forward_pre_hook(_pre(prefix + "hidden_in")))
                handles.append(lin.out_proj.register_forward_pre_hook(_pre(prefix + "o_in")))
            handles.append(layer.mlp.gate_proj.register_forward_pre_hook(_pre(prefix + "fc_gate_in")))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(_pre(prefix + "proj_in")))

        def rope_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], value: Any) -> None:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise RuntimeError("Qwen3.5 rotary hook returned an unexpected value")
            # Keep fp32 precision for cos/sin; RoPE is re-applied on CPU below.
            rope["value"] = (value[0].detach().cpu(), value[1].detach().cpu())

        handles.append(text.rotary_emb.register_forward_hook(rope_hook))
        for panel_index, model_layer_index in enumerate(plan):
            layer = model_layers[model_layer_index]
            register_layer(panel_index, layer)
            if panel_index in fa_set:
                weights[panel_index] = {
                    "q": _q_half_weight(layer.self_attn.q_proj, q_heads),
                    "k": _weight16(layer.self_attn.k_proj),
                    "v": _weight16(layer.self_attn.v_proj),
                    "o": _weight16(layer.self_attn.o_proj),
                    "fc_gate": _weight16(layer.mlp.gate_proj),
                    "fc_up": _weight16(layer.mlp.up_proj),
                    "proj": _weight16(layer.mlp.down_proj),
                }
            else:
                in_proj = layer.linear_attn.in_proj_qkv.weight
                key_dim = int(layer.linear_attn.key_dim)
                value_dim = int(layer.linear_attn.value_dim)
                weights[panel_index] = {
                    "q": _cpu16(in_proj[:key_dim, :]),
                    "k": _cpu16(in_proj[key_dim : 2 * key_dim, :]),
                    "v": _cpu16(in_proj[2 * key_dim : 2 * key_dim + value_dim, :]),
                    "o": _weight16(layer.linear_attn.out_proj),
                    "fc_gate": _weight16(layer.mlp.gate_proj),
                    "fc_up": _weight16(layer.mlp.up_proj),
                    "proj": _weight16(layer.mlp.down_proj),
                }
            weight_shapes[str(panel_index)] = {
                role: [int(tensor.shape[0]), int(tensor.shape[1])]
                for role, tensor in weights[panel_index].items()
            }

        def run_window(window: v2.Window, window_index: int, is_calibration: bool) -> None:
            if not is_calibration and window_index not in TEST_STORED_WINDOWS:
                # The forward still runs (deeper layers need the context), but
                # nothing is stored for this window.
                return
            captured.clear()
            rope.clear()
            input_ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            if "value" not in rope:
                raise RuntimeError("Qwen3.5 forward did not expose rotary embeddings")
            cos, sin = rope["value"]
            seq_len = len(window.input_ids)
            for panel_index, model_layer_index in enumerate(plan):
                layer = model_layers[model_layer_index]
                prefix = f"{panel_index}:"
                hidden_in = _flat16(captured[prefix + "hidden_in"])
                o_in = _flat16(captured[prefix + "o_in"])
                fc_gate_in = _flat16(captured[prefix + "fc_gate_in"])
                proj_in = _flat16(captured[prefix + "proj_in"])
                if panel_index in fa_set:
                    # Q/K are post q_norm/k_norm; partial RoPE is re-applied on
                    # CPU (cos/sin kept at fp32) and results stored as fp16.
                    q = captured[prefix + "qn"].transpose(1, 2)
                    k = captured[prefix + "kn"].transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    q = _flat16(q.transpose(1, 2).reshape(seq_len, -1))
                    k = _flat16(k.transpose(1, 2).reshape(seq_len, -1))
                    v = _flat16(captured[prefix + "vn"])
                    qkv = (q, k, v)
                else:
                    qkv = None
                role_tensors = {
                    "q": hidden_in,
                    "k": hidden_in,
                    "v": hidden_in,
                    "o": o_in,
                    "fc_gate": fc_gate_in,
                    "fc_up": fc_gate_in,
                    "proj": proj_in,
                }
                if is_calibration:
                    if window_index in LINEAR_CALIBRATION_WINDOWS:
                        for role, tensor in role_tensors.items():
                            activations[role][window_index][panel_index] = tensor
                    cal_qkv[window_index][panel_index] = qkv
                else:
                    if window_index in TEST_STORED_WINDOWS:
                        for role, tensor in role_tensors.items():
                            test_activations[role][window_index][panel_index] = tensor
                        test_qkv[window_index][panel_index] = qkv
                del role_tensors
            captured.clear()

        started = time.perf_counter()
        total_windows = len(calibration_windows) + len(test_windows)
        done = 0
        for window_index, window in enumerate(calibration_windows):
            run_window(window, window_index, is_calibration=True)
            done += 1
            print(
                f"[capture] window {done}/{total_windows} calib len={len(window.input_ids)}",
                flush=True,
            )
        for window_index, window in enumerate(test_windows):
            run_window(window, window_index, is_calibration=False)
            done += 1
            print(
                f"[capture] window {done}/{total_windows} test len={len(window.input_ids)}",
                flush=True,
            )
        elapsed = time.perf_counter() - started

        for handle in handles:
            handle.remove()

        metadata = {
            "protocol": v2.PROTOCOL,
            "panel_schema": PANEL_SCHEMA,
            "panel": PANEL_NAME,
            "model": MODEL_NAME,
            "model_path": str(model_path),
            "panel_design": (
                "24 of 32 layers, blocks {0,1,2,4,5,7} of [linear,linear,linear,full]x8; "
                "18 DeltaNet + 6 full-attention; shard-balanced order (one FA per strided shard)"
            ),
            "panel_model_layers": plan,
            "total_layers": layers_count,
            "layer_types": panel_types,
            "attention_layers": fa_positions,
            "weight_shapes": weight_shapes,
            "storage_dtype": "float16",
            "calibration_activation_windows": list(LINEAR_CALIBRATION_WINDOWS),
            "test_activation_windows": list(TEST_STORED_WINDOWS),
            "attention_qkv_note": (
                "full-attention layers only; DeltaNet q/k/v (16x128 qk vs 32x128 v) is not "
                "representable in the official Q/K/V API contract and the scenario scores "
                "softmax attention; in_proj_z/b/a and conv1d are not Linear roles"
            ),
            "linear_roles": list(v2.ROLES),
            "model_revision": "Qwen/Qwen3.5-4B@modelscope-download-2026-09-07",
            "dataset": "Salesforce/wikitext",
            "dataset_config": v2.WIKITEXT_CONFIG,
            "dataset_revision": v2.WIKITEXT_REVISION,
            "calibration_lengths": list(v2.CALIBRATION_LENGTHS),
            "test_length": v2.TEST_LENGTH,
            "test_lengths": list(v2.TEST_LENGTHS),
            "test_window_count": len(test_windows),
            "test_splits": sorted({window.split for window in test_windows}),
            "capture_device": str(device),
            "capture_seconds": elapsed,
            "weights_dtype": "float16 (exact forward outputs; nvfp4_encode upcasts to float32)",
            "weight_layout": "[out_features, in_features]",
            "input_codec": v2.NVFP4_INPUT_CODEC,
            "input_mode": v2.NVFP4_MODE,
            "data_sha256": {split: v2.sha256_file(path) for split, path in paths.items()},
            "geometry": {
                "hidden_size": hidden,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "full_attention_q_width": q_heads * head_dim,
                "kv_width": kv_heads * head_dim,
                "ffn_intermediate": int(weights[0]["fc_gate"].shape[0]),
                "deltanet_key_dim": int(model_layers[plan[2]].linear_attn.key_dim),
                "deltanet_value_dim": int(model_layers[plan[2]].linear_attn.value_dim),
            },
        }
        # Free the model before serialising: the pack plus the serialization
        # buffer otherwise peak at model RAM + pack RAM simultaneously.
        model = None
        text = None
        model_layers = None
        captured.clear()
        rope.clear()
        gc.collect()

        pack = v2.RawPack(
            weights, activations, test_activations, cal_qkv, test_qkv,
            calibration_windows, test_windows, layers_count, hidden,
            q_heads, kv_heads, head_dim, metadata,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        v2.save_pack(pack, output_path)
        sidecar = {
            "panel": PANEL_NAME,
            "panel_schema": PANEL_SCHEMA,
            "cache": str(output_path.resolve()),
            "total_layers": layers_count,
            "attention_layers": fa_positions,
            "layer_types": panel_types,
            "roles": list(v2.ROLES),
            "calibration_activation_windows": list(LINEAR_CALIBRATION_WINDOWS),
            "test_activation_windows": list(TEST_STORED_WINDOWS),
        }
        sidecar_path = output_path.with_suffix(output_path.suffix + ".panel.json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(
            f"[capture] saved {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB) "
            f"in {elapsed:.1f}s; sidecar {sidecar_path}",
            flush=True,
        )
        return pack
    finally:
        del model, text
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cpu",
        help="capture device; default cpu (bf16 forward) because the fp16 4B "
        "model exceeds the local 8 GiB GPU; use cuda only with enough free VRAM",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    capture_pack(args.model, args.device, args.output)


if __name__ == "__main__":
    main()
