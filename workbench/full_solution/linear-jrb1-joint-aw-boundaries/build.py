"""Build L-JRB1 from the retained v202 Linear + v195 Attention complete root.

The activation mantissa is generated in exactly two live places
(``_dense_to_hif4``'s initial encode and ``_solve_exact_hierarchy``'s
per-candidate encode, including the edge-extension and L1 refinement solves),
and both live activation entry points (``hif4_dynamic_quantize_activation`` and
the v202 block-order wrapper ``_combined_dynamic_fast_reordered_gptq``) must
forward the learned table.  The append-only override pattern cannot replace a
function partially, so the table is threaded by exact-text patch on the copied
root.  Every patch asserts its anchor appears exactly once.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
EXPECTED_PARENT_SHA256 = (
    "56dc805d6e5a3ae f896db8021045740292735725d688b48e3d4393e55efcb2bd"
).replace(" ", "")

PATCHES = (
    (
        "solve-exact-hierarchy signature",
        """    sign: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    \"\"\"Exactly solve lv2/lv3 for fixed scales using three loss tables.""",
        """    sign: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    boundaries: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    \"\"\"Exactly solve lv2/lv3 for fixed scales using three loss tables.""",
    ),
    (
        "solve-exact-hierarchy mantissa",
        """        if group_gram is not None and sign is not None:
            mantissa = _adaround_mantissa(x_abs, local_scale, sign, group_gram)
        else:
            mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
            mantissa = mant_code * 0.25""",
        """        if group_gram is not None and sign is not None:
            mantissa = _adaround_mantissa(x_abs, local_scale, sign, group_gram)
        else:
            mantissa = _jrb1_boundary_mantissa(x_abs, local_scale, boundaries)""",
    ),
    (
        "dense-to-hif4 signature",
        """    max_refine_blocks: Optional[int] = None,
    full_sweep_top_k: int = 0,
) -> dict[str, torch.Tensor]:
    \"\"\"Quantize a dense tensor into valid HiF4 parameters.\"\"\"""",
        """    max_refine_blocks: Optional[int] = None,
    full_sweep_top_k: int = 0,
    boundaries: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    \"\"\"Quantize a dense tensor into valid HiF4 parameters.\"\"\"""",
    ),
    (
        "dense-to-hif4 mantissa",
        """    if group_gram is not None:
        mantissa = _adaround_mantissa(x_abs, denominator, sign, group_gram)
    else:
        mantissa = (
            torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
        )""",
        """    if group_gram is not None:
        mantissa = _adaround_mantissa(x_abs, denominator, sign, group_gram)
    else:
        mantissa = _jrb1_boundary_mantissa(x_abs, denominator, boundaries)""",
    ),
    (
        "dense-to-hif4 candidate solve",
        """        sign_expanded,
        gram_expanded,
    )""",
        """        sign_expanded,
        gram_expanded,
        boundaries=boundaries,
    )""",
    ),
    (
        "dense-to-hif4 edge extension",
        """                            else group_gram_hard.index_select(0, edge_indices)
                        ),
                    )
                )""",
        """                            else group_gram_hard.index_select(0, edge_indices)
                        ),
                        boundaries=boundaries,
                    )
                )""",
    ),
    (
        "dense-to-hif4 L1 solve",
        """                else group_gram_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1, -1
                )
            ),
        )""",
        """                else group_gram_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1, -1
                )
            ),
            boundaries=boundaries,
        )""",
    ),
    (
        "activation-gptq-quantize signature",
        """    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    \"\"\"GPTQ-style activation quantization with sequential block compensation.""",
        """    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
    boundaries: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    \"\"\"GPTQ-style activation quantization with sequential block compensation.""",
    ),
    (
        "activation-gptq-quantize single-block pass-through",
        """    if num_blocks < 2:
        return _dense_to_hif4(
            dense,
            importance=importance,
            group_gram=group_gram,
            search_offsets=search_offsets,
            error_threshold=error_threshold,
            accept_margin=accept_margin,
            max_refine_ratio=max_refine_ratio,
            max_refine_blocks=max_refine_blocks,
        )""",
        """    if num_blocks < 2:
        return _dense_to_hif4(
            dense,
            importance=importance,
            group_gram=group_gram,
            search_offsets=search_offsets,
            error_threshold=error_threshold,
            accept_margin=accept_margin,
            max_refine_ratio=max_refine_ratio,
            max_refine_blocks=max_refine_blocks,
            boundaries=boundaries,
        )""",
    ),
    (
        "activation-gptq-quantize per-block pass-through",
        """        block_result = _dense_to_hif4(
            X[..., start:end],
            importance=b_importance,
            group_gram=bg,
            search_offsets=search_offsets,
            error_threshold=error_threshold,
            accept_margin=accept_margin,
            max_refine_ratio=max_refine_ratio,
            max_refine_blocks=max_refine_blocks,
        )""",
        """        block_result = _dense_to_hif4(
            X[..., start:end],
            importance=b_importance,
            group_gram=bg,
            search_offsets=search_offsets,
            error_threshold=error_threshold,
            accept_margin=accept_margin,
            max_refine_ratio=max_refine_ratio,
            max_refine_blocks=max_refine_blocks,
            boundaries=boundaries,
        )""",
    ),
    (
        "dynamic-activation gptq entry",
        """        return _activation_gptq_quantize(
            dense,
            h_inv,
            importance=activation_state["importance"],
            group_gram=gram,
            search_offsets=activation_state["offsets"],
            error_threshold=float(activation_state["error_threshold"]),
            accept_margin=float(activation_state["accept_margin"]),
            max_refine_ratio=float(activation_state["max_refine_ratio"]),
            max_refine_blocks=int(activation_state["max_refine_blocks"]),
        )""",
        """        return _activation_gptq_quantize(
            dense,
            h_inv,
            importance=activation_state["importance"],
            group_gram=gram,
            search_offsets=activation_state["offsets"],
            error_threshold=float(activation_state["error_threshold"]),
            accept_margin=float(activation_state["accept_margin"]),
            max_refine_ratio=float(activation_state["max_refine_ratio"]),
            max_refine_blocks=int(activation_state["max_refine_blocks"]),
            boundaries=activation_state.get("activation_boundaries"),
        )""",
    ),
    (
        "dynamic-activation dense entry",
        """        return _dense_to_hif4(
            dense,
            importance=activation_state["importance"],
            group_gram=gram,
            search_offsets=activation_state["offsets"],
            error_threshold=float(activation_state["error_threshold"]),
            accept_margin=float(activation_state["accept_margin"]),
            max_refine_ratio=float(activation_state["max_refine_ratio"]),
            max_refine_blocks=int(activation_state["max_refine_blocks"]),
        )""",
        """        return _dense_to_hif4(
            dense,
            importance=activation_state["importance"],
            group_gram=gram,
            search_offsets=activation_state["offsets"],
            error_threshold=float(activation_state["error_threshold"]),
            accept_margin=float(activation_state["accept_margin"]),
            max_refine_ratio=float(activation_state["max_refine_ratio"]),
            max_refine_blocks=int(activation_state["max_refine_blocks"]),
            boundaries=activation_state.get("activation_boundaries"),
        )""",
    ),
    (
        "v202 block-order wrapper entry",
        """        gram_ordered = gram_ordered.index_select(0, order).unsqueeze(0).expand(
            int(dense.shape[0]), blocks, 8, 2, 4, 4
        )
    params = _activation_gptq_quantize(
        dense_ordered,
        h_inv_ordered,
        importance=importance,
        group_gram=gram_ordered,
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=int(state.get("max_refine_blocks", 0)),
    )""",
        """        gram_ordered = gram_ordered.index_select(0, order).unsqueeze(0).expand(
            int(dense.shape[0]), blocks, 8, 2, 4, 4
        )
    params = _activation_gptq_quantize(
        dense_ordered,
        h_inv_ordered,
        importance=importance,
        group_gram=gram_ordered,
        search_offsets=state.get("offsets"),
        error_threshold=float(state.get("error_threshold", 0.0)),
        accept_margin=float(state.get("accept_margin", 0.0)),
        max_refine_ratio=float(state.get("max_refine_ratio", 0.0)),
        max_refine_blocks=int(state.get("max_refine_blocks", 0)),
        boundaries=state.get("activation_boundaries"),
    )""",
    ),
)

actual_parent_sha256 = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if actual_parent_sha256 != EXPECTED_PARENT_SHA256:
    raise RuntimeError(
        f"expected retained root {EXPECTED_PARENT_SHA256}, "
        f"got {actual_parent_sha256}"
    )

source = PARENT.read_text(encoding="utf-8")
for name, old, new in PATCHES:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"patch anchor {name!r} matched {count} times, expected 1")
    source = source.replace(old, new)

implementation = (HERE / "implementation.py").read_text(encoding="utf-8")
CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source + "\n\n" + implementation, encoding="utf-8")

candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "linear-jrb1-joint-aw-boundaries",
    "version": "v227",
    "mechanism": "joint activation/weight deployed rounding boundaries (one coordinate-descent round)",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "patches": [name for name, _, _ in PATCHES],
    "fit_data": "all Linear calibration windows supplied by eval-v3, case-equal output MSE",
    "coordinate": "frozen parent deployed A/W pair; 6 sign-shared activation boundaries + 12 sign-split weight boundaries per layer",
    "solver": "exact g_x = E W_hat, h_x[j] = sum_i W_hat[i,j]^2 and L-RB1 exact quadratic; 64 buckets; one fixed-order round",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
