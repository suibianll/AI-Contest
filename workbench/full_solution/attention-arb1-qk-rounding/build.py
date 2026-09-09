"""Build A-RB1 from the retained v202 + v195 complete root.

The Q/K mantissa is generated in exactly two places (``_dense_to_hif4``'s
initial encode and ``_solve_exact_hierarchy``'s per-candidate encode).  Both
are threaded with an explicit ``boundaries`` argument by exact-text patch on
the copied root, because the append-only override pattern cannot replace a
function partially.  Every patch asserts its anchor appears exactly once.
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
            mantissa = _arb1_boundary_mantissa(x_abs, local_scale, boundaries)""",
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
        mantissa = _arb1_boundary_mantissa(x_abs, denominator, boundaries)""",
    ),
    (
        "nvfp4-to-hif4 signature",
        """    learned_center: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant_float, scale_float)""",
        """    learned_center: Optional[torch.Tensor] = None,
    boundaries: Optional[torch.Tensor] = None,
    capture_dense: Optional[list] = None,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant_float, scale_float)""",
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
        "nvfp4-to-hif4 pass-through",
        """    params = _dense_to_hif4(
        dense,
        importance=refine_importance,
        group_gram=gram,
        source_scale_float=(scale_float if source_scale_proposal else None),
        search_offsets=search_offsets,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
    )""",
        """    if capture_dense is not None:
        capture_dense.append(dense)
    params = _dense_to_hif4(
        dense,
        importance=refine_importance,
        group_gram=gram,
        source_scale_float=(scale_float if source_scale_proposal else None),
        search_offsets=search_offsets,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
        boundaries=boundaries,
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
    "run_id": "attention-arb1-qk-rounding",
    "version": "v227",
    "mechanism": "Q/K joint softmax-output shared rounding boundaries",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "patches": [name for name, _, _ in PATCHES],
    "fit_data": "all attention calibration windows supplied by eval-v3, case-equal MSE",
    "coordinate": "frozen parent deployed Q/K; one sign-shared boundary per lower-code class",
    "solver": "one backward gradient + one fixed-seed Hutchinson probe, 64 buckets, analytic argmin, one joint deployed acceptance",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
