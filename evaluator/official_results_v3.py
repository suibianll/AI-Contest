"""Versioned official-score manifest used by the proxy-v3 audit.

The hidden judge is not available locally, so these values are observations,
not labels for the proxy score.  The manifest is deliberately separate from
``official_eval.py``'s legacy archive table: the latter predates the current
new-weight runs and is kept as a compatibility backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import official_eval as v2
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name != "official_eval":
        raise
    from . import official_eval as v2


ROOT = Path(__file__).resolve().parents[1]


# Results added after the legacy manifest was frozen.  Keep paths repository
# relative so the audit output remains portable between checkouts.
CURRENT_OFFICIAL_RESULTS: tuple[dict[str, Any], ...] = (
    {
        "name": "v155",
        "path": "solutions/20260902_v155_l5a-permutation-stability_rejected/solution.py",
        "score": 16581,
        "time_seconds": 208.5,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v156",
        "path": "solutions/20260902_v156_l4-weight-decoupled_rejected/solution.py",
        "score": 16580,
        "time_seconds": 204.3,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v157",
        "path": "solutions/20260902_v157_v86-roab-only_rejected/solution.py",
        "score": 16729,
        "time_seconds": 218.96,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v159",
        "path": "solutions/20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA/solution.py",
        "score": 17532,
        "time_seconds": None,
        "status": "pass",
        "cohort": "new-weight",
        "source_note": "official score binds the submitted SHA; archive source is a research snapshot",
        "source_reproducibility": "unconfirmed",
    },
    {
        "name": "v160",
        "path": "solutions/20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA/solution.py",
        "score": 17532,
        "time_seconds": 232.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v162",
        "path": "solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py",
        "score": 1001,
        "time_seconds": 146.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v163",
        "path": "solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/solution.py",
        "score": 4587,
        "time_seconds": 202.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v164",
        "path": "solutions/20260903_v164_standard-linear_v160-attn_scoreNA_timeNA/solution.py",
        "score": 13945,
        "time_seconds": 204.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v166",
        "path": "solutions/20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA/solution.py",
        "score": 4590,
        "time_seconds": 226.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v168",
        "path": "solutions/20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA/solution.py",
        "score": 14005,
        "time_seconds": 210.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v171",
        "path": "solutions/20260903_v171_standard-linear_moment-threshold-attn_rejected/solution.py",
        "score": 13657,
        "time_seconds": 214.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v174",
        "path": "solutions/20260903_v174_kronecker-cat_standard-attn_rejected/solution.py",
        "score": 4508,
        "time_seconds": 190.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v175",
        "path": "solutions/20260903_v175_rank1-linear_logit-gain-attn_scoreNA_timeNA/solution.py",
        "score": 17594,
        "time_seconds": 245.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v176",
        "path": "solutions/20260903_v176_k-outlier-eq-attn_rejected/solution.py",
        "score": 13964,
        "time_seconds": 205.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v180",
        "path": "solutions/20260904_v180_a1-asym-fold-attn_scoreNA_timeNA/solution.py",
        "score": 17597,
        "time_seconds": 242.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v182",
        "path": "solutions/20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA/solution.py",
        "score": 17598,
        "time_seconds": 273.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v183",
        "path": "solutions/20260904_v183_attn-bsm-full-refine_rejected/solution.py",
        "score": 17598,
        "time_seconds": 279.7,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v185",
        "path": "solutions/20260904_v185_cleanroom-robust-operator_rejected/solution.py",
        "score": 8446,
        "time_seconds": 165.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v186",
        "path": "solutions/20260904_v186_attn-plus4-single-window_scoreNA_timeNA/solution.py",
        "score": 17599,
        "time_seconds": 272.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v187",
        "path": "solutions/20260904_v187_attn-jacobian-sensitivity_research-retained/solution.py",
        "score": 9167,
        "time_seconds": 169.0,
        "status": "pass",
        "cohort": "new-weight",
    },
    {
        "name": "v188",
        "path": "solutions/20260904_v188_attn-jacobian-port_rejected/solution.py",
        "score": 17595,
        "time_seconds": 268.0,
        "status": "pass",
        "cohort": "new-weight",
        "source_note": "official result recorded in the archived rejection report",
    },
)


def official_results(
    *,
    names: Iterable[str] | None = None,
    cohort: str | None = None,
    existing_only: bool = False,
) -> list[dict[str, Any]]:
    """Return a de-duplicated official-score manifest.

    ``official_eval.ARCHIVE_MANIFEST`` remains the source for the historical
    entries.  Current entries override a same-name legacy row, which makes a
    corrected score (for example v168) explicit and deterministic.
    """

    merged: dict[str, dict[str, Any]] = {}
    for name, item in v2.ARCHIVE_MANIFEST.items():
        score = item.get("official_score")
        if score is None:
            continue
        merged[name] = {
            "name": name,
            "path": str(item["path"]),
            "score": int(score),
            "time_seconds": item.get("official_time"),
            "status": item.get("official_status", "pass"),
            "cohort": item.get("official_cohort"),
        }
    for item in CURRENT_OFFICIAL_RESULTS:
        merged[str(item["name"])] = dict(item)

    wanted = None if names is None else {str(name) for name in names}
    rows = []
    for name in sorted(merged):
        item = dict(merged[name])
        if wanted is not None and name not in wanted:
            continue
        if cohort is not None and item.get("cohort") != cohort:
            continue
        source = (ROOT / str(item["path"])).resolve()
        item["source"] = str(source)
        item["source_exists"] = source.is_file()
        if existing_only and not item["source_exists"]:
            continue
        rows.append(item)
    return rows


def manifest_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    return {
        "count": len(values),
        "by_cohort": {
            cohort: sum(item.get("cohort") == cohort for item in values)
            for cohort in sorted({str(item.get("cohort")) for item in values})
        },
        "missing_sources": [item["name"] for item in values if not item.get("source_exists")],
        "names": [item["name"] for item in values],
    }
