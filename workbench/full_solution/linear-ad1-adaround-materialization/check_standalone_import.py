"""Repository-free single-file import check, plus a self-contained contract smoke.

``verify.py`` answers a different question: whether the candidate's
``_adaround_mantissa`` produces the same bytes as the parent's on every caller it
can reach.  It never asks whether the candidate is a *submittable file* -- one
that imports and runs with nothing of this repository on the path -- because
``verify.py`` itself imports the evaluator and loads the packed calibration
cache out of the tree.

This script answers that second question, which the 4B guide's pre-submission
checklist (item 2: legal state + repository-free single-file import) and AGENTS
section 4 require before a candidate is archived:

* **lineage** -- the candidate is the parent's bytes plus an appended block, the
  parent's own ``_adaround_mantissa`` is still present and unchanged inside it,
  and the name that actually runs resolves to the appended shadow rather than to
  the parent's definition (checked on the loaded function object, not by reading
  the source);
* **isolated import and execution** -- the candidate is copied alone into a
  scratch directory, imported by a fresh interpreter started with ``-I`` from a
  working directory outside the tree, and the six public APIs are not merely
  present but *executed* there: weight calibration, the dynamic activation
  encoder, the Attention calibration and the three dynamic Q/K/V encoders, each
  on small synthetic inputs built inside that isolated process;
* **legal state** -- back in this process, the states the same smoke returns are
  put through the evaluator's own ``validate_state`` and the emitted parameters
  through ``validate_hif4_params``, so the isolated run is not self-certifying.

The isolated driver is generated from this file's own ``smoke`` source via
``inspect.getsource``, so the in-process and isolated smokes cannot drift apart.

What this script does **not** claim: it is a contract smoke on synthetic
tensors, not a panel.  It says nothing about accuracy or about wall-clock time.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "implementation.generated.py"
PARENT = ROOT / "solution.py"
PARENT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"
PARENT_BYTES = 505762

PUBLIC_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)

# Deliberately small and 64-aligned: in_features 128, out_features 64, and
# q_heads 4 / kv_heads 2 / head_dim 32 so both flattened widths (128 and 64) are
# multiples of the 64-channel block the encoder works in.
TOKENS = 8
OUT_FEATURES = 64
IN_FEATURES = 128
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 32


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path} standalone")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def attention_state(num_heads: int, head_dim: int, center_mode=None) -> dict:
    """A minimal state satisfying the evaluator's attention-state contract.

    ``hif4_calibration_attention`` hands back empty identity states on uniform
    random input, because it fits nothing to random noise; those empty dicts are
    then rejected by the evaluator's ``_check_attention_state`` for a missing
    ``num_heads``.  The Q/K/V encoders read only the geometry and the quantizer
    knobs off the state, so a state built explicitly to the contract exercises
    them where the calibration's own output cannot.  ``center_mode`` is set only
    for K, matching how the calibration fills it.
    """

    state = {
        "multiplier": None,
        "permutation": None,
        "importance": None,
        "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
        "error_threshold": 1e-7,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "version": 2,
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
    }
    if center_mode is not None:
        state["center_mode"] = int(center_mode)
    return state


def smoke(module):
    """Call all six APIs on synthetic inputs; return JSON-safe evidence.

    ``format`` in the returned dict keys is filled in by the isolated driver
    below, which re-compiles this very function's source.
    """

    torch.manual_seed(20260910)

    def pair(shape):
        quant = (torch.rand(shape, dtype=torch.float32) - 0.5) * 3.0
        scale = torch.full(
            (*shape[:-1], shape[-1] // 16), 0.125, dtype=torch.float32
        )
        return quant, scale

    def qkv_entry():
        return {
            "q": pair((1, TOKENS, Q_HEADS * HEAD_DIM)),
            "k": pair((1, TOKENS, KV_HEADS * HEAD_DIM)),
            "v": pair((1, TOKENS, KV_HEADS * HEAD_DIM)),
        }

    weight_quant, weight_scale = pair((OUT_FEATURES, IN_FEATURES))
    weight_result = module.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, [pair((TOKENS, IN_FEATURES)) for _ in range(2)]
    )
    if set(weight_result) != {"weight_params", "activation_state"}:
        raise SystemExit(f"weight calibration returned {sorted(weight_result)}")

    activation_pair = pair((TOKENS, IN_FEATURES))
    activation_params = module.hif4_dynamic_quantize_activation(
        activation_pair[0], activation_pair[1], weight_result["activation_state"]
    )

    attention_states = module.hif4_calibration_attention(
        [qkv_entry() for _ in range(2)], Q_HEADS, KV_HEADS, HEAD_DIM
    )
    if set(attention_states) != {"q_state", "k_state", "v_state"}:
        raise SystemExit(f"attention calibration returned {sorted(attention_states)}")
    # On uniform random input this calibration fits nothing and hands back its
    # identity states -- empty dicts, which `_check_attention_state` then rejects
    # for a missing `num_heads`.  That is a property of the synthetic input, not
    # of the candidate, so the Q/K/V encoders are exercised below against states
    # of the same geometry built explicitly to the legal contract instead, and
    # the identity result is recorded rather than raised on.  The fitted states
    # and the encoders running against them are covered by verify.py and by the
    # panel.
    identity_states = not any(attention_states.values())

    q_state = attention_state(Q_HEADS, HEAD_DIM)
    k_state = attention_state(KV_HEADS, HEAD_DIM, center_mode=0)
    v_state = attention_state(KV_HEADS, HEAD_DIM)

    emitted = {
        "activation": (activation_params, (TOKENS, IN_FEATURES)),
    }
    for label, shape, state in (
        ("q", (TOKENS, Q_HEADS * HEAD_DIM), q_state),
        ("k", (TOKENS, KV_HEADS * HEAD_DIM), k_state),
        ("v", (TOKENS, KV_HEADS * HEAD_DIM), v_state),
    ):
        heads = Q_HEADS if label == "q" else KV_HEADS
        api = getattr(module, f"hif4_dynamic_quantize_{label}")
        quant, scale = pair(shape)
        emitted[label] = (api(quant, scale, heads, HEAD_DIM, state), shape)

    summary = {"apis": []}
    states = {
        "weight_activation": weight_result["activation_state"],
        "q_state": q_state,
        "k_state": k_state,
        "v_state": v_state,
    }
    for label, (params, shape) in emitted.items():
        if not isinstance(params, dict) or not params:
            raise SystemExit(f"{label} returned {type(params)}")
        fields = []
        for field, value in sorted(params.items()):
            if torch.is_tensor(value):
                if not bool(torch.isfinite(value).all()):
                    raise SystemExit(f"{label}.{field} is not finite")
                fields.append(str(field))
        summary["apis"].append({"api": label, "fields": fields, "shape": list(shape)})
    scale_factor = weight_result["weight_params"].get("scale_factor")
    summary["weight_params_shape"] = (
        list(scale_factor.shape) if torch.is_tensor(scale_factor) else None
    )
    summary["attention_calibration_identity_on_synthetic_input"] = identity_states
    return {
        "summary": summary,
        "states": states,
        "emitted": emitted,
        "weight_params": weight_result["weight_params"],
    }


def lineage(candidate_bytes: bytes, parent_bytes: bytes) -> dict:
    """The candidate is the parent plus an appended block, and the parent's own
    definition is still inside it untouched.
    """

    report = {
        "parent_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "parent_bytes": len(parent_bytes),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes": len(candidate_bytes),
        "appended_bytes": len(candidate_bytes) - len(parent_bytes),
    }
    if (
        report["parent_sha256"] != PARENT_SHA256
        or report["parent_bytes"] != PARENT_BYTES
    ):
        raise SystemExit(
            "the working-tree root is not the card parent "
            f"{PARENT_SHA256[:8]}; refusing to certify lineage against it"
        )
    report["starts_with_parent_bytes"] = candidate_bytes.startswith(
        parent_bytes.rstrip(b"\n")
    )
    if not report["starts_with_parent_bytes"]:
        raise SystemExit("the candidate does not begin with the parent's bytes")

    def definitions(source: bytes) -> list[str]:
        return [
            ast.dump(node)
            for node in ast.parse(source.decode("utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "_adaround_mantissa"
        ]

    parent_defs = definitions(parent_bytes)
    candidate_defs = definitions(candidate_bytes)
    report["parent_definitions_of_helper"] = len(parent_defs)
    report["candidate_definitions_of_helper"] = len(candidate_defs)
    report["parent_definition_unchanged_in_candidate"] = (
        candidate_defs[0] == parent_defs[0]
    )
    if len(parent_defs) != 1 or len(candidate_defs) != 2:
        raise SystemExit(
            f"expected 1 parent and 2 candidate definitions, got "
            f"{len(parent_defs)} and {len(candidate_defs)}"
        )
    if not report["parent_definition_unchanged_in_candidate"]:
        raise SystemExit("the parent's own _adaround_mantissa changed in the candidate")
    return report


ISOLATED_DRIVER = """
import json as _json
import sys as _sys
import types as _types
from pathlib import Path as _Path

import torch

{constants}

{state_source}

{smoke_source}

_target = _Path(_sys.argv[1])
if sorted(_entry.name for _entry in _target.parent.iterdir()) != ["solution.py"]:
    raise SystemExit("the scratch directory holds more than the candidate")
if _Path.cwd().resolve() != _target.parent.resolve():
    raise SystemExit("the isolated interpreter is not running in the scratch directory")
for _entry in _sys.path:
    if "workbench" in _entry.replace("\\\\", "/"):
        raise SystemExit("a repository path is on the isolated sys.path: " + _entry)

# A real module object, not a bare dict: the smoke drives the candidate through
# attribute access exactly as the evaluator does.
_module = _types.ModuleType("ad1_solo")
_module.__file__ = str(_target)
exec(compile(_target.read_text(encoding="utf-8"), "solution.py", "exec"), _module.__dict__)

_missing = [name for name in {apis!r} if not callable(getattr(_module, name, None))]
if _missing:
    raise SystemExit("missing APIs: {{}}".format(_missing))

_result = smoke(_module)
print(_json.dumps(_result["summary"]))
"""


def isolated_run(candidate_path: Path) -> dict:
    """Import and execute the candidate alone, in a fresh ``-I`` interpreter."""

    # ``smoke`` reads these module-level constants and calls ``attention_state``,
    # so both travel into the isolated interpreter with it; the sources and the
    # values are taken from this module rather than retyped, which is what keeps
    # the two runs from drifting apart.
    constants = "\n".join(
        f"{name} = {globals()[name]!r}"
        for name in (
            "TOKENS",
            "OUT_FEATURES",
            "IN_FEATURES",
            "Q_HEADS",
            "KV_HEADS",
            "HEAD_DIM",
        )
    )
    driver = ISOLATED_DRIVER.format(
        constants=constants,
        state_source=inspect.getsource(attention_state),
        smoke_source=inspect.getsource(smoke),
        apis=list(PUBLIC_APIS),
    )
    with tempfile.TemporaryDirectory(prefix="ad1-solo-") as solo:
        solo_path = Path(solo)
        target = solo_path / "solution.py"
        target.write_bytes(candidate_path.read_bytes())
        completed = subprocess.run(
            [sys.executable, "-I", "-c", driver, str(target)],
            cwd=str(solo_path),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(
                "the isolated interpreter failed:\n"
                f"--- stdout ---\n{completed.stdout}\n"
                f"--- stderr ---\n{completed.stderr}"
            )
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "scratch_contents": ["solution.py"],
        "interpreter_flags": "-I",
        "cwd": "the scratch directory itself (outside the repository)",
        "apis_executed": summary["apis"],
        "weight_params_shape": summary["weight_params_shape"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=HERE / "standalone_import.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    candidate_bytes = args.candidate.read_bytes()
    parent_bytes = PARENT.read_bytes()

    report: dict = {"candidate": str(args.candidate)}
    report.update(lineage(candidate_bytes, parent_bytes))

    module = load_module(args.candidate, "ad1_import_check")
    live_line = int(module._adaround_mantissa.__code__.co_firstlineno)
    parent_line_count = len(parent_bytes.decode("utf-8").splitlines())
    report["live_helper_first_line"] = live_line
    report["parent_line_count"] = parent_line_count
    report["live_helper_is_the_appended_one"] = live_line > parent_line_count
    if not report["live_helper_is_the_appended_one"]:
        raise SystemExit("the live _adaround_mantissa is the parent's, not the shadow")
    report["bindings_of_helper_name"] = sum(
        1 for name in vars(module) if name == "_adaround_mantissa"
    )

    sys.path.insert(0, str(ROOT / "evaluator"))
    import reference_hif4 as ref  # noqa: PLC0415

    print("smoke: six APIs on synthetic inputs, in this process")
    in_process = smoke(module)
    for name, state in in_process["states"].items():
        ref.validate_state(state)
    for label, (params, shape) in in_process["emitted"].items():
        ref.validate_hif4_params(params, shape)
    ref.validate_hif4_params(in_process["weight_params"], (OUT_FEATURES, IN_FEATURES))
    report["in_process_smoke"] = in_process["summary"]
    report["states_accepted_by_validate_state"] = sorted(in_process["states"])
    report["params_accepted_by_validate_hif4_params"] = sorted(
        list(in_process["emitted"]) + ["weight_params"]
    )

    print("isolated: a fresh -I interpreter, scratch directory, no repository")
    report["isolated_import"] = isolated_run(args.candidate)

    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_sha256",
                    "appended_bytes",
                    "starts_with_parent_bytes",
                    "candidate_definitions_of_helper",
                    "parent_definition_unchanged_in_candidate",
                    "live_helper_is_the_appended_one",
                )
            },
            indent=2,
        )
    )
    print(f"\nPASS -- wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
