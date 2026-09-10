"""Equivalence evidence for L-AD1, for every caller of the shared helper.

The card rewrites the middle of ``_adaround_mantissa``, which the weight and
static paths call as well as the activation path, so "the activation path still
produces the same bytes" is not the claim.  The claim is that the helper itself
returns bit-identical values for every input, and that each caller reaches it
with the same inputs.  Both are checked here.

Why the rewrite is exact, before any measurement.  The only thing that moved is
the position of two elementwise operations relative to one ``torch.where``:

    parent     where(m, ceil_code, floor_code).to(float32) * 0.25
    candidate  where(m, ceil_code.to(float32) * 0.25, floor_code.to(float32) * 0.25)

The two int64 code expressions are unchanged statement for statement.  Both
``.to(torch.float32)`` and ``* 0.25`` are elementwise, and an elementwise
function commutes with ``where`` by definition -- the selection picks one value
per position and the function is then applied to that value -- so the two forms
agree for every int64 pair and every mask, including the ones a non-finite input
produces, with no exactness assumption about the conversion.  The control set
below is what keeps that argument honest against implementation surprises
(dtype promotion, non-contiguous layouts, the device-keyed mask cache).

Three parts:

  unit       parent vs candidate ``_adaround_mantissa`` on synthetic inputs
             across shapes 1-D to 5-D, four dtypes, non-contiguous layouts and
             a value set that includes the clamp boundaries, exact codes, and
             non-finite values.  Compared by bytes, never by tolerance.
  activation the real ``hif4_dynamic_quantize_activation`` on real cached states
             (a Gram layer and a non-Gram layer, two row counts), all five HiF4
             fields compared field by field.
  weight     the real ``hif4_calibration_and_quantize_weight`` on a real layer,
             which is the path that reaches ``_solve_exact_hierarchy`` through
             ``_refine_weight_blocks64`` and the JDRQ refiners -- the callers
             that have nothing to do with activations.  The helper's call count
             is recorded on both sides as well: a change that altered control
             flow would show up there even if the values happened to match.

Read-only with respect to the repo: writes ``verify.json`` next to itself.  No
timing is taken here; see ``timing.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "implementation.generated.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")
TARGETS = ((0, "q"), (0, "o"))
ROW_COUNTS = (128, 512)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bits(value) -> tuple:
    """A tensor's identity down to the byte; never ``torch.equal``, never a
    tolerance -- ``torch.equal`` reports NaN as unequal to itself and would
    hide a real difference behind a false one.

    The comparison is over the raw memory bytes (a ``uint8`` view), not over
    ``.numpy()``: bfloat16 has no numpy dtype, and a byte view also compares NaN
    payloads, which is the strictest form available.
    """
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return (
            "tensor",
            str(tensor.dtype),
            tuple(int(v) for v in tensor.shape),
            raw,
        )
    return ("other", type(value).__name__, repr(value).encode("utf-8"))


def compare(label: str, left, right, report: dict, key: str) -> bool:
    same = bits(left) == bits(right)
    report[key].append({"label": label, "identical": same})
    if not same:
        shape_l = getattr(left, "shape", None)
        shape_r = getattr(right, "shape", None)
        print(f"  DIFFERS: {label}  {shape_l} vs {shape_r}")
    return same


def mapping_bits(mapping) -> dict:
    if not isinstance(mapping, dict):
        return {"<value>": bits(mapping)}
    return {str(name): bits(value) for name, value in sorted(mapping.items())}


# --------------------------------------------------------------------------
# unit: parent vs candidate _adaround_mantissa
# --------------------------------------------------------------------------

SHAPES = ((4,), (5, 4), (3, 7, 4), (2, 3, 8, 2, 4), (6, 509, 8, 2, 4))
DTYPES = (torch.float32, torch.float64, torch.bfloat16, torch.float16)


def value_sets(shape, dtype: torch.dtype, device, generator):
    """Inputs chosen for the edges of the rewrite, not for generality.

    Everything downstream is decided by ``raw_code = x_abs * (4.0 / local_scale)``,
    so the raw codes are what these sets target: exact integers (``floor == ceil``,
    where the two candidates coincide), the 6/7 asymmetry between the two clamps,
    values whose codes fall below 0 and above 7, and the three non-finite cases.
    """

    def full(value):
        return torch.full(shape, value, dtype=torch.float64).to(
            dtype=dtype, device=device
        )

    limit = torch.finfo(dtype).max

    yield "random", torch.rand(
        shape, generator=generator, dtype=torch.float64
    ).to(dtype=dtype, device=device)
    yield "zeros", full(0.0)
    yield "huge", full(1e30)
    yield "tiny", full(1e-30)
    yield "negative", full(-1.0)
    yield "above7", full(7.5)
    yield "inf", full(float("inf"))
    yield "nan", full(float("nan"))
    mixed = full(0.0)
    flat = mixed.reshape(-1)
    for index, value in enumerate(
        (float("inf"), float("nan"), -1.0, 1e30, 0.0, 7.5, 1.75, 1e-30)
    ):
        if index >= flat.numel():
            break
        # In-place assignment, unlike a cast, refuses to overflow -- so the
        # finite values are clipped to the dtype's range instead of silently
        # becoming inf.  The infinities and the NaN are assignable as they are.
        flat[index] = (
            max(-limit, min(limit, value)) if math.isfinite(value) else value
        )
    yield "mixed", mixed


def exact_code_set(shape, dtype, device):
    """``x_abs`` values whose raw code at unit scale is exactly an integer 0-7.

    Unreachable through a random draw, and it is the only set where the floor
    and the ceil candidate coincide -- including at code 7, where the floor
    clamp is 6 and the ceil clamp is 7.
    """
    count = _numel(shape)
    codes = (torch.arange(count, dtype=torch.float64) % 8) / 4.0
    return codes.reshape(shape).to(dtype=dtype, device=device)


def unit_part(parent, candidate, device, report: dict) -> int:
    failures = 0
    total = 0
    generator = torch.Generator().manual_seed(20260910)
    for shape in SHAPES:
        for dtype in DTYPES:
            sets = list(value_sets(shape, dtype, device, generator))
            sets.append(("exact_codes", exact_code_set(shape, dtype, device)))
            for name, x_abs in sets:
                # The scale, sign and Gram live in the compute dtype, not in
                # ``x_abs``'s: float32 for the half and single-precision cases,
                # which is what the real callers pass, and float64 only where
                # the operand is float64.  Building them in the operand's dtype
                # instead leaves half of this set raising on both sides (the
                # parent rejects a half-precision scale, and ``einsum`` refuses
                # mixed operand dtypes), which compares nothing.
                compute_dtype = (
                    torch.float64 if dtype == torch.float64 else torch.float32
                )
                for scale_name, local_scale in (
                    (
                        "unit",
                        torch.tensor(1.0, dtype=compute_dtype, device=device),
                    ),
                    (
                        "vector",
                        torch.linspace(
                            0.25, 4.0, 4, dtype=compute_dtype, device=device
                        ),
                    ),
                ):
                    sign = (
                        torch.where(
                            torch.arange(4, device=device) % 2 == 0, 1.0, -1.0
                        )
                        .to(compute_dtype)
                        .expand(x_abs.shape)
                    )
                    gram = torch.eye(4, dtype=compute_dtype, device=device).expand(
                        *x_abs.shape[:-1], 4, 4
                    )
                    label = f"{shape} {dtype} {name}/{scale_name}"
                    left = right = None
                    errors = {}
                    for side, module in (("parent", parent), ("candidate", candidate)):
                        try:
                            value = module._adaround_mantissa(
                                x_abs, local_scale, sign, gram
                            )
                        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                            errors[side] = f"{type(exc).__name__}: {exc}"
                            value = None
                        if side == "parent":
                            left = value
                        else:
                            right = value
                    if errors:
                        # A raise on both sides is the same behaviour; a raise on
                        # one side only is a difference and is counted as one.
                        report["unit_errors"].append({"label": label, **errors})
                        if set(errors) != {"parent", "candidate"}:
                            failures += 1
                        continue
                    total += 1
                    if not compare(f"unit {label}", left, right, report, "unit"):
                        failures += 1
    # Non-contiguous inputs: a transposed view, a strided slice and a stride-0
    # expansion.  The 16x tensor inherits the layout of its operands, so a
    # layout-sensitive rewrite would show up here and nowhere else.
    base = torch.rand((8, 2, 4, 8), dtype=torch.float32, device=device)
    layouts = {
        # last dimension keeps size 4 -- it is the subgroup -- but arrives with
        # a stride of 2 rather than 1
        "sliced": base[:, :, :, 1::2],
        "transposed": base.permute(3, 2, 1, 0)[..., :4],
        "expanded": torch.rand((1, 8, 2, 1), dtype=torch.float32, device=device).expand(
            6, 8, 2, 4
        ),
    }
    for name, x_abs in layouts.items():
        local_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        sign = torch.ones_like(x_abs)
        gram = torch.eye(4, dtype=torch.float32, device=device).expand(
            *x_abs.shape[:-1], 4, 4
        )
        left = parent._adaround_mantissa(x_abs, local_scale, sign, gram)
        right = candidate._adaround_mantissa(x_abs, local_scale, sign, gram)
        total += 1
        if not compare(f"unit layout {name}", left, right, report, "unit"):
            failures += 1
    report["unit_cases"] = total
    report["unit_failures"] = failures
    return failures


def _numel(shape) -> int:
    product = 1
    for value in shape:
        product *= value
    return product


# --------------------------------------------------------------------------
# activation: the real dynamic path
# --------------------------------------------------------------------------


def parent_sha_prefix() -> str:
    """The calibration cache is keyed by the solution's own sha256.

    Derived from the parent file rather than transcribed, so that rebasing the
    card onto a new root cannot leave a stale cache key behind: a hardcoded
    prefix silently returns the *previous* root's calibrated states, which would
    make the activation comparison compare two different states and pass.
    """
    return hashlib.sha256(PARENT.read_bytes()).hexdigest()[:16]


def find_state(layer: int, role: str) -> dict:
    prefix = parent_sha_prefix()
    cache = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
    for path in sorted(cache.glob(f"{prefix}-linear-*.pt")):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for entry in payload["weight_states"]:
            if int(entry["layer"]) == layer and str(entry["role"]) == role:
                del payload
                return dict(entry["state"])
    raise SystemExit(f"no cached state for layer {layer} / role {role}")


def activation_part(parent, candidate, evaluator, pack, device, report: dict) -> int:
    failures = 0
    raw = pack["test_activations"]
    for layer, role in TARGETS:
        state = find_state(layer, role)
        base_quant, base_scale = evaluator._pair(raw[role][1][layer].to(torch.float32))
        for rows in ROW_COUNTS:
            repeat = -(-rows // base_quant.shape[0])
            quant = base_quant.repeat(repeat, 1)[:rows].contiguous().to(device)
            scale = base_scale.repeat(repeat, 1)[:rows].contiguous().to(device)
            left = parent.hif4_dynamic_quantize_activation(quant, scale, dict(state))
            right = candidate.hif4_dynamic_quantize_activation(quant, scale, dict(state))
            label = f"activation layer{layer}/{role} rows={rows}"
            same = mapping_bits(left) == mapping_bits(right)
            missing = set(FIELDS) - set(left)
            if missing:
                raise SystemExit(f"{label}: missing fields {sorted(missing)}")
            report["activation"].append(
                {
                    "label": label,
                    "channels": int(state["in_features"]),
                    "has_gram": state.get("gram") is not None,
                    "fields": {
                        name: bits(left[name]) == bits(right[name]) for name in FIELDS
                    },
                    "identical": same,
                }
            )
            if not same:
                failures += 1
                print(f"  DIFFERS: {label}")
            else:
                print(
                    f"  {label}: 5/5 fields bit-identical "
                    f"(channels={int(state['in_features'])}, "
                    f"gram={state.get('gram') is not None})"
                )
    return failures


# --------------------------------------------------------------------------
# weight: the callers that have nothing to do with activations
# --------------------------------------------------------------------------


def weight_part(parent, candidate, evaluator, pack, device, report: dict) -> int:
    """Runs the real weight calibration for a few layer/role states.

    ``pack.weights[layer][role]`` and the packed calibration activations are used
    exactly as ``proxy_v3_eval._calibrate`` uses them, and the whole returned
    mapping is compared -- the weight parameters and the activation state the
    weight path derives from it.
    """
    indices = tuple(range(min(2, len(pack["calibration_windows"]))))
    keys = [(0, "q"), (0, "o")]
    failures = 0
    for layer, role in keys:
        # Same construction as ``proxy_v3_eval._calibrate``: the raw tensors are
        # nvfp4-encoded with the evaluator's own ``_pair`` and moved to the
        # device, so the API sees the inputs it sees in a real run.
        weight_pair = evaluator._move_pair(
            evaluator._pair(pack["weights"][layer][role]), device
        )
        calibration = [
            evaluator._move_pair(
                evaluator._pair(pack["calibration_activations"][role][sample][layer]),
                device,
            )
            for sample in indices
        ]
        counts = {}
        for name, module in (("parent", parent), ("candidate", candidate)):
            calls = {"all": 0, "with_gram": 0}
            original = module._adaround_mantissa

            def counter(*args, _calls=calls, _original=original, **kwargs):
                _calls["all"] += 1
                gram = args[3] if len(args) > 3 else kwargs.get("group_gram")
                if gram is not None:
                    _calls["with_gram"] += 1
                return _original(*args, **kwargs)

            module._adaround_mantissa = counter
            try:
                result = module.hif4_calibration_and_quantize_weight(
                    weight_pair[0], weight_pair[1], calibration
                )
            finally:
                module._adaround_mantissa = original
            counts[name] = calls
            if name == "parent":
                left = result
            else:
                right = result
        label = f"weight layer{layer}/{role}"
        left_fields = mapping_bits(left["weight_params"])
        right_fields = mapping_bits(right["weight_params"])
        state_same = mapping_bits(left["activation_state"]) == mapping_bits(
            right["activation_state"]
        )
        params_same = left_fields == right_fields
        call_same = counts["parent"] == counts["candidate"]
        report["weight"].append(
            {
                "label": label,
                "calibration_samples": list(indices),
                "fields": {
                    name: left_fields.get(name) == right_fields.get(name)
                    for name in FIELDS
                },
                "params_identical": params_same,
                "state_identical": state_same,
                "adaround_calls": counts,
                "call_counts_identical": call_same,
            }
        )
        print(
            f"  {label}: weight params {'bit-identical' if params_same else 'DIFFER'}, "
            f"state {'bit-identical' if state_same else 'DIFFER'}, "
            f"_adaround_mantissa calls parent={counts['parent']} "
            f"candidate={counts['candidate']}"
        )
        if not (params_same and state_same and call_same):
            failures += 1
    return failures


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default=str(HERE / "verify.json"))
    args = parser.parse_args()
    device = torch.device(args.device)

    parent_bytes = PARENT.read_bytes()
    candidate_bytes = CANDIDATE.read_bytes()

    report: dict = {
        "parent": str(PARENT),
        "parent_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "parent_bytes": len(parent_bytes),
        "candidate": str(CANDIDATE),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes": len(candidate_bytes),
        "device": str(device),
        "unit": [],
        "unit_errors": [],
        "activation": [],
        "weight": [],
    }

    # The candidate must contain the parent unchanged, not a copy of it: the
    # shadow definition is what makes the two versions differ, so if the parent
    # byte prefix is not there this is not the artifact it claims to be.
    head = parent_bytes.rstrip(b"\n")
    starts_with_parent = candidate_bytes.startswith(head)
    report["candidate_starts_with_parent_bytes"] = starts_with_parent
    if not starts_with_parent:
        raise SystemExit("the candidate does not begin with the parent's bytes")

    parent = load_module(PARENT, "ad1_parent")
    candidate = load_module(CANDIDATE, "ad1_candidate")

    # The parent's own definition must still be present and unchanged inside the
    # candidate: this is a shadow, not an edit.
    import ast

    def fn_ast(module):
        for node in ast.parse(module).body:
            if isinstance(node, ast.FunctionDef) and node.name == "_adaround_mantissa":
                yield ast.dump(node)

    parent_defs = list(fn_ast(parent_bytes.decode("utf-8")))
    candidate_defs = list(fn_ast(candidate_bytes.decode("utf-8")))
    report["parent_definitions_of_helper"] = len(parent_defs)
    report["candidate_definitions_of_helper"] = len(candidate_defs)
    report["parent_definition_unchanged_in_candidate"] = candidate_defs[0] == parent_defs[0]
    if not report["parent_definition_unchanged_in_candidate"]:
        raise SystemExit("the parent's own _adaround_mantissa changed in the candidate")
    if len(candidate_defs) != 2:
        raise SystemExit(f"expected two definitions of the helper, got {len(candidate_defs)}")

    failures = 0
    print("unit: parent vs candidate _adaround_mantissa")
    failures += unit_part(parent, candidate, device, report)
    print(
        f"  {report['unit_cases']} combinations, "
        f"{report['unit_failures']} differing, "
        f"{len(report['unit_errors'])} raising on both sides"
    )

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    print("activation: the real dynamic path, five fields")
    failures += activation_part(parent, candidate, evaluator, pack, device, report)

    print("weight: the real calibration path, all five fields")
    failures += weight_part(parent, candidate, evaluator, pack, device, report)

    report["failures"] = failures
    report["equivalent"] = failures == 0
    (HERE / "verify.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"\n{'EQUIVALENT' if failures == 0 else f'{failures} FAILURES'} "
        f"-- wrote {args.output}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
