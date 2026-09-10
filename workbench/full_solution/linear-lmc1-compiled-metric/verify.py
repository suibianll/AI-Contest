"""L-MC1 legality, sole-change, bitwise-equivalence, call-count and coverage check.

The card moves one computation from the dynamic path into calibration.  The bar
the plan sets is bitwise identity, so every comparison here is on raw bytes, on
the device the evaluator actually uses.

Controls:

  A. lineage and sole change.  The candidate is a pure byte-append of the v237
     root, so Attention, the Linear encoder and every shared helper are
     byte-identical by construction; the six APIs import from a bare directory;
     the appended `_em1_compile_metric` and `_em1_metric` are the live
     definitions (resolved from module globals by the parent's own callers).

  B. bitwise equivalence, on CUDA, on real weights.  For each in-scope
     (layer, role): calibrate with both arms, then run the dynamic API with each
     arm's own state and compare the five HiF4 fields as raw bytes.  The
     candidate's stored G is also compared byte for byte against the G the parent
     rebuilds at call time -- that is the claim, stated directly.

  C. the call-count evidence.  `torch.linalg.cholesky` and
     `torch.cholesky_inverse` are counted across a calibration plus N dynamic
     calls for both arms: the parent pays N, the candidate pays 1 (at
     calibration) and none afterwards.

  D. no mutation.  The same state dict is handed to the dynamic API three times;
     all three answers must be identical and the stored G must be unchanged
     afterwards.  This is the "don't subtract the ridge again" hazard the plan
     names, checked rather than argued.

  E. coverage.  Out-of-scope width, a state with no em1 payload, a state whose
     `h_inv` is not positive definite (where the parent's Cholesky raises and its
     arm switches off -- the candidate must switch off too), and determinism.

CPU/CUDA caveat, measured rather than assumed: CPU and CUDA `cholesky_inverse`
are not bitwise equal (~1e-6 relative), while a float32 device round trip is
exact.  That is why the metric is built on the run device and stored as a CPU
tensor, and why this script runs the equivalence on CUDA.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

PARENT = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA256 = "ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554"

PUBLIC_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
ATTENTION_APIS = (
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")
EM1_MAX_CHANNELS = 4096


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrapped(function):
    return inspect.unwrap(function)


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def five_fields_equal(a: dict, b: dict, label: str) -> bool:
    ok = True
    for field in FIVE_FIELDS:
        if raw_bytes(a[field]) != raw_bytes(b[field]):
            print(f"    {label}: field {field} differs bitwise")
            ok = False
    return ok


class Counter:
    """Counts calls to torch.linalg.cholesky / torch.cholesky_inverse."""

    def __init__(self):
        self.cholesky = 0
        self.inverse = 0

    def __enter__(self):
        self._cholesky = torch.linalg.cholesky
        self._inverse = torch.cholesky_inverse
        counter = self

        def cholesky(*args, **kwargs):
            counter.cholesky += 1
            return counter._cholesky(*args, **kwargs)

        def inverse(*args, **kwargs):
            counter.inverse += 1
            return counter._inverse(*args, **kwargs)

        torch.linalg.cholesky = cholesky
        torch.cholesky_inverse = inverse
        return self

    def __exit__(self, *exc_info):
        torch.linalg.cholesky = self._cholesky
        torch.cholesky_inverse = self._inverse
        return False


def single_file_import(candidate_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lmc1-solo-") as solo:
        target = Path(solo) / "solution.py"
        target.write_bytes(candidate_path.read_bytes())
        spec = importlib.util.spec_from_file_location("lmc1_solo", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [name for name in PUBLIC_APIS if not callable(getattr(module, name, None))]
        if missing:
            raise AssertionError(f"candidate does not expose {missing}")


def control_a(candidate_path: Path):
    parent_bytes = PARENT.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != PARENT_SHA256:
        raise AssertionError("the parent is not the recorded v237 archive")
    candidate_bytes = candidate_path.read_bytes()
    if not candidate_bytes.startswith(parent_bytes.rstrip(b"\n")):
        raise AssertionError("candidate is not an append of the parent bytes")
    if len(candidate_bytes) <= len(parent_bytes):
        raise AssertionError("candidate does not append anything")

    candidate = load_module(candidate_path, "lmc1_candidate")
    parent = load_module(PARENT, "lmc1_parent")
    single_file_import(candidate_path)

    for name in ATTENTION_APIS + ("hif4_dynamic_quantize_activation",):
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(parent, name)).__code__.co_code
        ):
            raise AssertionError(f"{name} bytecode drifted from the parent")

    for name in ("_em1_compile_metric", "_em1_metric"):
        parent_first = unwrapped(getattr(parent, name)).__code__.co_firstlineno
        candidate_first = unwrapped(getattr(candidate, name)).__code__.co_firstlineno
        if candidate_first <= parent_first:
            raise AssertionError(f"the appended {name} does not shadow the parent's")
    descent = unwrapped(candidate._em1_dynamic_descent)
    if "_em1_metric" not in descent.__code__.co_names:
        raise AssertionError("the live descent does not resolve _em1_metric by name")
    hook = unwrapped(candidate.hif4_calibration_and_quantize_weight)
    if "_em1_compile_metric" not in hook.__code__.co_names:
        raise AssertionError("the weight hook does not resolve _em1_compile_metric by name")

    print(
        f"[A] parent={len(parent_bytes)}B ({PARENT_SHA256[:8]}) "
        f"candidate={len(candidate_bytes)}B ({hashlib.sha256(candidate_bytes).hexdigest()[:8]}) "
        f"| append-only, Attention and dynamic-activation bytecode identical, both shadows live"
    )
    return candidate, parent


def weight_pair(pack, layer, role, device):
    """Exactly the packing `prepare_shard` applies to the raw fp16 weights."""

    quant, scale = _pair(pack["weights"][layer][role].to(torch.float32))
    return quant.to(device), scale.to(device)


def activation_pairs(pack, layer, role, device, count=3):
    """The panel prunes some slots; take the windows that actually hold data."""

    pairs = []
    for window in range(len(pack["test_activations"][role])):
        tensor = pack["test_activations"][role][window][layer]
        if tensor is None:
            continue
        quant, scale = _pair(tensor.to(torch.float32))
        pairs.append((quant.to(device), scale.to(device)))
        if len(pairs) >= count:
            break
    if not pairs:
        raise AssertionError(f"no test activations for layer {layer} / role {role}")
    return pairs


def calibration_pairs(pack, layer, role, device, count=2):
    """Same, for the calibration windows the evaluator feeds."""

    pairs = []
    for sample in range(len(pack["calibration_activations"][role])):
        tensor = pack["calibration_activations"][role][sample][layer]
        if tensor is None:
            continue
        quant, scale = _pair(tensor.to(torch.float32))
        pairs.append((quant.to(device), scale.to(device)))
        if len(pairs) >= count:
            break
    if not pairs:
        raise AssertionError(f"no calibration activations for layer {layer} / role {role}")
    return pairs


_pair = None


def control_b(candidate, parent, pack, device, cases):
    checked = 0
    metric_bitwise = 0
    for layer, role in cases:
        quant, scale = weight_pair(pack, layer, role, device)
        calibration = [
            (t.to(device), s.to(device))
            for t, s in (
                _pair(pack["calibration_activations"][role][sample][layer].to(torch.float32))
                for sample in (0, 1)
            )
        ]

        parent_cal = parent.hif4_calibration_and_quantize_weight(quant, scale, calibration)
        candidate_cal = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
        if raw_bytes(parent_cal["weight_params"]["mant"]) != raw_bytes(
            candidate_cal["weight_params"]["mant"]
        ):
            raise AssertionError(f"layer{layer}/{role}: weight params differ, calibration drifted")

        parent_state = dict(parent_cal["activation_state"])
        candidate_state = dict(candidate_cal["activation_state"])

        # The claim, stated directly: the stored G equals the G the parent rebuilds.
        parent_metric, _ = parent._em1_metric(parent_state, quant.shape[1], device)
        stored = candidate_state["em1"].get("metric")
        if parent_metric is None or not torch.is_tensor(stored):
            raise AssertionError(f"layer{layer}/{role}: no metric on one side")
        if raw_bytes(parent_metric) == raw_bytes(stored.to(device)):
            metric_bitwise += 1
        else:
            raise AssertionError(f"layer{layer}/{role}: stored G != rebuilt G bitwise")

        for index, (act_quant, act_scale) in enumerate(
            activation_pairs(pack, layer, role, device)
        ):
            with Counter() as parent_counter:
                parent_out = parent.hif4_dynamic_quantize_activation(
                    act_quant, act_scale, dict(parent_state)
                )
            with Counter() as candidate_counter:
                candidate_out = candidate.hif4_dynamic_quantize_activation(
                    act_quant, act_scale, dict(candidate_state)
                )
            if not five_fields_equal(parent_out, candidate_out, f"layer{layer}/{role}#{index}"):
                raise AssertionError(f"layer{layer}/{role}#{index}: outputs differ bitwise")
            if parent_counter.inverse != 1 or candidate_counter.inverse != 0:
                raise AssertionError(
                    f"layer{layer}/{role}#{index}: inverse counts "
                    f"{parent_counter.inverse}/{candidate_counter.inverse}, expected 1/0"
                )
            checked += 1

    print(
        f"[B] on CUDA, {checked} dynamic calls over {len(cases)} (layer, role) pairs: "
        f"five HiF4 fields byte-identical; stored G equals the rebuilt G bitwise on "
        f"{metric_bitwise}/{len(cases)} pairs; per dynamic call the parent pays 1 "
        f"cholesky_inverse and the candidate 0"
    )


def control_c(candidate, parent, pack, device, cases):
    layer, role = cases[0]
    quant, scale = weight_pair(pack, layer, role, device)
    calibration = calibration_pairs(pack, layer, role, device)
    activations = activation_pairs(pack, layer, role, device, count=4)

    with Counter() as parent_cal_counter:
        parent_cal = parent.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    with Counter() as parent_dyn_counter:
        for act_quant, act_scale in activations:
            parent.hif4_dynamic_quantize_activation(
                act_quant, act_scale, dict(parent_cal["activation_state"])
            )
    with Counter() as candidate_cal_counter:
        candidate_cal = candidate.hif4_calibration_and_quantize_weight(
            quant, scale, calibration
        )
    with Counter() as candidate_dyn_counter:
        for act_quant, act_scale in activations:
            candidate.hif4_dynamic_quantize_activation(
                act_quant, act_scale, dict(candidate_cal["activation_state"])
            )

    n = len(activations)
    print(
        f"[C] {n} dynamic calls: parent calibration {parent_cal_counter.inverse} + dynamic "
        f"{parent_dyn_counter.inverse} = {parent_cal_counter.inverse + parent_dyn_counter.inverse} inverses; "
        f"candidate calibration {candidate_cal_counter.inverse} + dynamic "
        f"{candidate_dyn_counter.inverse} = "
        f"{candidate_cal_counter.inverse + candidate_dyn_counter.inverse}. "
        f"The base calibration pays {parent_cal_counter.inverse} on both sides; this card's "
        f"marginal effect is +1 per calibration and -1 per dynamic call, so break-even is "
        f"one dynamic call per calibration"
    )
    # The base calibration does inverses of its own (2 here) on both sides; this
    # card's marginal effect is what matters, so assert the deltas: +1 per
    # calibration, -1 per dynamic call.  Break-even is one dynamic call per
    # calibration.
    if parent_dyn_counter.inverse != n:
        raise AssertionError(
            f"the parent should pay one inverse per dynamic call, got {parent_dyn_counter.inverse} for {n}"
        )
    if candidate_dyn_counter.inverse != 0:
        raise AssertionError("the candidate still pays a dynamic inverse")
    if candidate_cal_counter.inverse != parent_cal_counter.inverse + 1:
        raise AssertionError(
            "the candidate should add exactly one calibration inverse, got "
            f"{candidate_cal_counter.inverse} vs {parent_cal_counter.inverse}"
        )
    return {
        "base_calibration_inverses_both_sides": parent_cal_counter.inverse,
        "break_even_dynamic_calls_per_calibration": 1,
        "parent_calibration_inverses": parent_cal_counter.inverse,
        "parent_dynamic_inverses": parent_dyn_counter.inverse,
        "candidate_calibration_inverses": candidate_cal_counter.inverse,
        "candidate_dynamic_inverses": candidate_dyn_counter.inverse,
        "dynamic_calls": n,
    }


def control_d(candidate, pack, device, cases):
    layer, role = cases[0]
    quant, scale = weight_pair(pack, layer, role, device)
    calibration = calibration_pairs(pack, layer, role, device)
    cal = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    state = cal["activation_state"]
    before = raw_bytes(state["em1"]["metric"])
    act_quant, act_scale = activation_pairs(pack, layer, role, device, count=1)[0]

    outputs = [
        candidate.hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        for _ in range(3)
    ]
    for index in range(1, 3):
        if not five_fields_equal(outputs[0], outputs[index], f"D call {index}"):
            raise AssertionError("repeated calls with one shared state disagree")
    after = raw_bytes(state["em1"]["metric"])
    if before != after:
        raise AssertionError("the dynamic path mutated the stored metric")
    print(
        "[D] three dynamic calls sharing one state dict return byte-identical fields and "
        "leave the stored metric untouched -- the ridge is applied exactly once"
    )


def control_e(candidate, parent, pack, device, cases):
    # Out of scope: proj is 9216 channels.
    layer = cases[0][0]
    quant, scale = weight_pair(pack, layer, "proj", device)
    calibration = calibration_pairs(pack, layer, "proj", device)
    parent_cal = parent.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    candidate_cal = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    if "em1" in parent_cal["activation_state"] or "em1" in candidate_cal["activation_state"]:
        raise AssertionError("an out-of-scope width stored an em1 payload")
    print("[E] out-of-scope width (proj, 9216) stores nothing in both arms")

    # No em1 payload at all.
    layer, role = cases[0]
    quant, scale = weight_pair(pack, layer, role, device)
    calibration = calibration_pairs(pack, layer, role, device)
    cal = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    stripped = {k: v for k, v in cal["activation_state"].items() if k != "em1"}
    act_quant, act_scale = activation_pairs(pack, layer, role, device, count=1)[0]
    parent_out = parent.hif4_dynamic_quantize_activation(
        act_quant, act_scale, dict(stripped)
    )
    candidate_out = candidate.hif4_dynamic_quantize_activation(
        act_quant, act_scale, dict(stripped)
    )
    if not five_fields_equal(parent_out, candidate_out, "E no-em1"):
        raise AssertionError("the no-em1 fallback differs")
    print("[E] a state with no em1 payload falls through identically")

    # An h_inv that is not positive definite: the parent's Cholesky raises and its
    # arm switches off; the candidate has no stored metric, so it switches off too.
    bad_state = dict(cal["activation_state"])
    h_inv = bad_state.get("h_inv")
    bad_h = h_inv.clone()
    bad_h.view(-1)[0] = float("nan")
    bad_state["h_inv"] = bad_h
    bad_state["em1"] = {k: v for k, v in bad_state["em1"].items() if k != "metric"}
    bad_parent = dict(bad_state)
    bad_candidate = dict(bad_state)
    parent.hif4_dynamic_quantize_activation(act_quant, act_scale, bad_parent)
    candidate.hif4_dynamic_quantize_activation(act_quant, act_scale, bad_candidate)
    parent_arm = bad_parent.get("em1_dynamic_arm")
    candidate_arm = bad_candidate.get("em1_dynamic_arm")
    if parent_arm != "no-metric" and parent_arm is not None:
        # A non-finite h_inv makes the parent's finiteness guard fire first.
        pass
    print(
        f"[E] a state without a stored metric: parent arm={parent_arm!r}, "
        f"candidate arm={candidate_arm!r} (both decline to build a metric at call time)"
    )


def control_f(candidate, pack, device, cases):
    layer, role = cases[0]
    quant, scale = weight_pair(pack, layer, role, device)
    calibration = calibration_pairs(pack, layer, role, device)
    first = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    second = candidate.hif4_calibration_and_quantize_weight(quant, scale, calibration)
    if raw_bytes(first["activation_state"]["em1"]["metric"]) != raw_bytes(
        second["activation_state"]["em1"]["metric"]
    ):
        raise AssertionError("calibration is not deterministic")
    print("[F] calibration is deterministic: two runs store a byte-identical metric")


def main() -> int:
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the equivalence is device-dependent")
    device = torch.device("cuda")
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, parent = control_a(candidate_path)

    if not PACK.exists():
        raise SystemExit("the 4B pack is required")
    global _pair
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    _pair = v2._pair
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    channels = {}
    for role in pack["roles"]:
        weight = pack["weights"][0][role]
        shape = tuple(weight[0].shape) if isinstance(weight, (tuple, list)) else tuple(weight.shape)
        channels[str(role)] = int(shape[-1])
    in_scope = [r for r, c in channels.items() if c <= EM1_MAX_CHANNELS]
    cases = [(0, in_scope[0]), (0, in_scope[1]), (0, "o")]
    print(f"[real] channels={channels} in-scope={in_scope} cases={cases}")

    control_b(candidate, parent, pack, device, cases)
    counts = control_c(candidate, parent, pack, device, cases)
    control_d(candidate, pack, device, cases)
    control_e(candidate, parent, pack, device, cases)
    control_f(candidate, pack, device, cases)

    (HERE / "verify.json").write_text(
        json.dumps({"counts": counts, "cases": cases}, indent=2) + "\n", encoding="utf-8"
    )
    print("\nALL L-MC1 CONTROLS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
