"""L1 calibration timing probe (read-only, workbench only).

Wraps the live hotspots inside ``hif4_calibration_and_quantize_weight`` with
per-stage timers and runs the exact compact-panel weight states used by the
official evaluator, without modifying ``solution.py`` or any of its outputs.

Stage attribution (solution.py anchors, 2026-09-02 audit):
  s1  joint candidate search   : _linear_output_candidate_metrics(5031),
                                 _linear_candidate_metrics(4806),
                                 _linear_smooth_hybrid_metrics(5087)
  s2  weight GPTQ              : _gptq_quantize_weight(5323, call 8080),
                                 _transformed_covariance(4782, call 8056)
  s3  weight e2e refine        : _weight_e2e_refine(5123, call 8107)
  s4  activation Gram/Hessian  : torch.linalg.cholesky / torch.cholesky_inverse
                                 outside s1/s2 wrappers (live call 8258);
                                 inline transformed-sample loop and
                                 weight_output_gram matmul fall into "other"
  s5  CPU state construction   : _cpu_state_tensor calls
  other                        : total - (s1..s5)

Usage:
  python workbench/l1_calib_timing_probe.py --layers 0,8,15,23
  python workbench/l1_calib_timing_probe.py --all --output logs/l1_calib_timing_probe.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import official_eval as oe  # noqa: E402


class StageTimer:
    """Accumulates outermost-call time only (depth guard)."""

    def __init__(self, name: str, exclude_depths: list["StageTimer"] | None = None):
        self.name = name
        self.total = 0.0
        self.calls = 0
        self.depth = 0
        self.exclude_depths = exclude_depths or []

    def wrap(self, fn):
        def inner(*args, **kwargs):
            if any(guard.depth > 0 for guard in self.exclude_depths):
                return fn(*args, **kwargs)
            self.depth += 1
            outermost = self.depth == 1
            if outermost:
                torch.cuda.synchronize()
                started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.depth -= 1
                if outermost:
                    torch.cuda.synchronize()
                    self.total += time.perf_counter() - started
                    self.calls += 1

        return inner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0,8,15,23", help="comma-separated layers")
    parser.add_argument("--all", action="store_true", help="all compact weight states")
    parser.add_argument(
        "--output", default=str(ROOT / "logs" / "l1_calib_timing_probe.md")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    raw = oe.load_pack(oe.DEFAULT_CACHE)
    prepared = oe.prepare_pack(raw, compact_panel=True, evaluation_scenario="linear")
    state_keys = [
        (int(layer), str(role))
        for layer, role in prepared.metadata["linear_state_keys"]
    ]
    calib_indices = tuple(
        int(index) for index in prepared.metadata["linear_calibration_indices"]
    )
    if args.all:
        selected = state_keys
    else:
        layers = {int(item) for item in args.layers.split(",") if item != ""}
        selected = [(layer, role) for layer, role in state_keys if layer in layers]
    if not selected:
        raise RuntimeError(f"no weight states selected from {state_keys}")

    solution = oe.load_solution(ROOT / "solution.py")

    s1 = StageTimer("s1_joint_candidate_metrics")
    s2 = StageTimer("s2_weight_gptq")
    s3 = StageTimer("s3_weight_e2e_refine")
    s4 = StageTimer("s4_activation_gram_cholesky")
    s5 = StageTimer("s5_cpu_state")

    solution._linear_output_candidate_metrics = s1.wrap(
        solution._linear_output_candidate_metrics
    )
    solution._linear_output_candidate_metrics_combos = s1.wrap(
        solution._linear_output_candidate_metrics_combos
    )
    solution._linear_candidate_metrics = s1.wrap(solution._linear_candidate_metrics)
    solution._linear_smooth_hybrid_metrics = s1.wrap(
        solution._linear_smooth_hybrid_metrics
    )
    solution._gptq_quantize_weight = s2.wrap(solution._gptq_quantize_weight)
    solution._transformed_covariance = s2.wrap(solution._transformed_covariance)
    solution._weight_e2e_refine = s3.wrap(solution._weight_e2e_refine)
    solution._cpu_state_tensor = s5.wrap(solution._cpu_state_tensor)
    # s4 must not double-count Cholesky calls already covered by the s2 GPTQ
    # wrapper or by s1 candidate metrics; the depth guard routes those calls to
    # the enclosing stage.
    s4_excluded = [s1, s2, s3]
    orig_cholesky = torch.linalg.cholesky
    orig_cholesky_inverse = torch.cholesky_inverse

    def timed_cholesky(*a, **k):
        if any(guard.depth > 0 for guard in s4_excluded):
            return orig_cholesky(*a, **k)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return orig_cholesky(*a, **k)
        finally:
            torch.cuda.synchronize()
            s4.total += time.perf_counter() - started
            s4.calls += 1

    def timed_cholesky_inverse(*a, **k):
        if any(guard.depth > 0 for guard in s4_excluded):
            return orig_cholesky_inverse(*a, **k)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return orig_cholesky_inverse(*a, **k)
        finally:
            torch.cuda.synchronize()
            s4.total += time.perf_counter() - started
            s4.calls += 1

    torch.linalg.cholesky = timed_cholesky
    torch.cholesky_inverse = timed_cholesky_inverse

    timers = [s1, s2, s3, s4, s5]
    rows: list[dict[str, float | int | str]] = []
    print(
        f"[probe] {len(selected)} weight states, calibration indices {calib_indices}, "
        f"device {device}",
        flush=True,
    )
    try:
        for index, (layer, role) in enumerate(selected):
            weight_pair = oe._move_pair(prepared.weights[layer][role], device)
            calibration = [
                oe._move_pair(
                    prepared.linear_calibration_activations[role][sample][layer],
                    device,
                )
                for sample in calib_indices
            ]
            snapshot = {timer.name: timer.total for timer in timers}
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = solution.hif4_calibration_and_quantize_weight(
                weight_pair[0], weight_pair[1], calibration
            )
            torch.cuda.synchronize()
            total = time.perf_counter() - started
            row: dict[str, float | int | str] = {
                "layer": layer,
                "role": role,
                "total_ms": total * 1000.0,
            }
            attributed = 0.0
            for timer in timers:
                delta = timer.total - snapshot[timer.name]
                row[timer.name] = delta * 1000.0
                attributed += delta
            row["other_ms"] = max(0.0, (total - attributed) * 1000.0)
            rows.append(row)
            print(
                f"[probe] {index + 1}/{len(selected)} L{layer} {role}: "
                f"{row['total_ms']:.1f} ms",
                flush=True,
            )
            del result, weight_pair, calibration
    finally:
        torch.linalg.cholesky = orig_cholesky
        torch.cholesky_inverse = orig_cholesky_inverse

    sha = hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest()[:16]
    stage_total = {
        timer.name: sum(float(row[timer.name]) for row in rows) / 1000.0
        for timer in timers
    }
    grand_total = sum(float(row["total_ms"]) for row in rows) / 1000.0
    other_total = sum(float(row["other_ms"]) for row in rows) / 1000.0

    lines = [
        "# L1 calibration timing probe",
        "",
        f"- solution.py SHA256[:16]: `{sha}`",
        f"- device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}",
        f"- panel: linear compact, {len(selected)} weight states "
        f"(layers {sorted({row['layer'] for row in rows})}), "
        f"calibration indices {calib_indices}",
        "- attribution: outermost-call timing with per-call CUDA synchronize; "
        "per-call sync overhead slightly inflates s1 (many small calls), so "
        "read stage shares as ranking evidence, not exact ratios",
        "",
        "## Per weight state (ms)",
        "",
        "| layer | role | s1 joint | s2 gptq | s3 refine | s4 act | s5 state | other | total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {layer} | {role} | {s1:.1f} | {s2:.1f} | {s3:.1f} | {s4:.1f} | "
            "{s5:.1f} | {other:.1f} | {total:.1f} |".format(
                layer=row["layer"],
                role=row["role"],
                s1=float(row["s1_joint_candidate_metrics"]),
                s2=float(row["s2_weight_gptq"]),
                s3=float(row["s3_weight_e2e_refine"]),
                s4=float(row["s4_activation_gram_cholesky"]),
                s5=float(row["s5_cpu_state"]),
                other=float(row["other_ms"]),
                total=float(row["total_ms"]),
            )
        )
    lines += [
        "",
        "## Stage summary",
        "",
        "| stage | total s | share % |",
        "| --- | ---: | ---: |",
    ]
    for name, seconds in stage_total.items():
        share = seconds / grand_total * 100.0 if grand_total else 0.0
        lines.append(f"| {name} | {seconds:.3f} | {share:.1f} |")
    lines.append(f"| other | {other_total:.3f} | {other_total / grand_total * 100.0:.1f} |")
    lines.append(f"| **total** | **{grand_total:.3f}** | 100.0 |")
    lines.append("")
    lines.append(
        f"- s1 calls: {s1.calls}, s2 calls: {s2.calls}, s3 calls: {s3.calls}, "
        f"s4 calls: {s4.calls}, s5 calls: {s5.calls}"
    )

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[probe] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
