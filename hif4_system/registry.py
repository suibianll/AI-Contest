"""File-backed immutable Champion registry.

Each source file is copied into a version directory and addressed by its
content hash.  The only mutable state is the small Champion pointer, the
candidate index, and the append-only history file.  Promotion re-hashes the
candidate immediately before changing the pointer, so editing a pending
candidate can never silently change the Champion.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .solution_loader import sha256_file


class RegistryError(RuntimeError):
    """Base class for registry state and lifecycle errors."""


class HashMismatch(RegistryError):
    """Raised when source bytes do not match the hash bound to a record."""


class PromotionRejected(RegistryError):
    """Raised when a candidate has not passed the required promotion reports."""


@dataclass(frozen=True)
class VersionRecord:
    id: str
    sha256: str
    solution_path: Path
    reports: Mapping[str, Any]
    created_at: str
    kind: str


@dataclass(frozen=True)
class CandidateRecord(VersionRecord):
    pass


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_bytes(path, payload)


def _utc_id(sha256: str) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{sha256[:12]}", timestamp


def _nested_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_value(child, key)
            if found is not None:
                return found
    elif is_dataclass(value) and not isinstance(value, type):
        return _nested_value(asdict(value), key)
    return None


def _report_status(report: Any) -> str:
    value = _nested_value(report, "status")
    return str(value).lower() if value is not None else ""


def _report_hash(report: Any) -> str | None:
    value = _nested_value(report, "candidate_sha256")
    return str(value).lower() if value is not None else None


def _report_gate_passed(report: Any) -> bool:
    """Honor an explicit decision/gate failure while allowing track reports."""
    decision = _nested_value(report, "decision")
    if isinstance(decision, Mapping) and "promote" in decision:
        return bool(decision["promote"])
    promote = _nested_value(report, "promote")
    if promote is not None:
        return bool(promote)
    return True


class Registry:
    """Manage immutable candidate snapshots and the active Champion pointer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.versions_dir = self.root / "versions"
        self.champion_path = self.root / "champion.json"
        self.candidates_path = self.root / "candidates.json"
        self.history_path = self.root / "history.json"

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _new_version(
        self, source: Path, sha256: str, reports: Mapping[str, Any], kind: str
    ) -> VersionRecord:
        self._ensure_layout()
        version_id, created_at = _utc_id(sha256)
        version_dir = self.versions_dir / version_id
        while version_dir.exists():
            version_id, created_at = _utc_id(sha256)
            version_dir = self.versions_dir / version_id
        version_dir.mkdir()
        destination = version_dir / "solution.py"
        with source.open("rb") as source_stream:
            _atomic_bytes(destination, source_stream.read())
        if sha256_file(destination) != sha256:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise HashMismatch("copied candidate hash does not match source hash")
        normalized = _jsonable(dict(reports))
        _atomic_json(
            version_dir / "version.json",
            {
                "id": version_id,
                "sha256": sha256,
                "reports": normalized,
                "created_at": created_at,
                "kind": kind,
            },
        )
        return VersionRecord(version_id, sha256, destination, normalized, created_at, kind)

    def _read_version(self, version_id: str) -> VersionRecord:
        version_dir = self.versions_dir / version_id
        metadata_path = version_dir / "version.json"
        solution_path = version_dir / "solution.py"
        if not metadata_path.is_file() or not solution_path.is_file():
            raise RegistryError(f"unknown or incomplete version: {version_id}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        kind = str(payload.get("kind", "candidate"))
        record_type = CandidateRecord if kind == "candidate" else VersionRecord
        return record_type(
            id=str(payload["id"]),
            sha256=str(payload["sha256"]),
            solution_path=solution_path,
            reports=dict(payload.get("reports", {})),
            created_at=str(payload.get("created_at", "")),
            kind=kind,
        )

    def _write_pointer(self, record: VersionRecord) -> None:
        _atomic_json(
            self.champion_path,
            {
                "version_id": record.id,
                "sha256": record.sha256,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            return []
        value = json.loads(self.history_path.read_text(encoding="utf-8"))
        return list(value) if isinstance(value, list) else []

    def _append_history(self, event: Mapping[str, Any]) -> None:
        history = self._read_history()
        history.append(dict(event, at=datetime.now(timezone.utc).isoformat()))
        _atomic_json(self.history_path, history)

    def initialize(self, solution_path: Path, expected_sha256: str) -> VersionRecord:
        source = Path(solution_path).resolve()
        actual = sha256_file(source)
        expected = str(expected_sha256).lower()
        if actual.lower() != expected:
            raise HashMismatch(f"initial Champion hash mismatch: expected {expected}, got {actual}")
        if self.champion_path.is_file():
            current = self.champion()
            if current.sha256.lower() == actual.lower():
                return current
            raise RegistryError("registry is already initialized with another Champion")
        record = self._new_version(source, actual, {}, "initial")
        self._write_pointer(record)
        _atomic_json(self.candidates_path, [])
        self._append_history({"action": "initialize", "version_id": record.id, "previous_version_id": None})
        return record

    def register_candidate(self, solution_path: Path, reports: Mapping[str, Any]) -> CandidateRecord:
        source = Path(solution_path).resolve()
        actual = sha256_file(source)
        record = self._new_version(source, actual, reports, "candidate")
        entries = []
        if self.candidates_path.is_file():
            entries = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        entries.append({"id": record.id, "status": "pending"})
        _atomic_json(self.candidates_path, entries)
        return CandidateRecord(record.id, record.sha256, record.solution_path, record.reports, record.created_at, record.kind)

    def pending(self) -> tuple[CandidateRecord, ...]:
        if not self.candidates_path.is_file():
            return ()
        entries = json.loads(self.candidates_path.read_text(encoding="utf-8"))
        records: list[CandidateRecord] = []
        for entry in entries:
            if str(entry.get("status")) != "pending":
                continue
            record = self._read_version(str(entry["id"]))
            records.append(CandidateRecord(record.id, record.sha256, record.solution_path, record.reports, record.created_at, record.kind))
        return tuple(records)

    def _candidate(self, candidate_id: str) -> CandidateRecord:
        record = self._read_version(candidate_id)
        if record.kind != "candidate":
            raise RegistryError(f"version is not a pending candidate: {candidate_id}")
        return CandidateRecord(record.id, record.sha256, record.solution_path, record.reports, record.created_at, record.kind)

    @staticmethod
    def _required_report(reports: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
        lowered = {str(key).lower(): value for key, value in reports.items()}
        for name in names:
            if name in lowered:
                return lowered[name]
        return None

    def _validate_reports(self, candidate: CandidateRecord, expected_sha256: str) -> None:
        required = {
            "gpu_dev": ("gpu_dev", "gpu", "accuracy_dev", "accuracy"),
            "cpu_dev": ("cpu_dev", "cpu", "timing_dev"),
            "holdout": ("holdout", "holdout_dev"),
        }
        for label, names in required.items():
            report = self._required_report(candidate.reports, names)
            if report is None:
                raise PromotionRejected(f"missing required {label} report")
            if _report_status(report) != "passed":
                raise PromotionRejected(f"{label} report is not passed")
            bound_hash = _report_hash(report)
            if bound_hash != expected_sha256.lower():
                raise HashMismatch(f"{label} report is not bound to candidate hash")
            if not _report_gate_passed(report):
                raise PromotionRejected(f"{label} report contains a failed promotion decision")

    def promote(self, candidate_id: str, expected_sha256: str) -> VersionRecord:
        candidate = self._candidate(candidate_id)
        actual = sha256_file(candidate.solution_path)
        expected = str(expected_sha256).lower()
        if actual.lower() != candidate.sha256.lower() or actual.lower() != expected:
            raise HashMismatch(f"candidate hash changed: expected {expected}, got {actual}")
        self._validate_reports(candidate, actual)
        previous = self.champion() if self.champion_path.is_file() else None
        self._write_pointer(candidate)
        entries = json.loads(self.candidates_path.read_text(encoding="utf-8")) if self.candidates_path.is_file() else []
        for entry in entries:
            if str(entry.get("id")) == candidate.id:
                entry["status"] = "promoted"
        _atomic_json(self.candidates_path, entries)
        self._append_history(
            {
                "action": "promote",
                "version_id": candidate.id,
                "previous_version_id": previous.id if previous else None,
            }
        )
        return self._read_version(candidate.id)

    def champion(self) -> VersionRecord:
        if not self.champion_path.is_file():
            raise RegistryError("registry has no Champion")
        payload = json.loads(self.champion_path.read_text(encoding="utf-8"))
        return self._read_version(str(payload["version_id"]))

    def history(self) -> list[dict[str, Any]]:
        return self._read_history()

    def rollback(self, version_id: str | None = None) -> VersionRecord:
        current = self.champion()
        history = self._read_history()
        target_id = version_id
        if target_id is None:
            for event in reversed(history):
                if event.get("version_id") == current.id and event.get("previous_version_id"):
                    target_id = str(event["previous_version_id"])
                    break
        if not target_id:
            raise RegistryError("no previous Champion is available for rollback")
        target = self._read_version(target_id)
        self._write_pointer(target)
        self._append_history(
            {"action": "rollback", "version_id": target.id, "previous_version_id": current.id}
        )
        return target
