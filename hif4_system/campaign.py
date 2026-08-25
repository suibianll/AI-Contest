from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config


class HoldoutBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedReservation:
    split: str
    tier: str
    seeds: tuple[int, ...]
    attempt: int | None
    commitment: str


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class Campaign:
    def __init__(self, directory: Path, manifest: dict[str, Any], secret: bytes) -> None:
        self.directory = directory
        self.manifest = manifest
        self.secret = secret
        self.manifest_path = directory / "campaign.json"

    @classmethod
    def create(cls, directory: Path, max_holdout_uses: int = 3) -> "Campaign":
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "campaign.json"
        secret_path = directory / ".holdout_secret"
        if manifest_path.exists() or secret_path.exists():
            raise FileExistsError("campaign already exists")
        secret = secrets.token_bytes(32)
        secret_path.write_bytes(secret)
        manifest = {"schema_version": 1, "campaign_id": secrets.token_hex(8), "dev_seeds": [101, 211, 307], "holdout_uses": 0, "max_holdout_uses": int(max_holdout_uses), "thresholds_locked": False, "runs": []}
        _atomic_json(manifest_path, manifest)
        return cls(directory, manifest, secret)

    @classmethod
    def open(cls, directory: Path) -> "Campaign":
        manifest_path = directory / "campaign.json"
        secret_path = directory / ".holdout_secret"
        if not manifest_path.exists() or not secret_path.exists():
            raise FileNotFoundError("campaign.json and .holdout_secret must exist")
        return cls(directory, json.loads(manifest_path.read_text(encoding="utf-8")), secret_path.read_bytes())

    def _save(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def reserve(self, split: str, tier: str) -> SeedReservation:
        tier_config = load_config(None).tier(tier)
        if split == "dev":
            seeds = tuple(self.manifest["dev_seeds"][: tier_config.dev_seeds])
            attempt = None
            commitment = hashlib.sha256(json.dumps(list(seeds)).encode("utf-8")).hexdigest()
        elif split == "holdout":
            used = int(self.manifest["holdout_uses"])
            maximum = int(self.manifest["max_holdout_uses"])
            if used >= maximum:
                raise HoldoutBudgetExhausted(f"holdout budget exhausted ({used}/{maximum})")
            attempt = used + 1
            seeds = tuple(self._derive_seeds(attempt, tier_config.holdout_seeds))
            commitment = hmac.new(self.secret, json.dumps(list(seeds), separators=(",", ":")).encode("utf-8"), hashlib.sha256).hexdigest()
            self.manifest["holdout_uses"] = attempt
            self.manifest["thresholds_locked"] = True
        else:
            raise ValueError("split must be dev or holdout")
        self.manifest["runs"].append({"split": split, "tier": tier, "attempt": attempt, "seed_commitment": commitment, "status": "reserved"})
        self._save()
        return SeedReservation(split, tier, seeds, attempt, commitment)

    def _derive_seeds(self, attempt: int, count: int) -> list[int]:
        values: list[int] = []
        for index in range(count):
            message = f"{self.manifest['campaign_id']}:{attempt}:{index}".encode("utf-8")
            digest = hmac.new(self.secret, message, hashlib.sha256).digest()
            values.append(int.from_bytes(digest[:8], "big") % 2_000_000_000 + 1)
        return values

    def finish(self, reservation: SeedReservation, status: str, report: str | None) -> None:
        for entry in reversed(self.manifest["runs"]):
            if entry["seed_commitment"] == reservation.commitment:
                entry["status"] = status
                if report is not None:
                    entry["report"] = str(report)
                self._save()
                return
        raise KeyError("seed reservation not found")
