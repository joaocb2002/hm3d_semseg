"""Explicit HM3D raw-label to model-target mapping."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from hm3d_semseg.config.schema import TaxonomyConfig
from hm3d_semseg.exceptions import HM3DSemsegError
from hm3d_semseg.taxonomy.constants import MPCAT40_NAMES, UNKNOWN_ID
from hm3d_semseg.utils.hashing import sha256_file


def normalize_raw_name(name: str) -> str:
    """Apply the project's only label normalization: NFC, trim, casefold, whitespace."""
    normalized = unicodedata.normalize("NFC", name).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


def _delimiter_normalized_text(text: str) -> str:
    """Convert the upstream four-space TSV variant to tabs outside quoted fields."""
    if "\t" in text.partition("\n")[0]:
        return text
    output: List[str] = []
    quoted = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted = not quoted
            output.append(char)
            index += 1
        elif not quoted and text.startswith("    ", index):
            output.append("\t")
            index += 4
        else:
            output.append(char)
            index += 1
    return "".join(output)


@dataclass(frozen=True)
class MatterportRow:
    raw_category: str
    normalized_raw_category: str
    category: str
    mpcat40index: int
    mpcat40: str


class MatterportMapping:
    """Parsed and validated authoritative Matterport category mapping."""

    SOURCE_URL: ClassVar[str] = (
        "https://github.com/facebookresearch/habitat-sim/blob/main/"
        "data/matterport_semantics/matterport_category_mappings.tsv"
    )
    REQUIRED_COLUMNS: ClassVar[Set[str]] = {
        "raw_category",
        "category",
        "mpcat40index",
        "mpcat40",
    }

    def __init__(self, rows: Iterable[MatterportRow], source: Path, sha256: str) -> None:
        self.rows = list(rows)
        self.source = source
        self.source_url = self.SOURCE_URL
        self.sha256 = sha256
        grouped: Dict[str, List[MatterportRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.normalized_raw_category, []).append(row)
        self.by_raw_name = grouped
        self._validate_classes()

    @classmethod
    def from_file(
        cls, path: Path, expected_sha256: Optional[str] = None
    ) -> "MatterportMapping":
        """Load a quoted TSV/four-space-delimited upstream asset."""
        if not path.is_file():
            raise FileNotFoundError(f"Matterport mapping does not exist: {path}")
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise HM3DSemsegError(
                f"Taxonomy mapping hash mismatch for {path}: expected "
                f"{expected_sha256}, found {digest}"
            )
        text = _delimiter_normalized_text(path.read_text(encoding="utf-8-sig"))
        reader = csv.DictReader(
            io.StringIO(text), delimiter="\t", quotechar='"', skipinitialspace=True
        )
        missing = cls.REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise HM3DSemsegError(
                f"Matterport mapping is missing columns: {', '.join(sorted(missing))}"
            )
        rows: List[MatterportRow] = []
        for line_number, values in enumerate(reader, start=2):
            raw = (values["raw_category"] or "").strip()
            index_text = (values["mpcat40index"] or "").strip()
            if not raw or not index_text:
                continue
            try:
                mpcat_index = int(index_text)
            except ValueError as error:
                raise HM3DSemsegError(
                    f"{path}:{line_number}: invalid mpcat40index {index_text!r}"
                ) from error
            rows.append(
                MatterportRow(
                    raw_category=raw,
                    normalized_raw_category=normalize_raw_name(raw),
                    category=(values["category"] or "").strip(),
                    mpcat40index=mpcat_index,
                    mpcat40=(values["mpcat40"] or "").strip(),
                )
            )
        return cls(rows, path.resolve(), digest)

    def _validate_classes(self) -> None:
        observed: Dict[int, Set[str]] = {}
        for row in self.rows:
            if 1 <= row.mpcat40index <= 40:
                observed.setdefault(row.mpcat40index, set()).add(row.mpcat40)
        for index, expected in enumerate(MPCAT40_NAMES, start=1):
            names = observed.get(index, set())
            if names != {expected}:
                raise HM3DSemsegError(
                    f"Mapping class {index} should be {expected!r}, found {sorted(names)!r}"
                )

    def lookup(self, raw_name: str) -> Optional[MatterportRow]:
        """Return an unambiguous exact-normalized row or fail on conflicts."""
        rows = self.by_raw_name.get(normalize_raw_name(raw_name), [])
        if not rows:
            return None
        outcomes = {(row.mpcat40index, row.mpcat40) for row in rows}
        if len(outcomes) != 1:
            raise HM3DSemsegError(
                f"Raw category {raw_name!r} has conflicting Matterport mappings: "
                f"{sorted(outcomes)!r}"
            )
        return rows[0]


@dataclass(frozen=True)
class MappingDecision:
    raw_name: Optional[str]
    target_id: int
    status: str
    reason: str
    mpcat40index: Optional[int] = None
    mpcat40: Optional[str] = None


class TaxonomyMapper:
    """Apply the explicit unknown-versus-ignore policy."""

    def __init__(self, mapping: MatterportMapping, config: TaxonomyConfig) -> None:
        self.mapping = mapping
        self.config = config

    def _policy_target(self, policy_name: str) -> int:
        return UNKNOWN_ID if policy_name == "unknown" else self.config.ignore_index

    def map_raw_name(self, raw_name: Optional[str]) -> MappingDecision:
        if raw_name is None:
            policy = self.config.policy.missing_id
            return MappingDecision(None, self._policy_target(policy), policy, "missing_id")
        normalized = normalize_raw_name(raw_name)
        row = self.mapping.lookup(raw_name)
        if row is None:
            policy = self.config.policy.unmapped_name
            return MappingDecision(
                raw_name, self._policy_target(policy), policy, "unmapped_raw_name"
            )
        if 1 <= row.mpcat40index <= 40:
            return MappingDecision(
                raw_name,
                row.mpcat40index,
                "known",
                "authoritative_mpcat40",
                row.mpcat40index,
                row.mpcat40,
            )
        if row.mpcat40index == 41 or normalized in {"unknown", "unlabeled"}:
            key = "unlabeled" if normalized == "unlabeled" else "unknown"
            policy = getattr(self.config.policy, key)
            return MappingDecision(
                raw_name,
                self._policy_target(policy),
                policy,
                f"mapping_{key}",
                row.mpcat40index,
                row.mpcat40,
            )
        if row.mpcat40index == 0 or normalized in {"void", "remove", "delete"}:
            key = "void" if normalized == "void" else "remove"
            policy = getattr(self.config.policy, key)
            return MappingDecision(
                raw_name,
                self._policy_target(policy),
                policy,
                f"mapping_{key}",
                row.mpcat40index,
                row.mpcat40,
            )
        policy = self.config.policy.unmapped_name
        return MappingDecision(
            raw_name,
            self._policy_target(policy),
            policy,
            f"unsupported_mpcat40index_{row.mpcat40index}",
            row.mpcat40index,
            row.mpcat40,
        )

    def map_semantic_observation(
        self, semantic_ids: np.ndarray, id_to_raw_name: Mapping[int, str]
    ) -> Tuple[np.ndarray, Dict[int, MappingDecision]]:
        """Map a semantic-ID image and return the per-visible-ID audit trail."""
        if semantic_ids.ndim != 2:
            raise ValueError(f"Expected a 2D semantic observation, got {semantic_ids.shape}")
        mask = np.full(semantic_ids.shape, self.config.ignore_index, dtype=np.uint8)
        decisions: Dict[int, MappingDecision] = {}
        for value in np.unique(semantic_ids):
            semantic_id = int(value)
            decision = self.map_raw_name(id_to_raw_name.get(semantic_id))
            decisions[semantic_id] = decision
            mask[semantic_ids == value] = decision.target_id
        return mask, decisions
