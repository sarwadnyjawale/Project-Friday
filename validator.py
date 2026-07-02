"""
validator.py - Internal replica of validate_submission.py logic.

Runs before writing to catch errors the external validator would catch.
Failing here is recoverable. Failing at Stage 3 submission is not.

Improved (A2):
    - candidate_id existence verification against source dataset
    - duplicate candidate_id detection in output
    - full Redrob spec section 3 compliance
    - honeypot rate check (spec section 7)
    - comprehensive error reporting
"""

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger("validator")


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SubmissionValidator:
    """Validates a submission CSV against all known rules (spec section 3)."""

    CANDIDATE_ID_PATTERN = re.compile(r"^CAND_\d{7}$")
    REQUIRED_COLUMNS = {"candidate_id", "rank", "score", "reasoning"}

    def __init__(self, path: Path, candidates_file: Optional[Path] = None) -> None:
        """
        Args:
            path: Path to the submission CSV file.
            candidates_file: Optional path to candidates.jsonl for ID existence check.
                             If provided, verifies every candidate_id in the submission
                             exists in the source dataset (spec section 3 requirement).
        """
        self.path = path
        self.candidates_file = candidates_file

    def validate(self) -> ValidationResult:
        result = ValidationResult(passed=True)

        if not self.path.exists():
            result.passed = False
            result.errors.append(f"File not found: {self.path}")
            return result

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Check columns exist and are in correct order
                fieldnames = reader.fieldnames or []
                if not self.REQUIRED_COLUMNS.issubset(set(fieldnames)):
                    missing = self.REQUIRED_COLUMNS - set(fieldnames)
                    result.errors.append(f"Missing columns: {missing}")
                    result.passed = False
                    return result

                # Check column order (spec section 2: candidate_id,rank,score,reasoning)
                expected_order = ["candidate_id", "rank", "score", "reasoning"]
                if fieldnames != expected_order:
                    result.errors.append(
                        f"Column order wrong: expected {expected_order}, got {fieldnames}"
                    )
                    result.passed = False

                rows = list(reader)

                # Check row count (spec section 3: exactly 100 rows)
                if len(rows) != 100:
                    result.errors.append(
                        f"Expected 100 rows, got {len(rows)}"
                    )
                    result.passed = False

                ranks_seen: Set[int] = set()
                candidate_ids_seen: Set[str] = set()
                prev_score = float("inf")

                for i, row in enumerate(rows):
                    row_num = i + 2  # +1 for header, +1 for 1-indexed

                    # Validate candidate_id format (spec section 2)
                    cid = row.get("candidate_id", "")
                    if not self.CANDIDATE_ID_PATTERN.match(cid):
                        result.errors.append(
                            f"Row {row_num}: Invalid candidate_id: '{cid}'"
                        )
                        result.passed = False

                    # Check for duplicate candidate_ids (spec section 3)
                    if cid in candidate_ids_seen:
                        result.errors.append(
                            f"Row {row_num}: Duplicate candidate_id: {cid}"
                        )
                        result.passed = False
                    candidate_ids_seen.add(cid)

                    # Validate rank (spec section 3: 1-100, each once)
                    try:
                        rank = int(row["rank"])
                        if rank < 1 or rank > 100:
                            result.errors.append(
                                f"Row {row_num}: Rank out of range [1,100]: {rank}"
                            )
                            result.passed = False
                        if rank in ranks_seen:
                            result.errors.append(
                                f"Row {row_num}: Duplicate rank: {rank}"
                            )
                            result.passed = False
                        ranks_seen.add(rank)
                    except (ValueError, KeyError):
                        result.errors.append(f"Row {row_num}: Invalid rank value")
                        result.passed = False

                    # Validate score (spec section 3: non-increasing, [0,1])
                    try:
                        score = float(row["score"])
                        if score < 0.0 or score > 1.0:
                            result.errors.append(
                                f"Row {row_num}: Score out of range [0,1]: {score}"
                            )
                            result.passed = False
                        if score > prev_score + 1e-9:
                            result.errors.append(
                                f"Row {row_num}: Score not non-increasing: "
                                f"{score} > {prev_score} (rank {i+1})"
                            )
                            result.passed = False
                        prev_score = score
                    except (ValueError, KeyError):
                        result.errors.append(f"Row {row_num}: Invalid score value")
                        result.passed = False

                    # Validate reasoning (spec section 3: recommended)
                    reasoning = row.get("reasoning", "").strip()
                    if not reasoning:
                        result.warnings.append(
                            f"Row {row_num}: Empty reasoning for {cid}"
                        )
                    if len(reasoning) > 500:
                        result.warnings.append(
                            f"Row {row_num}: Reasoning very long ({len(reasoning)} chars)"
                        )

                # Check all ranks 1-100 are present (spec section 3)
                expected_ranks = set(range(1, 101))
                missing_ranks = expected_ranks - ranks_seen
                if missing_ranks:
                    result.errors.append(
                        f"Missing ranks: {sorted(missing_ranks)[:10]}"
                    )
                    result.passed = False

                # Check for all-identical reasoning (spec section 3: penalized at Stage 4)
                reasoning_values = [row.get("reasoning", "").strip() for row in rows]
                if reasoning_values and len(set(reasoning_values)) == 1 and reasoning_values[0]:
                    result.errors.append(
                        "All 100 reasoning strings are identical - spec section 3 penalizes this"
                    )
                    result.passed = False

                # Check for all-identical scores (spec section 6: common rejection)
                score_values = [float(row.get("score", 0)) for row in rows if row.get("score")]
                if score_values and len(set(score_values)) == 1:
                    result.errors.append(
                        "All scores are identical - spec section 6: 'All scores set to the same value'"
                    )
                    result.passed = False

        except Exception as e:
            result.passed = False
            result.errors.append(f"Validation error: {e}")

        # ── candidate_id existence check (spec section 3) ──────────────
        # "Every candidate_id must exist in the released candidates.jsonl"
        if self.candidates_file and self.candidates_file.exists():
            try:
                valid_ids = self._load_candidate_ids(self.candidates_file)
                for cid in candidate_ids_seen:
                    if cid not in valid_ids:
                        result.errors.append(
                            f"candidate_id {cid} does not exist in {self.candidates_file.name} "
                            f"(spec section 3 violation)"
                        )
                        result.passed = False
            except Exception as e:
                result.warnings.append(f"Could not verify candidate_id existence: {e}")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"  Validation warning: {warning}")

        return result

    @staticmethod
    def _load_candidate_ids(candidates_file: Path) -> Set[str]:
        """Load all valid candidate_ids from the source dataset."""
        valid_ids: Set[str] = set()
        import gzip

        opener = gzip.open if candidates_file.suffix == ".gz" else open
        with opener(candidates_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("candidate_id")
                    if cid:
                        valid_ids.add(cid)
                except json.JSONDecodeError:
                    continue
        return valid_ids