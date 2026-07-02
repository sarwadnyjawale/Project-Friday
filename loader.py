"""
loader.py - Streaming candidate loader for gzipped or plain JSONL.

Design notes:
    - Streams line-by-line: never holds full file in memory
    - Handles both .gz and plain .jsonl transparently
    - Skips malformed JSON lines with a warning (defensive - don't crash on bad data)
    - The `limit` parameter enables fast dry-run testing
    - Returns a list of dicts - downstream modules expect this structure
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loader")


class CandidateLoader:
    """
    Streams candidates from a JSONL or JSONL.GZ file.

    The file format is one JSON object per line. Each object must have
    a 'candidate_id' field matching CAND_XXXXXXX pattern.
    """

    def __init__(self, path: Path, limit: Optional[int] = None) -> None:
        """
        Args:
            path:   Path to candidates.jsonl.gz or candidates.jsonl
            limit:  If set, stop after loading this many candidates (dry-run)
        """
        self.path = path
        self.limit = limit
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Candidates file not found: {self.path}")
        if self.path.stat().st_size == 0:
            raise ValueError(f"Candidates file is empty: {self.path}")

    def _open_file(self):
        """Return a file-like object supporting iteration, handling .gz transparently."""
        suffix = self.path.suffix.lower()
        if suffix == ".gz":
            return gzip.open(self.path, "rt", encoding="utf-8")
        elif suffix in (".jsonl", ".json", ".ndjson"):
            return open(self.path, "r", encoding="utf-8")
        else:
            # Try gzip first, fall back to plain text
            try:
                f = gzip.open(self.path, "rt", encoding="utf-8")
                f.read(1)
                f.seek(0)
                return f
            except Exception:
                return open(self.path, "r", encoding="utf-8")

    def load(self) -> list[dict]:
        """
        Load and return all candidates as a list of dicts.

        Skips malformed lines. Logs a warning for each skipped line.
        Stops at self.limit if set.

        Returns:
            List of candidate dicts in the order they appear in the file.
        """
        candidates = []
        skipped = 0
        line_number = 0

        with self._open_file() as f:
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue  # Skip empty lines silently

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped += 1
                    if skipped <= 5:  # Only log first 5 to avoid log spam
                        logger.warning(
                            f"Malformed JSON at line {line_number}: {e} - skipping"
                        )
                    continue

                if not isinstance(candidate, dict):
                    skipped += 1
                    logger.warning(
                        f"Line {line_number} is not a JSON object - skipping"
                    )
                    continue

                # Enforce presence of candidate_id - cannot rank without it
                if "candidate_id" not in candidate:
                    skipped += 1
                    logger.warning(
                        f"Line {line_number} missing candidate_id - skipping"
                    )
                    continue

                # Apply defensive defaults for missing top-level keys
                candidate = self._apply_defaults(candidate)
                candidates.append(candidate)

                if self.limit and len(candidates) >= self.limit:
                    logger.info(
                        f"Dry-run limit reached - stopping at {self.limit} candidates"
                    )
                    break

                # Progress logging every 10K
                if len(candidates) % 10_000 == 0:
                    logger.info(f"  Loaded {len(candidates):,} candidates...")

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} malformed lines out of {line_number} total"
            )

        logger.info(
            f"Loaded {len(candidates):,} candidates from {self.path.name}"
        )
        return candidates

    def _apply_defaults(self, candidate: dict) -> dict:
        """
        Apply defensive defaults for missing top-level sections.

        This ensures downstream modules never crash on KeyError for structural
        keys. Field-level missing values are handled per-module.

        The defaults are chosen to be neutral - they neither boost nor penalize
        a candidate simply for having a missing section.
        """
        candidate.setdefault("profile", {})
        candidate.setdefault("career_history", [])
        candidate.setdefault("education", [])
        candidate.setdefault("skills", [])
        candidate.setdefault("certifications", [])
        candidate.setdefault("languages", [])

        # Ensure redrob_signals has all expected sub-keys with neutral defaults
        signals_defaults = {
            "profile_completeness_score": 50,
            "signup_date": "2020-01-01",
            "last_active_date": "2020-01-01",  # Old date -> low availability score
            "open_to_work_flag": False,
            "profile_views_received_30d": 0,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5,  # Neutral
            "avg_response_time_hours": 72.0,
            "skill_assessment_scores": {},
            "connection_count": 0,
            "endorsements_received": 0,
            "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
            "preferred_work_mode": "flexible",
            "willing_to_relocate": False,
            "github_activity_score": -1,  # -1 = not linked
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.7,  # Neutral
            "offer_acceptance_rate": -1,  # -1 = no prior offers
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
        }
        candidate.setdefault("redrob_signals", {})
        for key, default in signals_defaults.items():
            candidate["redrob_signals"].setdefault(key, default)

        # Ensure profile has minimal defaults
        profile_defaults = {
            "anonymized_name": "Unknown",
            "headline": "",
            "summary": "",
            "location": "",
            "country": "",
            "years_of_experience": 0.0,
            "current_title": "",
            "current_company": "",
            "current_company_size": "unknown",
            "current_industry": "",
        }
        for key, default in profile_defaults.items():
            candidate["profile"].setdefault(key, default)

        return candidate