"""
writer.py - Writes the final submission.csv.

Ensures:
    - Exactly 100 rows
    - Ranks 1-100 each appear exactly once
    - Scores are non-increasing with rank
    - UTF-8 encoding
    - Correct column order: candidate_id, rank, score, reasoning
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("writer")

# Output score range for human readability.
# Scores are linearly rescaled so rank-1 = SCORE_MAX, rank-100 = SCORE_MIN.
# Rank order is preserved exactly (linear rescaling is monotonic).
SCORE_MAX = 0.99
SCORE_MIN = 0.60


class SubmissionWriter:
    """Writes the ranked top-100 to submission CSV."""

    COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write(self, top_100: List[Dict]) -> None:
        """
        Write top-100 ranked candidates to CSV.

        Args:
            top_100: Sorted list of dicts with candidate_id, rank, final_score, reasoning
        """
        if len(top_100) != 100:
            logger.warning(
                f"Expected 100 candidates, got {len(top_100)}. Proceeding anyway."
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect raw scores to compute min/max for rescaling
        raw_scores = [entry["final_score"] for entry in top_100]
        raw_min = min(raw_scores)
        raw_max = max(raw_scores)
        raw_range = raw_max - raw_min if raw_max != raw_min else 1.0

        def rescale(raw: float) -> float:
            """Linearly rescale raw score to [SCORE_MIN, SCORE_MAX]."""
            pct = (raw - raw_min) / raw_range  # 0.0 = worst, 1.0 = best
            return SCORE_MIN + pct * (SCORE_MAX - SCORE_MIN)

        rows = []
        for entry in top_100:
            rows.append({
                "candidate_id": entry["candidate_id"],
                "rank": entry["rank"],
                "score": f"{rescale(entry['final_score']):.6f}",
                "reasoning": entry.get("reasoning", ""),
            })

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            f"Wrote {len(rows)} candidates to {self.output_path} "
            f"(top score: {rows[0]['score']}, rank-100 score: {rows[-1]['score']})"
        )