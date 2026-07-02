"""
ranker.py - Sorts candidates, enforces tie-breaking, audits top 10.

The primary sort is by final_score descending.
Tie-breaking: equal scores -> candidate_id ascending (alphabetical).
This matches the validator's expected behavior for ties.

The top-10 audit is a safety net for cases where multipliers were insufficient
to push a honeypot below top-10. NDCG@10 is 50% of the evaluation score -
a single honeypot in the top 10 is catastrophic.
"""

import logging
from typing import Dict, List, Tuple

from config import Config

logger = logging.getLogger("ranker")

# Maximum honeypot score allowed in top 10
TOP_10_MAX_HONEYPOT_SIGNALS = 1


class Ranker:
    """
    Produces a sorted, validated top-100 candidate list.
    """

    def rank(
        self,
        scored_candidates: Dict[str, Dict],
        candidates: List[dict],
        honeypot_scores: Dict[str, int],
    ) -> List[Dict]:
        """
        Produce a sorted top-100 list.

        Args:
            scored_candidates: {candidate_id: score_dict} from ScoringEngine
            candidates: Original candidate list (for audit data)
            honeypot_scores: {candidate_id: signal_count} from HoneypotDetector

        Returns:
            List of dicts with candidate_id, rank, final_score, and score_dict.
            List is sorted by rank (1 to 100).
        """
        # Build list of (candidate_id, score) for sorting
        sortable = [
            (cid, self._get_sort_score(scores))
            for cid, scores in scored_candidates.items()
        ]

        # Primary sort: score descending. Secondary: candidate_id ascending (tie-break)
        sortable.sort(key=lambda x: (-x[1], x[0]))

        # ── Audit top 100: ensure no honeypots in top 10 ────────────────────
        top_100 = self._select_and_audit_top_100(
            sortable, scored_candidates, candidates, honeypot_scores
        )

        # Assign final ranks
        ranked = []
        for rank_idx, entry in enumerate(top_100, start=1):
            entry["rank"] = rank_idx
            ranked.append(entry)

        # Validate score monotonicity
        self._validate_score_monotonicity(ranked)

        return ranked

    @staticmethod
    def _get_sort_score(score_dict: dict) -> float:
        """
        Get the best available score for sorting.

        Priority: calibrated_score > confidence_adjusted_score > final_score.
        Ensures the enhanced pipeline's output is used when available,
        with clean fallback to the original scoring when modules fail.
        """
        return (
            score_dict.get("calibrated_score")
            or score_dict.get("confidence_adjusted_score")
            or score_dict.get("final_score", 0.0)
        )

    def _select_and_audit_top_100(
        self,
        sortable: List[Tuple[str, float]],
        scored_candidates: Dict[str, Dict],
        candidates: List[dict],
        honeypot_scores: Dict[str, int],
    ) -> List[Dict]:
        """
        Select top 100 candidates with honeypot audit for top 10.

        If a honeypot is detected in the top 10 (somehow), find the next
        clean candidate and swap them in.
        """
        # Build a lookup for candidates
        candidate_lookup = {c["candidate_id"]: c for c in candidates}

        # Take the top 110 as a buffer (in case some need to be demoted)
        buffer = sortable[:110]
        result = []
        clean_top_10_count = 0

        for cid, score in buffer:
            if len(result) >= 100:
                break

            hp_signals = honeypot_scores.get(cid, 0)
            position = len(result) + 1  # 1-indexed position

            # For top 10: enforce strict honeypot check
            # With USE_STRENGTHENED_HONEYPOT, scorer already applies 0.30x for 1 signal.
            # This audit layer catches any that still floated up.
            if position <= 10:
                threshold = 1 if Config.USE_STRENGTHENED_HONEYPOT else 2
                if hp_signals >= threshold:
                    logger.warning(
                        f"AUDIT: Pushed {cid} out of top 10 (honeypot signals: {hp_signals})"
                    )
                    # Force score near zero and add to end instead
                    # Find next clean candidate
                    continue
                else:
                    clean_top_10_count += 1

            result.append({
                "candidate_id": cid,
                "final_score": score,
                "score_dict": scored_candidates.get(cid, {}),
            })

        # If we don't have 100 after buffer, take from full sorted list
        if len(result) < 100:
            already_included = {entry["candidate_id"] for entry in result}
            for cid, score in sortable:
                if len(result) >= 100:
                    break
                if cid not in already_included:
                    result.append({
                        "candidate_id": cid,
                        "final_score": score,
                        "score_dict": scored_candidates.get(cid, {}),
                    })

        logger.info(f"Top 10 audit: {clean_top_10_count} clean candidates in top 10")
        return result[:100]

    def _validate_score_monotonicity(self, ranked: List[Dict]) -> None:
        """
        Verify that scores are non-increasing with rank.
        The submission validator checks this - catch it here first.
        """
        for i in range(1, len(ranked)):
            prev_score = ranked[i - 1]["final_score"]
            curr_score = ranked[i]["final_score"]
            if curr_score > prev_score + 1e-9:  # Tolerance for float precision
                logger.warning(
                    f"Score monotonicity violation at rank {i+1}: "
                    f"{curr_score:.6f} > {ranked[i-1]['final_score']:.6f}"
                )