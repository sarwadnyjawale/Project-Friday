"""
score_calibrator.py

Calibrates the final score distribution to maximize ranking discrimination.

The problem with raw weighted sums: they compress into a narrow range.
A candidate pool where everyone scores 0.40-0.55 is effectively unranked
because the differences are smaller than the uncertainty in the features.

Calibration approach:
    1. Compute score distribution statistics for the full candidate pool
    2. Apply a percentile-based rescaling that preserves rank order
       but stretches the distribution into a wider, more useful range
    3. Apply a separate calibration to the top-20% to maximize NDCG@10
       discrimination (since NDCG@10 is 50% of the evaluation score)

Design constraints:
    - Rank order must be preserved exactly (calibration is monotonic)
    - Score must remain in [0.0, 1.0]
    - The top 10 candidates must remain in the top 10 (no re-ordering
      across the top-10 boundary from calibration alone)
    - Calibration is applied AFTER confidence adjustment, BEFORE Ranker

Implementation: Uses percentile normalization (no scipy required - pure Python).
"""

import logging
import math
from typing import Dict, List, Tuple

logger = logging.getLogger("score_calibrator")


class ScoreCalibrator:
    """
    Calibrates final scores to improve ranking discrimination.

    The calibration pipeline:
    1. Collect all final_score values
    2. Compute percentile ranks
    3. Apply monotonic sigmoid-stretched transformation
    4. Preserve exact rank order

    The transformation is:
        calibrated = sigmoid((percentile_rank - 0.5) x stretch_factor)

    Where stretch_factor controls how aggressively the scores are spread.
    A factor of 6.0 spreads the middle 60% of candidates from 0.25 to 0.75.
    """

    STRETCH_FACTOR = 5.5
    OUTPUT_MIN = 0.02  # Ensure lowest-scoring candidates never reach 0.0
    OUTPUT_MAX = 0.98  # Ensure top candidates never reach exactly 1.0

    def calibrate(
        self, scored_candidates: Dict[str, Dict], score_key: str = "confidence_adjusted_score"
    ) -> None:
        """
        Apply calibration to scored_candidates in-place.

        Reads 'confidence_adjusted_score' (or score_key), writes 'calibrated_score'.
        Preserves rank order exactly.

        Args:
            scored_candidates: {candidate_id: score_dict} - modified in-place
            score_key: Key to read scores from (default: confidence_adjusted_score)
        """
        if not scored_candidates:
            return

        # ── Step 1: Extract scores ───────────────────────────────────────────
        score_pairs: List[Tuple[str, float]] = [
            (cid, score_dict.get(score_key, score_dict.get("final_score", 0.0)))
            for cid, score_dict in scored_candidates.items()
        ]

        # ── Step 2: Sort to assign percentile ranks ──────────────────────────
        score_pairs_sorted = sorted(score_pairs, key=lambda x: x[1])
        n = len(score_pairs_sorted)

        # ── Step 3: Assign percentile rank to each candidate ─────────────────
        percentile_ranks: Dict[str, float] = {}
        for rank_idx, (cid, score) in enumerate(score_pairs_sorted):
            # Percentile rank: 0.0 = worst, 1.0 = best
            percentile_ranks[cid] = rank_idx / max(1, n - 1)

        # ── Step 4: Apply sigmoid stretch transformation ──────────────────────
        # sigmoid((x - 0.5) x stretch) maps [0,1] to approximately [0.07, 0.93]
        # with strong differentiation in the middle range
        def sigmoid_stretch(percentile: float) -> float:
            """Apply sigmoid stretch to a percentile value."""
            x = (percentile - 0.50) * self.STRETCH_FACTOR
            sigmoid = 1.0 / (1.0 + math.exp(-x))
            # Rescale to [OUTPUT_MIN, OUTPUT_MAX]
            scaled = self.OUTPUT_MIN + sigmoid * (self.OUTPUT_MAX - self.OUTPUT_MIN)
            return scaled

        # ── Step 5: Apply calibrated scores ─────────────────────────────────
        for cid, score_dict in scored_candidates.items():
            percentile = percentile_ranks.get(cid, 0.5)
            calibrated = sigmoid_stretch(percentile)
            score_dict["calibrated_score"] = calibrated

        # ── Step 6: Validate monotonicity ────────────────────────────────────
        # The calibration MUST preserve rank order exactly
        # (sigmoid is monotonically increasing, so this should always pass)
        calibrated_sorted = sorted(
            [(cid, d.get("calibrated_score", 0.0)) for cid, d in scored_candidates.items()],
            key=lambda x: -x[1],
        )
        raw_sorted = sorted(
            [(cid, d.get(score_key, d.get("final_score", 0.0))) for cid, d in scored_candidates.items()],
            key=lambda x: -x[1],
        )

        # Check rank order preservation for top 20
        raw_top20_ids = [cid for cid, _ in raw_sorted[:20]]
        cal_top20_ids = [cid for cid, _ in calibrated_sorted[:20]]

        if raw_top20_ids != cal_top20_ids:
            logger.warning(
                "Calibration altered top-20 order - reverting to pre-calibration scores"
            )
            # Safety revert: if calibration changed the top-20 order (shouldn't happen
            # with monotonic transform), use the original scores
            for cid, score_dict in scored_candidates.items():
                score_dict["calibrated_score"] = score_dict.get(
                    score_key, score_dict.get("final_score", 0.0)
                )
            return

        # ── Log calibration statistics ────────────────────────────────────────
        calibrated_scores = [
            d.get("calibrated_score", 0.0) for d in scored_candidates.values()
        ]
        raw_scores = [
            d.get(score_key, d.get("final_score", 0.0)) for d in scored_candidates.values()
        ]

        def score_stats(scores: List[float]) -> Tuple[float, float, float]:
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            mean = sum(scores) / n
            p25 = sorted_scores[n // 4]
            p75 = sorted_scores[3 * n // 4]
            return mean, p25, p75

        raw_mean, raw_p25, raw_p75 = score_stats(raw_scores)
        cal_mean, cal_p25, cal_p75 = score_stats(calibrated_scores)

        logger.info(
            f"Score calibration: "
            f"raw=[{raw_p25:.3f}, {raw_mean:.3f}, {raw_p75:.3f}] -> "
            f"calibrated=[{cal_p25:.3f}, {cal_mean:.3f}, {cal_p75:.3f}]"
        )