"""
scorer.py - Final score computation with weighted components and multipliers.

The scoring engine applies the weighted formula from the architecture spec and
then applies graduated multipliers for trap patterns and honeypot detection.

Key design decision: Multipliers are applied multiplicatively, not additively.
This means penalties compound. A consulting-only (0.6x) behavioral-dead (0.7x)
candidate gets a combined multiplier of 0.42x - correctly reflecting that
both problems together are worse than either alone.

The behavioral score modifies the final score through a soft multiplier:
    final_score = technical_score x (0.85 + 0.15 x behavioral_score)
This ensures behavioral signals never dominate - they can only shift the score
by +-15%, preventing a great technical candidate from being buried by low
platform activity.
"""

import logging
from typing import Dict

from config import Config
from feature_extractor import FeatureBundle

logger = logging.getLogger("scorer")

# Component weights - must sum to 1.0
COMPONENT_WEIGHTS = {
    "A": 0.35,  # Core technical relevance
    "B": 0.20,  # Production ML depth
    "C": 0.12,  # Assessment score match
    "D": 0.13,  # Career quality and trajectory
    "E": 0.10,  # Behavioral availability
    "F": 0.05,  # Location and logistics
    "G": 0.03,  # Education signal
    "H": 0.02,  # GitHub activity
}

# Multipliers for trap patterns
MULTIPLIERS = {
    "honeypot_2": 0.20,      # 2 honeypot signals
    "honeypot_3plus": 0.05,  # 3+ honeypot signals
    "consulting_only": 0.60,
    "behavioral_dead": 0.70,
    "pure_researcher": 0.50,
    "domain_mismatch": 0.65,
    "langchain_only": 0.75,  # Not a hard disqualifier, but significant penalty
    "recent_hype_pivot": 0.80,
    "title_chaser": 0.85,
    "keyword_stuffer": 0.85,
    "certification_padder": 0.80,
    "experience_gap": 0.75,
}


class ScoringEngine:
    """
    Computes final scores for all candidates using feature bundles.
    """

    def score_all(
        self,
        candidates: list,
        feature_bundles: Dict[str, FeatureBundle],
        honeypot_scores: Dict[str, int],
        trap_flags: Dict[str, Dict[str, bool]],
    ) -> Dict[str, Dict]:
        """
        Score all candidates and return {candidate_id: score_dict}.

        score_dict keys:
            A, B, C, D, E, F, G, H - component scores
            raw_score               - weighted sum before multipliers
            multiplier              - product of all applicable multipliers
            final_score             - raw_score x multiplier
            score_breakdown         - dict of applied multipliers for reasoning
        """
        results: Dict[str, Dict] = {}

        for candidate in candidates:
            cid = candidate["candidate_id"]
            bundle = feature_bundles.get(cid)
            if bundle is None:
                results[cid] = self._zero_score(cid)
                continue

            hp_count = honeypot_scores.get(cid, 0)
            traps = trap_flags.get(cid, {})

            try:
                score_dict = self._score_candidate(
                    bundle=bundle,
                    honeypot_count=hp_count,
                    trap_flags=traps,
                )
                results[cid] = score_dict
            except Exception as e:
                logger.warning(f"Scoring error for {cid}: {e}")
                results[cid] = self._zero_score(cid)

        return results

    def _score_candidate(
        self,
        bundle: FeatureBundle,
        honeypot_count: int,
        trap_flags: Dict[str, bool],
    ) -> Dict:
        """Compute final score for a single candidate."""

        # ── Step 1: Near-zero for 3+ honeypot signals ──────────────────────
        if honeypot_count >= 3:
            mult = Config.HONEYPOT_3PLUS_MULT if Config.USE_STRENGTHENED_HONEYPOT else MULTIPLIERS["honeypot_3plus"]
            return {
                "A": bundle.score_A, "B": bundle.score_B, "C": bundle.score_C,
                "D": bundle.score_D, "E": bundle.score_E, "F": bundle.score_F,
                "G": bundle.score_G, "H": bundle.score_H,
                "raw_score": 0.02,
                "multiplier": mult,
                "final_score": 0.02 * mult,
                "score_breakdown": {"honeypot_3plus": True},
            }

        # ── Step 2: Compute weighted raw score ──────────────────────────────
        # Note: Behavioral score (E) is handled via the soft multiplier,
        # NOT included in the base weighted sum. This prevents behavioral
        # signals from dominating the technical evaluation.
        raw_score = (
            bundle.score_A * COMPONENT_WEIGHTS["A"]
            + bundle.score_B * COMPONENT_WEIGHTS["B"]
            + bundle.score_C * COMPONENT_WEIGHTS["C"]
            + bundle.score_D * COMPONENT_WEIGHTS["D"]
            + bundle.score_F * COMPONENT_WEIGHTS["F"]
            + bundle.score_G * COMPONENT_WEIGHTS["G"]
            + bundle.score_H * COMPONENT_WEIGHTS["H"]
        )
        # Rescale: we excluded E (10% weight), so raw_score maxes at 0.90
        # Normalize to 1.0 ceiling
        raw_score = raw_score / 0.90

        # ── Step 3: Apply behavioral soft multiplier ─────────────────────────
        # This ensures behavioral signals can shift score by at most +-15%
        behavioral_multiplier = 0.85 + 0.15 * bundle.score_E
        raw_score_with_behavior = raw_score * behavioral_multiplier

        # ── Step 3b: Career depth bonus ──────────────────────────────────────
        # Reward deep product company experience to differentiate top candidates.
        # Max +0.05 for candidates with 5+ relevant years at product companies
        # with strong tenure stability. Creates natural score spread in top-100.
        depth_bonus = 0.0
        if bundle.has_product_company and bundle.effective_relevant_years >= 3:
            years_factor = min(1.0, bundle.effective_relevant_years / 8.0)
            tenure_factor = bundle.score_D3
            depth_bonus = 0.05 * years_factor * tenure_factor
        raw_score_with_behavior += depth_bonus

        # ── Step 4: Apply hard multipliers ──────────────────────────────────
        combined_multiplier = 1.0
        score_breakdown: Dict[str, bool] = {}

        if honeypot_count == 2:
            hp_mult = Config.HONEYPOT_2_SIGNALS_MULT if Config.USE_STRENGTHENED_HONEYPOT else MULTIPLIERS["honeypot_2"]
            combined_multiplier *= hp_mult
            score_breakdown["honeypot_2"] = True

        if Config.USE_STRENGTHENED_HONEYPOT and honeypot_count == 1:
            combined_multiplier *= Config.HONEYPOT_1_HIGH_CONFIDENCE_MULT
            score_breakdown["honeypot_1_high_conf"] = True

        if trap_flags.get("consulting_only", False):
            combined_multiplier *= MULTIPLIERS["consulting_only"]
            score_breakdown["consulting_only"] = True

        if trap_flags.get("behavioral_dead", False):
            bd_mult = Config.BEHAVIORAL_DEAD_MULT if Config.USE_IMPROVED_KEYWORD_STUFFER else MULTIPLIERS["behavioral_dead"]
            combined_multiplier *= bd_mult
            score_breakdown["behavioral_dead"] = True

        if trap_flags.get("pure_researcher", False):
            combined_multiplier *= MULTIPLIERS["pure_researcher"]
            score_breakdown["pure_researcher"] = True

        if trap_flags.get("domain_mismatch", False):
            combined_multiplier *= MULTIPLIERS["domain_mismatch"]
            score_breakdown["domain_mismatch"] = True

        if trap_flags.get("langchain_only", False):
            combined_multiplier *= MULTIPLIERS["langchain_only"]
            score_breakdown["langchain_only"] = True

        if trap_flags.get("recent_hype_pivot", False):
            combined_multiplier *= MULTIPLIERS["recent_hype_pivot"]
            score_breakdown["recent_hype_pivot"] = True

        if trap_flags.get("title_chaser", False):
            combined_multiplier *= MULTIPLIERS["title_chaser"]
            score_breakdown["title_chaser"] = True

        if trap_flags.get("keyword_stuffer", False):
            kw_mult = Config.KEYWORD_STUFFER_MULT if Config.USE_IMPROVED_KEYWORD_STUFFER else MULTIPLIERS["keyword_stuffer"]
            combined_multiplier *= kw_mult
            score_breakdown["keyword_stuffer"] = True

        if trap_flags.get("certification_padder", False):
            combined_multiplier *= MULTIPLIERS["certification_padder"]
            score_breakdown["certification_padder"] = True

        if trap_flags.get("experience_gap", False):
            combined_multiplier *= MULTIPLIERS["experience_gap"]
            score_breakdown["experience_gap"] = True

        final_score = min(1.0, raw_score_with_behavior * combined_multiplier)

        return {
            "A": bundle.score_A,
            "B": bundle.score_B,
            "C": bundle.score_C,
            "D": bundle.score_D,
            "E": bundle.score_E,
            "F": bundle.score_F,
            "G": bundle.score_G,
            "H": bundle.score_H,
            "raw_score": raw_score,
            "behavioral_multiplier": behavioral_multiplier,
            "multiplier": combined_multiplier,
            "final_score": final_score,
            "score_breakdown": score_breakdown,
        }

    @staticmethod
    def _zero_score(candidate_id: str) -> Dict:
        """Return a near-zero score for candidates with extraction errors."""
        return {
            "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
            "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0,
            "raw_score": 0.0,
            "behavioral_multiplier": 1.0,
            "multiplier": 1.0,
            "final_score": 0.0,
            "score_breakdown": {"extraction_error": True},
        }