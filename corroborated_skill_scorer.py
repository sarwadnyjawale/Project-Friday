"""
corroborated_skill_scorer.py

Computes a corroborated score_C by blending the raw assessment score (score_C)
with evidence from career descriptions.

The raw score_C is based purely on platform assessment scores. This module
augments that with evidence that the skill actually appears in the candidate's
career history - providing a more complete and trustworthy signal.

Integration:
    - Called after EvidenceConsistencyEngine and ContextEvidenceAnalyzer
    - Receives feature_bundles, evidence_bundles, context_bundles, candidates
    - Returns {candidate_id: corroborated_C_score (0.0-1.0)}
    - rank.py blends the returned score with the raw score_C (60/40 weight)

Design:
    The corroborated score combines three signals:
    1. Raw assessment score (from feature_extractor score_C)
    2. JD-relevant skill corroboration ratio (from EvidenceConsistencyBundle)
    3. Context quality of relevant technical terms (from ContextEvidenceBundle)

    This avoids inflating candidates who list JD keywords without evidence,
    and avoids penalizing strong performers who didn't take the platform's
    specific assessments.
"""

import logging
from typing import Dict, List, Optional

from jd_analyzer import JDContext
from feature_extractor import FeatureBundle
from evidence_consistency_engine import EvidenceConsistencyBundle
from context_evidence_analyzer import ContextEvidenceBundle

logger = logging.getLogger("corroborated_skill_scorer")


class CorroboratedSkillScorer:
    """
    Produces a corroborated C-score per candidate.

    The corroborated score blends the platform's assessment score with
    evidence from career descriptions to produce a richer signal.
    """

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context

    def compute_all(
        self,
        feature_bundles: Dict[str, FeatureBundle],
        evidence_bundles: Dict[str, EvidenceConsistencyBundle],
        context_bundles: Dict[str, ContextEvidenceBundle],
        candidates: List[dict],
    ) -> Dict[str, float]:
        """
        Compute corroborated C scores for all candidates.

        Args:
            feature_bundles:  {candidate_id: FeatureBundle} from FeatureExtractor
            evidence_bundles: {candidate_id: EvidenceConsistencyBundle} from ECE
            context_bundles:  {candidate_id: ContextEvidenceBundle} from CEA
            candidates:       Original candidate list

        Returns:
            {candidate_id: corroborated_C_score (0.0-1.0)}
        """
        results: Dict[str, float] = {}

        for candidate in candidates:
            cid = candidate["candidate_id"]
            fb = feature_bundles.get(cid)
            eb = evidence_bundles.get(cid)
            cb = context_bundles.get(cid)

            try:
                results[cid] = self._compute_candidate(fb, eb, cb)
            except Exception as e:
                logger.warning(f"Corroborated scoring error for {cid}: {e}")
                # Fall back to raw score_C if available
                results[cid] = fb.score_C if fb is not None else 0.0

        return results

    def _compute_candidate(
        self,
        fb: Optional[FeatureBundle],
        eb: Optional[EvidenceConsistencyBundle],
        cb: Optional[ContextEvidenceBundle],
    ) -> float:
        """Compute corroborated C score for one candidate."""

        # ── Signal 1: Raw assessment score (0.0-1.0) ─────────────────────────
        # This is score_C from feature_extractor - platform's own assessment signal.
        raw_C = fb.score_C if fb is not None else 0.0

        # ── Signal 2: Career corroboration of JD-relevant skills ──────────────
        # How well does the career history back up the skills claimed?
        corroboration = 0.5  # Neutral default (no evidence either way)
        if eb is not None:
            # Use the overall corroboration score directly
            corroboration = eb.overall_corroboration_score
            # Boost if high-quality evidence exists
            if eb.high_quality_evidence_count >= 3:
                corroboration = min(1.0, corroboration + 0.10)
            # Penalize uncorroborated expert claims (keyword stuffing signal)
            if eb.uncorroborated_expert_claims >= 3:
                corroboration = max(0.0, corroboration - 0.10)

        # ── Signal 3: Context quality around JD terms ─────────────────────────
        # Strong positive contexts (built/deployed at scale) indicate genuine depth.
        context_quality = 0.5  # Neutral default
        if cb is not None:
            context_quality = cb.overall_context_quality
            # Quantified claims are a strong positive signal
            if cb.quantified_claim_count >= 2:
                context_quality = min(1.0, context_quality + 0.05)

        # ── Blend all three signals ───────────────────────────────────────────
        # Weights: raw_C anchors the result; corroboration and context refine it.
        # When no assessments were taken (raw_C is 0.5 neutral), the evidence
        # signals carry more weight.
        if raw_C > 0.0:
            # Assessments exist: weight raw_C higher
            blended = (
                raw_C * 0.50
                + corroboration * 0.30
                + context_quality * 0.20
            )
        else:
            # No assessments: rely more on career evidence
            blended = (
                corroboration * 0.55
                + context_quality * 0.45
            )

        return max(0.0, min(1.0, blended))
