"""
confidence_estimator.py

Estimates score confidence for each candidate based on:
1. Number of independent corroborating signals
2. Internal consistency of those signals
3. Evidence quality distribution
4. Profile completeness and data reliability

The confidence score modifies how the Ranker resolves close-score ties.
A candidate with confidence 0.90 and score 0.75 should rank above a
candidate with confidence 0.40 and score 0.76 - the difference in score
is smaller than the difference in confidence.

Integration:
    - Called after all scoring modules, before Ranker
    - Produces {candidate_id: confidence_score (0.0-1.0)}
    - Ranker uses confidence for tie-breaking and near-tie resolution
    - ConfidenceAdjustedScore = score x (0.90 + 0.10 x confidence)
      This means confidence can shift the effective score by +-10%

Why this does NOT replace the existing Ranker:
    The Ranker's primary sort is still by final_score.
    Confidence is a secondary adjustment that affects close-score pairs.
    The existing tie-breaking (candidate_id ascending) is preserved when
    confidence-adjusted scores are also equal.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from feature_extractor import FeatureBundle
from evidence_consistency_engine import EvidenceConsistencyBundle
from context_evidence_analyzer import ContextEvidenceBundle

logger = logging.getLogger("confidence_estimator")


@dataclass
class ConfidenceBundle:
    """
    Confidence assessment for a single candidate's score.
    """
    candidate_id: str

    # Component confidence signals (0.0-1.0 each)
    assessment_confidence: float = 0.5    # Based on number and relevance of assessments
    career_evidence_confidence: float = 0.5  # Based on specificity of career descriptions
    signal_consistency_confidence: float = 0.5  # Based on agreement across signals
    profile_completeness_confidence: float = 0.5

    # Overall confidence
    overall_confidence: float = 0.5

    # Confidence-adjusted score (set by the Ranker)
    confidence_adjusted_score: float = 0.0

    # Explanation for reasoning
    confidence_narrative: str = ""


class ConfidenceEstimator:
    """
    Estimates how confident we should be in each candidate's score.

    The confidence model has four components:

    1. Assessment confidence: Did the candidate take relevant assessments?
       More relevant assessments = higher confidence in score_C.

    2. Career evidence confidence: Are the career descriptions specific and
       detailed? Vague descriptions mean we have less evidence to work with.

    3. Signal consistency confidence: Do different independent signals agree?
       When score_A (career text), score_C (assessments), and behavioral signals
       all point in the same direction, confidence is high.

    4. Profile completeness confidence: Incomplete profiles may be hiding
       negative information.
    """

    def estimate_all(
        self,
        candidates: List[dict],
        feature_bundles: Dict[str, FeatureBundle],
        evidence_bundles: Dict[str, EvidenceConsistencyBundle],
        scored_candidates: Dict[str, Dict],
    ) -> Dict[str, ConfidenceBundle]:
        """
        Compute confidence bundles for all candidates.

        Args:
            candidates: Original candidate list
            feature_bundles: From FeatureExtractor
            evidence_bundles: From EvidenceConsistencyEngine
            scored_candidates: From ScoringEngine (needed for signal consistency check)

        Returns:
            {candidate_id: ConfidenceBundle}
        """
        results: Dict[str, ConfidenceBundle] = {}

        for candidate in candidates:
            cid = candidate["candidate_id"]
            fb = feature_bundles.get(cid)
            eb = evidence_bundles.get(cid)
            scores = scored_candidates.get(cid, {})

            try:
                results[cid] = self._estimate_candidate(
                    candidate, fb, eb, scores
                )
            except Exception as e:
                logger.warning(f"Confidence estimation error for {cid}: {e}")
                results[cid] = ConfidenceBundle(candidate_id=cid)

        return results

    def _estimate_candidate(
        self,
        candidate: dict,
        feature_bundle: Optional[FeatureBundle],
        evidence_bundle: Optional[EvidenceConsistencyBundle],
        scores: Dict,
    ) -> ConfidenceBundle:
        """Estimate confidence for a single candidate."""
        bundle = ConfidenceBundle(candidate_id=candidate["candidate_id"])
        signals = candidate["redrob_signals"]
        career = candidate.get("career_history", [])
        skills_list = candidate.get("skills", [])

        # ── Component 1: Assessment confidence ──────────────────────────────
        assessment_scores = signals.get("skill_assessment_scores") or {}
        relevant_assessments = {
            k: v for k, v in assessment_scores.items()
            if self._is_assessment_relevant(k)
        }
        if len(relevant_assessments) >= 4:
            bundle.assessment_confidence = 0.95
        elif len(relevant_assessments) >= 2:
            bundle.assessment_confidence = 0.80
        elif len(relevant_assessments) >= 1:
            bundle.assessment_confidence = 0.65
        else:
            bundle.assessment_confidence = 0.40  # No assessments - lower confidence

        # ── Component 2: Career evidence confidence ──────────────────────────
        # Based on specificity of career descriptions
        total_desc_chars = sum(
            len(r.get("description") or "") for r in career
        )
        avg_desc_chars = total_desc_chars / max(1, len(career))

        # Also check evidence bundle for corroboration quality
        if evidence_bundle is not None:
            corroboration = evidence_bundle.overall_corroboration_score
            hq_count = evidence_bundle.high_quality_evidence_count
        else:
            corroboration = 0.5
            hq_count = 0

        if avg_desc_chars >= 800 and hq_count >= 3:
            bundle.career_evidence_confidence = 0.92
        elif avg_desc_chars >= 500 and corroboration >= 0.50:
            bundle.career_evidence_confidence = 0.78
        elif avg_desc_chars >= 300:
            bundle.career_evidence_confidence = 0.60
        elif avg_desc_chars >= 100:
            bundle.career_evidence_confidence = 0.45
        else:
            bundle.career_evidence_confidence = 0.25  # Very short descriptions

        # ── Component 3: Signal consistency confidence ────────────────────────
        # Do different independent signals agree with each other?
        score_A = scores.get("A", 0.5)
        score_B = scores.get("B", 0.5)
        score_C_raw = scores.get("C", 0.5)
        score_D = scores.get("D", 0.5)

        # Compute variance across the primary technical signals
        technical_scores = [score_A, score_B, score_C_raw]
        mean_technical = sum(technical_scores) / len(technical_scores)
        variance = sum((s - mean_technical) ** 2 for s in technical_scores) / len(technical_scores)
        std_dev = variance ** 0.5

        # Low std_dev = signals agree = high confidence
        if std_dev <= 0.10:
            base_consistency = 0.95
        elif std_dev <= 0.15:
            base_consistency = 0.80
        elif std_dev <= 0.22:
            base_consistency = 0.65
        elif std_dev <= 0.30:
            base_consistency = 0.50
        else:
            base_consistency = 0.30  # Signals disagree strongly

        # Detect specific conflict patterns the PDF identifies as low-confidence:
        # (1) High retrieval vocabulary (score_A) + low production depth (score_B)
        #     = surface terms without operational evidence
        conflict_penalty = 0.0
        if score_A >= 0.55 and score_B <= 0.25:
            conflict_penalty += 0.12  # Vocabulary without production depth

        # (2) High assessment score (score_C) + no career evidence (score_A low)
        #     = tested well but career history doesn't support the skill
        if score_C_raw >= 0.65 and score_A <= 0.20:
            conflict_penalty += 0.10  # Assessment score without career corroboration

        # (3) Strong career quality (score_D) + no ML content in career (score_A + score_B low)
        #     = well-credentialed but not in the right domain
        if score_D >= 0.70 and (score_A + score_B) <= 0.30:
            conflict_penalty += 0.08  # Career quality signal without domain relevance

        bundle.signal_consistency_confidence = max(
            0.15, base_consistency - conflict_penalty
        )

        # ── Component 3b: Summary cross-validation ────────────────────────────
        # PDF: "Never use the summary as a primary feature. Use it only as a
        # cross-validation field." If summary claims JD-relevant technical terms
        # that career history does not corroborate, reduce confidence.
        # This ONLY reduces confidence — it never increases score.
        summary_penalty = self._check_summary_consistency(
            candidate, career
        )
        bundle.signal_consistency_confidence = max(
            0.15, bundle.signal_consistency_confidence - summary_penalty
        )
        completeness = signals.get("profile_completeness_score", 50) or 50
        verified_email = signals.get("verified_email", False)
        verified_phone = signals.get("verified_phone", False)
        has_github = signals.get("github_activity_score", -1) != -1
        linkedin = signals.get("linkedin_connected", False)

        verification_count = sum([verified_email, verified_phone, has_github, linkedin])
        completeness_score = completeness / 100.0
        verification_bonus = verification_count * 0.05

        bundle.profile_completeness_confidence = min(
            1.0, completeness_score * 0.80 + verification_bonus
        )

        # ── Overall confidence: weighted average ──────────────────────────────
        # Signal consistency is most important - it catches conflicting signals
        # Career evidence is second - it is the primary trust source
        # Assessment and completeness are secondary
        bundle.overall_confidence = (
            bundle.signal_consistency_confidence * 0.35
            + bundle.career_evidence_confidence * 0.35
            + bundle.assessment_confidence * 0.20
            + bundle.profile_completeness_confidence * 0.10
        )

        # ── Build confidence narrative for reasoning ──────────────────────────
        bundle.confidence_narrative = self._build_narrative(bundle)

        return bundle

    def _check_summary_consistency(
        self, candidate: dict, career: list
    ) -> float:
        """
        Check if summary claims JD-relevant technical terms that career history
        doesn't support. Returns a penalty (0.0-0.20) to subtract from
        signal_consistency_confidence.

        The PDF says summaries are heavily templated and should only function
        as cross-validation. A summary that claims expertise not found anywhere
        in the career history is a consistency red flag.
        """
        profile = candidate.get("profile", {})
        summary = (profile.get("summary") or "").lower()
        if not summary or len(summary) < 50:
            return 0.0  # No summary to check

        # JD-critical technical terms that are specific enough to be meaningful
        # (excludes generic terms like "python", "machine learning" that appear everywhere)
        HIGH_SPECIFICITY_JD_TERMS = {
            "faiss", "hnsw", "pinecone", "weaviate", "qdrant", "milvus",
            "sentence-transformer", "dense retrieval", "bi-encoder", "cross-encoder",
            "embedding drift", "index refresh", "retrieval quality",
            "hybrid search", "learning to rank", "ndcg", "mrr", "bm25",
            "vector database", "vector index", "rag", "retrieval-augmented",
            "openai embedding", "bge", "e5 embedding",
        }

        # Build career description text for cross-validation
        all_career_text = " ".join(
            (r.get("description") or "").lower() for r in career
        )

        # Count terms claimed in summary but absent from career descriptions
        claimed_in_summary = [
            term for term in HIGH_SPECIFICITY_JD_TERMS if term in summary
        ]
        if not claimed_in_summary:
            return 0.0  # Summary makes no high-specificity claims

        uncorroborated = [
            term for term in claimed_in_summary
            if term not in all_career_text
        ]

        if not uncorroborated:
            return 0.0  # All summary claims are supported by career history

        # Penalty scales with the ratio of uncorroborated claims
        uncorroborated_ratio = len(uncorroborated) / len(claimed_in_summary)
        if uncorroborated_ratio >= 0.75:
            return 0.20  # Most high-specificity claims are uncorroborated
        elif uncorroborated_ratio >= 0.50:
            return 0.12
        elif uncorroborated_ratio >= 0.25:
            return 0.06
        else:
            return 0.0

    def _is_assessment_relevant(self, assessment_name: str) -> bool:
        """Check if an assessment name is relevant to the JD."""
        relevant_keywords = {
            "retrieval", "faiss", "pinecone", "weaviate", "embedding",
            "recommendation", "nlp", "rag", "learning to rank",
            "machine learning", "deep learning", "python", "pytorch",
        }
        name_lower = assessment_name.lower()
        return any(kw in name_lower for kw in relevant_keywords)

    def _build_narrative(self, bundle: ConfidenceBundle) -> str:
        """Build a short narrative explaining confidence level."""
        conf = bundle.overall_confidence
        if conf >= 0.85:
            return "high confidence: multiple independent signals agree"
        elif conf >= 0.70:
            return "good confidence: career evidence corroborates claims"
        elif conf >= 0.55:
            return "moderate confidence: some signals ambiguous"
        elif conf >= 0.40:
            return "low confidence: limited corroborating evidence"
        else:
            return "very low confidence: signals inconsistent or sparse"

    def apply_confidence_adjustment(
        self,
        scored_candidates: Dict[str, Dict],
        confidence_bundles: Dict[str, ConfidenceBundle],
    ) -> None:
        """
        Apply confidence adjustment to final_score in-place.

        When Config.USE_NEW_CONFIDENCE is True (default):
            Does NOT apply +/-10% adjustment. Confidence is stored
            for tie-breaking only. This prevents the unvalidated confidence
            model from dominating the ranking when base scores are compressed.

        When Config.USE_NEW_CONFIDENCE is False (legacy mode):
            confidence_adjusted_score = final_score x (0.90 + 0.10 x confidence)
            This means:
                - A candidate with confidence=1.0 gets 1.0x of their final_score
                - A candidate with confidence=0.0 gets 0.90x of their final_score
                - Maximum influence: +-10% of final_score
        """
        new_mode = Config.USE_NEW_CONFIDENCE

        for cid, score_dict in scored_candidates.items():
            confidence = confidence_bundles.get(cid)
            final_score = score_dict.get("final_score", 0.0)

            if confidence is None:
                score_dict["confidence_adjusted_score"] = final_score
                continue

            if new_mode:
                # B4: New mode - no global adjustment, confidence stored for tie-breaking
                score_dict["confidence_adjusted_score"] = final_score
                score_dict["confidence_score"] = confidence.overall_confidence
            else:
                # Legacy mode: +/-10% adjustment
                adjusted = final_score * (0.90 + 0.10 * confidence.overall_confidence)
                score_dict["confidence_adjusted_score"] = min(1.0, adjusted)

            # Also store in confidence bundle for reasoning
            confidence.confidence_adjusted_score = score_dict["confidence_adjusted_score"]