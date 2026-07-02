"""
evidence_consistency_engine.py

Measures the consistency between a candidate's claimed skills and the actual
evidence in their career descriptions.

The central insight: a skill listed as "expert" that never appears in career
descriptions is a red flag. A skill not listed at all but whose vocabulary
appears heavily in multiple career descriptions is a strong positive signal.

This module produces an EvidenceConsistencyBundle per candidate that the
ScoringEngine uses to adjust score_A and score_B.

Integration with existing architecture:
    - Called AFTER feature_extractor.py in rank.py
    - Receives candidates (list) and jd_context (JDContext)
    - Returns {candidate_id: EvidenceConsistencyBundle}
    - ScoringEngine.score_all() accepts Optional[Dict[str, EvidenceConsistencyBundle]]
      as a new keyword argument - backward compatible because it defaults to None

Why this improves NDCG@10:
    The candidates in the ground truth top 10 will have heavy career description
    corroboration for the skills that matter. The candidates who should NOT be in
    the top 10 will have skills that are listed but uncorroborated. This module
    makes that distinction explicit and measurable.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from jd_analyzer import JDContext

logger = logging.getLogger("evidence_consistency_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Evidence quality tiers - the scoring ladder for a single piece of evidence
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_QUALITY = {
    "high":    1.00,  # Specific claim + quantified outcome + technical detail
    "medium":  0.60,  # Technical claim with some specificity
    "low":     0.25,  # Mention in context but no specificity
    "mention": 0.05,  # Term present but in a weak or negative context
}

# Patterns that indicate HIGH quality evidence (built + outcome + scale)
# Require a technical verb + technical term + scale/outcome indicator
HIGH_QUALITY_PATTERNS = [
    # Deployment pattern: "deployed/built/shipped X serving/handling Y"
    r"\b(deployed|built|designed|implemented|developed|created|shipped|launched)\b"
    r"[^.]{0,60}"
    r"\b(serving|processing|handling|supporting|powering)\b"
    r"[^.]{0,40}"
    r"\b(\d+[kmb]?[\s\+]*(qps|rps|users|queries|requests|million|billion|thousand))\b",

    # Evaluation pattern: "improved X by Y% using Z"
    r"\b(improved|increased|reduced|decreased|optimized)\b"
    r"[^.]{0,40}"
    r"\b(by\s+\d+[\.\d]*\s*%|from\s+\d+|latency|ndcg|mrr|recall|precision)\b",

    # Production deployment with latency/SLA
    r"\b(production|real[\s-]time|user[\s-]facing)\b"
    r"[^.]{0,40}"
    r"\b(p\d{2}\s*latency|sla|slo|\d+\s*ms|\d+\s*milliseconds)\b",

    # Scale indicators with technical system
    r"\b(\d+\s*(million|billion|thousand)\s*(users|queries|documents|records|candidates|items))\b"
    r"[^.]{0,60}"
    r"\b(index|retrieval|search|recommendation|ranking|embedding)\b",
]

# Patterns that indicate NEGATIVE context (looked at it but didn't use it)
NEGATIVE_CONTEXT_PATTERNS = [
    r"\b(evaluated\s+but|considered\s+but|investigated\s+but|experimented\s+with\s+but)\b",
    r"\b(decided\s+against|chose\s+not\s+to\s+use|moved\s+away\s+from|replaced\s+by)\b",
    r"\b(tutorial|workshop|course|academic|research\s+project|prototype\s+only|proof[\s-]of[\s-]concept)\b",
    r"\b(familiar\s+with|exposure\s+to|basic\s+knowledge\s+of|introductory)\b",
]

# Technical action verbs that indicate hands-on experience
TECHNICAL_ACTION_VERBS = {
    "built", "designed", "implemented", "developed", "deployed", "shipped",
    "optimized", "trained", "fine-tuned", "evaluated", "benchmarked",
    "maintained", "operated", "scaled", "migrated", "integrated", "architected",
    "created", "led", "owned", "managed", "launched", "improved",
}


@dataclass
class SkillEvidence:
    """
    Evidence quality assessment for a single skill claim.
    """
    skill_name: str
    claimed_proficiency: Optional[str]  # From skills list: beginner/intermediate/advanced/expert
    claimed_duration_months: int
    is_in_jd: bool  # Whether this skill is relevant to the JD

    # Evidence found in career descriptions
    corroborating_sentences: List[str] = field(default_factory=list)
    evidence_quality_scores: List[float] = field(default_factory=list)
    roles_with_evidence: List[int] = field(default_factory=list)  # Indices into career_history
    has_negative_context: bool = False
    has_action_verb: bool = False

    # Computed scores
    corroboration_score: float = 0.0      # 0.0-1.0: how well career backs this skill
    consistency_score: float = 0.0        # 0.0-1.0: does proficiency claim match evidence depth
    evidence_recency_weight: float = 0.0  # Recency-weighted evidence strength


@dataclass
class EvidenceConsistencyBundle:
    """
    Complete evidence consistency analysis for one candidate.

    This bundle is consumed by the ScoringEngine to adjust score_A and score_B.
    It is also passed to the ReasoningGenerator for richer explanations.
    """
    candidate_id: str

    # Per-skill evidence
    skill_evidence: Dict[str, SkillEvidence] = field(default_factory=dict)

    # Aggregate signals
    jd_relevant_skills_count: int = 0
    corroborated_jd_skills_count: int = 0
    high_quality_evidence_count: int = 0   # Skills with HIGH tier evidence
    uncorroborated_expert_claims: int = 0  # Expert skills with zero career evidence
    negative_context_skills: List[str] = field(default_factory=list)

    # Summary scores (0.0-1.0)
    overall_corroboration_score: float = 0.0   # Mean corroboration across JD-relevant skills
    overall_consistency_score: float = 0.0     # Claims match evidence depth
    skill_to_career_alignment: float = 0.0     # Career descriptions support JD requirements
                                               # even WITHOUT skill claims

    # Adjustment factors for ScoringEngine
    score_A_adjustment: float = 0.0  # Additive adjustment to score_A (-0.15 to +0.15)
    score_B_adjustment: float = 0.0  # Additive adjustment to score_B (-0.10 to +0.10)

    # Top corroborated JD skills for reasoning
    top_corroborated_skills: List[str] = field(default_factory=list)
    top_uncorroborated_claims: List[str] = field(default_factory=list)


class EvidenceConsistencyEngine:
    """
    Analyzes consistency between claimed skills and career description evidence.

    Processing pipeline per candidate:
    1. Extract claimed skills from skills[] that are JD-relevant
    2. For each claimed skill, search career descriptions for corroborating sentences
    3. Score each corroborating sentence by evidence quality tier
    4. Check for negative context (evaluated but didn't use)
    5. Compute per-skill corroboration and consistency scores
    6. Aggregate to bundle-level scores
    7. Compute score_A and score_B adjustment factors

    Performance: O(N x S x D) where N=candidates, S=skills per candidate (~20),
    D=sentences per candidate (~50). For 100K candidates: ~100M operations.
    Target: 60 seconds. Achievable with Python string operations (no regex per sentence).
    """

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context
        # Pre-compile regex patterns for efficiency
        self._high_quality_patterns = [
            re.compile(p, re.IGNORECASE) for p in HIGH_QUALITY_PATTERNS
        ]
        self._negative_patterns = [
            re.compile(p, re.IGNORECASE) for p in NEGATIVE_CONTEXT_PATTERNS
        ]
        # Pre-build JD skill vocabulary for fast lookup
        self._jd_term_set = set(self.jd.term_weights.keys())
        # Build expanded skill synonym map
        self._skill_synonyms = self._build_skill_synonyms()

    def analyze_all(
        self, candidates: List[dict]
    ) -> Dict[str, EvidenceConsistencyBundle]:
        """
        Run evidence consistency analysis on all candidates.

        Returns {candidate_id: EvidenceConsistencyBundle}
        """
        results: Dict[str, EvidenceConsistencyBundle] = {}

        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            try:
                results[cid] = self._analyze_candidate(candidate)
            except Exception as e:
                logger.warning(f"Evidence analysis error for {cid}: {e}")
                results[cid] = EvidenceConsistencyBundle(candidate_id=cid)

            if (i + 1) % 10_000 == 0:
                logger.info(
                    f"  Evidence consistency: {i+1:,} candidates processed"
                )

        # Log summary statistics
        total_corroborated = sum(
            b.corroborated_jd_skills_count for b in results.values()
        )
        total_uncorroborated = sum(
            b.uncorroborated_expert_claims for b in results.values()
        )
        logger.info(
            f"Evidence consistency complete: "
            f"{total_corroborated:,} corroborated JD skills, "
            f"{total_uncorroborated:,} uncorroborated expert claims"
        )
        return results

    def _analyze_candidate(self, candidate: dict) -> EvidenceConsistencyBundle:
        """Full evidence consistency analysis for one candidate."""
        bundle = EvidenceConsistencyBundle(candidate_id=candidate["candidate_id"])
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        if not career:
            return bundle

        # ── Step 1: Extract and segment career description sentences ─────────
        # Sentence segmentation is important: we need to score individual sentences
        # not entire paragraphs, because a single paragraph can contain both
        # positive and negative context for the same technology.
        role_sentences = self._extract_role_sentences(career)

        # ── Step 2: Analyze each JD-relevant skill ───────────────────────────
        for skill_obj in skills:
            skill_name = (skill_obj.get("name") or "").strip()
            if not skill_name:
                continue

            # Check if this skill is JD-relevant
            skill_lower = skill_name.lower()
            is_jd_relevant = self._is_jd_relevant(skill_lower)

            skill_ev = SkillEvidence(
                skill_name=skill_name,
                claimed_proficiency=skill_obj.get("proficiency"),
                claimed_duration_months=skill_obj.get("duration_months", 0) or 0,
                is_in_jd=is_jd_relevant,
            )

            if is_jd_relevant:
                bundle.jd_relevant_skills_count += 1

            # ── Step 3: Find corroborating sentences in career descriptions ──
            self._find_corroborating_evidence(
                skill_ev, skill_lower, role_sentences
            )

            # ── Step 4: Score the skill evidence ─────────────────────────────
            self._score_skill_evidence(skill_ev)

            bundle.skill_evidence[skill_name] = skill_ev

            # ── Step 5: Accumulate bundle-level signals ───────────────────────
            if is_jd_relevant:
                if skill_ev.corroboration_score >= 0.30:
                    bundle.corroborated_jd_skills_count += 1
                if skill_ev.corroboration_score >= 0.70:
                    bundle.high_quality_evidence_count += 1

            if (
                skill_ev.claimed_proficiency == "expert"
                and skill_ev.corroboration_score < 0.15
                and is_jd_relevant
            ):
                bundle.uncorroborated_expert_claims += 1

            if skill_ev.has_negative_context:
                bundle.negative_context_skills.append(skill_name)

        # ── Step 6: Score career descriptions for JD alignment ───────────────
        # This catches candidates whose career descriptions are rich with JD
        # vocabulary even if they did not explicitly list those skills.
        # This is the "Tier 5 candidate" the JD organizers mentioned.
        bundle.skill_to_career_alignment = self._score_career_jd_alignment(
            role_sentences
        )

        # ── Step 7: Compute aggregate scores ─────────────────────────────────
        self._compute_aggregate_scores(bundle)

        # ── Step 8: Compute adjustment factors ───────────────────────────────
        self._compute_adjustment_factors(bundle)

        # ── Step 9: Build top skills lists for reasoning ──────────────────────
        jd_skills_sorted = sorted(
            [
                (name, ev)
                for name, ev in bundle.skill_evidence.items()
                if ev.is_in_jd
            ],
            key=lambda x: -x[1].corroboration_score,
        )
        bundle.top_corroborated_skills = [
            name for name, ev in jd_skills_sorted
            if ev.corroboration_score >= 0.40
        ][:5]
        bundle.top_uncorroborated_claims = [
            name for name, ev in jd_skills_sorted
            if ev.corroboration_score < 0.15 and ev.claimed_proficiency in ("expert", "advanced")
        ][:3]

        return bundle

    def _extract_role_sentences(
        self, career: List[dict]
    ) -> List[Tuple[str, int, float]]:
        """
        Extract individual sentences from career descriptions.

        Returns list of (sentence_text, role_index, recency_weight).
        Recency weight: most recent role = 1.0, each prior = -0.15 decay.
        """
        sorted_career = sorted(
            enumerate(career),
            key=lambda x: x[1].get("start_date") or "2000-01-01",
            reverse=True,  # Most recent first
        )

        sentences: List[Tuple[str, int, float]] = []
        for rank_idx, (orig_idx, role) in enumerate(sorted_career):
            recency_weight = max(0.25, 1.0 - rank_idx * 0.15)
            description = (role.get("description") or "").strip()
            if not description:
                continue
            # Sentence splitting: split on period, semicolon, or newline
            # Keep sentences that are at least 20 chars (avoid fragments)
            raw_sentences = re.split(r'[.;\n]+', description)
            for sent in raw_sentences:
                sent = sent.strip()
                if len(sent) >= 20:
                    sentences.append((sent.lower(), orig_idx, recency_weight))

        return sentences

    def _is_jd_relevant(self, skill_lower: str) -> bool:
        """Check if a skill name overlaps with JD vocabulary."""
        # Direct match
        if skill_lower in self._jd_term_set:
            return True
        # Partial match - skill word appears in JD term
        skill_words = [w for w in skill_lower.split() if len(w) > 3]
        return any(
            any(word in jd_term for word in skill_words)
            for jd_term in self._jd_term_set
        )

    def _find_corroborating_evidence(
        self,
        skill_ev: SkillEvidence,
        skill_lower: str,
        role_sentences: List[Tuple[str, int, float]],
    ) -> None:
        """
        Find sentences in career descriptions that mention this skill.
        Populates skill_ev.corroborating_sentences and related fields.
        """
        # Build search terms for this skill (skill + synonyms)
        search_terms = self._get_search_terms(skill_lower)

        for sentence, role_idx, recency_weight in role_sentences:
            # Check if any search term appears in this sentence
            term_found = any(term in sentence for term in search_terms)
            if not term_found:
                continue

            # Sentence contains a reference to this skill
            skill_ev.corroborating_sentences.append(sentence)
            if role_idx not in skill_ev.roles_with_evidence:
                skill_ev.roles_with_evidence.append(role_idx)

            # Check for negative context patterns
            has_negative = any(
                p.search(sentence) for p in self._negative_patterns
            )
            if has_negative:
                skill_ev.has_negative_context = True
                # Negative context - add low quality evidence
                skill_ev.evidence_quality_scores.append(
                    EVIDENCE_QUALITY["mention"] * recency_weight
                )
                continue

            # Check for technical action verbs
            has_action_verb = any(
                verb in sentence for verb in TECHNICAL_ACTION_VERBS
            )
            if has_action_verb:
                skill_ev.has_action_verb = True

            # Score evidence quality
            quality = self._score_sentence_quality(sentence)
            skill_ev.evidence_quality_scores.append(quality * recency_weight)

    def _score_sentence_quality(self, sentence: str) -> float:
        """
        Score the quality of a single corroborating sentence.
        Returns a value from EVIDENCE_QUALITY tiers.
        """
        # Check for high quality patterns (production + scale + outcome)
        for pattern in self._high_quality_patterns:
            if pattern.search(sentence):
                return EVIDENCE_QUALITY["high"]

        # Check for medium quality: technical verb + specific context
        has_action = any(verb in sentence for verb in TECHNICAL_ACTION_VERBS)
        has_numbers = bool(re.search(r'\b\d+[\.,]?\d*\s*(%|k|m|b|ms|s|gb|tb)\b', sentence))
        has_scale = any(
            kw in sentence for kw in {
                "million", "billion", "thousand", "scale", "production",
                "real-time", "user-facing", "latency", "throughput",
            }
        )

        if has_action and (has_numbers or has_scale):
            return EVIDENCE_QUALITY["medium"]
        elif has_action:
            return EVIDENCE_QUALITY["low"]
        else:
            return EVIDENCE_QUALITY["mention"]

    def _score_skill_evidence(self, skill_ev: SkillEvidence) -> None:
        """
        Compute corroboration_score, consistency_score, evidence_recency_weight.
        Modifies skill_ev in-place.
        """
        if not skill_ev.evidence_quality_scores:
            skill_ev.corroboration_score = 0.0
            skill_ev.consistency_score = 0.5  # Neutral - no evidence either way
            skill_ev.evidence_recency_weight = 0.0
            return

        # Corroboration score: max of evidence quality scores
        # (the strongest piece of evidence is most important)
        # Plus a breadth bonus for corroboration across multiple roles
        max_evidence = max(skill_ev.evidence_quality_scores)
        role_breadth_bonus = min(0.15, len(skill_ev.roles_with_evidence) * 0.05)
        evidence_count_bonus = min(0.10, len(skill_ev.evidence_quality_scores) * 0.02)

        raw_corroboration = max_evidence + role_breadth_bonus + evidence_count_bonus

        # Negative context penalty
        if skill_ev.has_negative_context:
            raw_corroboration *= 0.40  # Dramatic penalty for negative context

        skill_ev.corroboration_score = min(1.0, raw_corroboration)
        skill_ev.evidence_recency_weight = sum(skill_ev.evidence_quality_scores) / len(
            skill_ev.evidence_quality_scores
        )

        # Consistency score: does the claimed proficiency match evidence depth?
        # Expert claim + high evidence = consistent
        # Expert claim + mention-level evidence = inconsistent
        proficiency_expectations = {
            "expert": 0.70,
            "advanced": 0.50,
            "intermediate": 0.30,
            "beginner": 0.10,
        }
        expected_corroboration = proficiency_expectations.get(
            skill_ev.claimed_proficiency or "intermediate", 0.30
        )
        actual_corroboration = skill_ev.corroboration_score
        consistency_gap = abs(expected_corroboration - actual_corroboration)
        skill_ev.consistency_score = max(0.0, 1.0 - consistency_gap * 2.0)

    def _score_career_jd_alignment(
        self, role_sentences: List[Tuple[str, int, float]]
    ) -> float:
        """
        Score how well career descriptions align with JD requirements,
        INDEPENDENT of what skills the candidate explicitly claimed.

        This captures the "Tier 5 candidate" effect: a candidate who built
        retrieval systems without listing them as skills should still score well.
        """
        if not role_sentences:
            return 0.0

        total_alignment = 0.0
        total_weight = 0.0

        for sentence, role_idx, recency_weight in role_sentences:
            # Count JD terms present in this sentence
            jd_terms_present = sum(
                1 for term in self._jd_term_set
                if (
                    (len(term.split()) > 1 and term in sentence)
                    or (len(term.split()) == 1 and re.search(r'\b' + re.escape(term) + r'\b', sentence))
                )
            )
            # Weight by term importance and recency
            if jd_terms_present > 0:
                # Multiple JD terms in one sentence is a strong signal
                sentence_alignment = min(1.0, jd_terms_present * 0.3)
                total_alignment += sentence_alignment * recency_weight
                total_weight += recency_weight

        if total_weight == 0:
            return 0.0

        # Normalize by the number of sentences (not total weight)
        # to avoid rewarding verbose candidates over concise ones
        raw_alignment = total_alignment / len(role_sentences)
        return min(1.0, raw_alignment * 4.0)  # Scale up to 1.0 range

    def _compute_aggregate_scores(self, bundle: EvidenceConsistencyBundle) -> None:
        """Compute bundle-level aggregate scores from per-skill evidence."""
        jd_skills = [
            ev for ev in bundle.skill_evidence.values() if ev.is_in_jd
        ]
        if not jd_skills:
            bundle.overall_corroboration_score = bundle.skill_to_career_alignment
            bundle.overall_consistency_score = 0.5
            return

        bundle.overall_corroboration_score = sum(
            ev.corroboration_score for ev in jd_skills
        ) / len(jd_skills)

        bundle.overall_consistency_score = sum(
            ev.consistency_score for ev in jd_skills
        ) / len(jd_skills)

    def _compute_adjustment_factors(self, bundle: EvidenceConsistencyBundle) -> None:
        """
        Compute score_A_adjustment and score_B_adjustment.

        These are ADDITIVE adjustments applied to score_A and score_B
        in the ScoringEngine. Range: -0.15 to +0.15.

        Positive adjustment: career evidence is richer than skills list suggests
        Negative adjustment: skills list makes claims career evidence does not support
        """
        # score_A adjustment: based on career-JD alignment independent of skill claims
        # High career alignment -> reward (up to +0.12)
        # Low alignment but many claimed JD skills -> penalize (down to -0.12)
        career_alignment = bundle.skill_to_career_alignment
        claimed_jd_skill_density = (
            bundle.jd_relevant_skills_count / max(1, bundle.jd_relevant_skills_count + 5)
        )

        if career_alignment >= 0.60:
            # Strong career evidence - boost
            bundle.score_A_adjustment = min(0.12, (career_alignment - 0.60) * 0.40)
        elif bundle.uncorroborated_expert_claims >= 3:
            # Many expert claims with no evidence - penalize
            bundle.score_A_adjustment = max(-0.12, -bundle.uncorroborated_expert_claims * 0.03)
        elif bundle.negative_context_skills:
            # Skills mentioned in negative context (evaluated but didn't use)
            bundle.score_A_adjustment = max(-0.08, -len(bundle.negative_context_skills) * 0.02)
        else:
            bundle.score_A_adjustment = 0.0

        # score_B adjustment: based on corroboration of production-relevant skills
        if bundle.high_quality_evidence_count >= 3:
            # Multiple skills with high-quality production evidence -> boost B
            bundle.score_B_adjustment = min(0.10, bundle.high_quality_evidence_count * 0.02)
        elif bundle.uncorroborated_expert_claims >= 4:
            # Pattern of expert claims without any evidence -> penalize B
            bundle.score_B_adjustment = max(-0.10, -bundle.uncorroborated_expert_claims * 0.02)
        else:
            bundle.score_B_adjustment = 0.0

    def _get_search_terms(self, skill_lower: str) -> List[str]:
        """Get skill name plus known synonyms for searching career descriptions."""
        terms = [skill_lower]
        terms.extend(self._skill_synonyms.get(skill_lower, []))
        return terms

    def _build_skill_synonyms(self) -> Dict[str, List[str]]:
        """
        Synonym mapping for common skill variations.
        Allows finding "sentence transformers" when skill is listed as "sbert",
        and finding "elasticsearch" when skill is listed as "elastic search".
        """
        return {
            "faiss": ["faiss", "facebook ai similarity search"],
            "sentence-transformers": ["sentence-transformers", "sentence transformers", "sbert", "sentence bert"],
            "elasticsearch": ["elasticsearch", "elastic search", "elastic"],
            "opensearch": ["opensearch", "open search"],
            "pinecone": ["pinecone"],
            "weaviate": ["weaviate"],
            "qdrant": ["qdrant"],
            "milvus": ["milvus"],
            "pytorch": ["pytorch", "torch"],
            "tensorflow": ["tensorflow", "tf"],
            "bge": ["bge", "baai/bge"],
            "e5": ["e5", "intfloat/e5"],
            "langchain": ["langchain", "lang chain"],
            "llamaindex": ["llamaindex", "llama index", "llama-index"],
            "ndcg": ["ndcg", "normalized discounted cumulative gain", "dcg"],
            "mrr": ["mrr", "mean reciprocal rank"],
            "lora": ["lora", "loralib", "low-rank adaptation"],
            "qlora": ["qlora", "quantized lora"],
            "peft": ["peft", "parameter efficient"],
            "rag": ["rag", "retrieval augmented generation", "retrieval-augmented"],
            "hnsw": ["hnsw", "hierarchical navigable small world"],
            "ann": ["ann", "approximate nearest neighbor", "approximate nearest-neighbor"],
            "mlflow": ["mlflow", "ml flow"],
            "lightgbm": ["lightgbm", "lgbm", "light gbm"],
            "xgboost": ["xgboost", "xgb"],
        }