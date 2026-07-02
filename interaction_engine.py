"""
interaction_engine.py - Feature interaction detection for multi-signal combinations.

Instead of treating features independently, this module detects meaningful
combinations that signal stronger candidate quality (or disqualifying patterns).

Design principles:
    1. Every interaction must be JD-justified (not arbitrary feature combos)
    2. Interactions provide BONUSES or PENALTIES, not replacements
    3. Interaction scores are bounded [-0.15, +0.15] to prevent domination
    4. Every interaction is explainable for Stage 4 review

Positive interactions (bonuses):
    - Retrieval + Vector DB + Embeddings => shipped a retrieval system
    - Production ML + A/B Testing + Metrics => evaluation framework experience
    - Recommendation Systems + Product Company + Scale => Tier 5 candidate
    - Career evidence + Assessment corroboration => trustworthy skills

Negative interactions (penalties):
    - Research vocabulary + No production deployment => pure researcher risk
    - LLM + Prompt engineering only => LangChain-only risk
    - High skills count + Low corroboration => keyword stuffer
    - High title + Short tenure => title chaser

Controlled by Config.USE_INTERACTION_ENGINE flag.
When False, no interaction adjustments are applied (original behavior).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from feature_extractor import FeatureBundle
from jd_analyzer import JDContext

logger = logging.getLogger("interaction_engine")


@dataclass
class InteractionResult:
    """Result of feature interaction analysis for one candidate."""
    candidate_id: str
    bonus: float = 0.0  # Positive adjustment (max +0.15)
    penalty: float = 0.0  # Negative adjustment (max -0.15)
    triggered_interactions: List[str] = field(default_factory=list)
    interaction_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def net_adjustment(self) -> float:
        """Net score adjustment (bonus - penalty), bounded [-0.15, +0.15]."""
        return max(-0.15, min(0.15, self.bonus - self.penalty))


class InteractionEngine:
    """
    Detects feature interactions and computes score adjustments.

    Called after feature extraction, before scoring. The adjustments are
    ADDITIVE to score_A (core technical relevance), reflecting the
    interaction's impact on the candidate's JD fit.
    """

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context

    def analyze_all(
        self,
        candidates: List[dict],
        feature_bundles: Dict[str, FeatureBundle],
    ) -> Dict[str, InteractionResult]:
        """
        Run interaction analysis on all candidates.

        Returns: {candidate_id: InteractionResult}
        """
        results: Dict[str, InteractionResult] = {}

        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            bundle = feature_bundles.get(cid)
            if bundle is None:
                results[cid] = InteractionResult(candidate_id=cid)
                continue

            try:
                results[cid] = self._analyze_candidate(candidate, bundle)
            except Exception as e:
                logger.warning(f"Interaction analysis error for {cid}: {e}")
                results[cid] = InteractionResult(candidate_id=cid)

            if (i + 1) % 10000 == 0:
                logger.info(f"  Interaction analysis: {i+1:,} candidates processed")

        triggered_count = sum(
            1 for r in results.values() if r.triggered_interactions
        )
        logger.info(
            f"Interaction analysis complete: {triggered_count:,} candidates "
            f"with triggered interactions"
        )
        return results

    def _analyze_candidate(
        self, candidate: dict, bundle: FeatureBundle
    ) -> InteractionResult:
        """Analyze a single candidate for feature interactions."""
        result = InteractionResult(candidate_id=candidate["candidate_id"])

        career = candidate.get("career_history", [])
        all_descriptions = " ".join(
            (r.get("description") or "").lower() for r in career
        )
        all_titles = [(r.get("title") or "").lower() for r in career]
        all_companies = [(r.get("company") or "").lower() for r in career]
        skills = candidate.get("skills", [])
        skill_names = {(s.get("name") or "").lower() for s in skills}
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {}) or {}

        # ── POSITIVE INTERACTIONS ────────────────────────────────────────

        # Interaction 1: Retrieval System Builder
        # JD: "Production experience with embeddings-based retrieval systems"
        # Trigger: retrieval terms + vector DB + embeddings evidence
        retrieval_terms = set(bundle.retrieval_terms_found or [])
        vector_db_terms = {"faiss", "pinecone", "weaviate", "qdrant", "milvus",
                          "opensearch", "elasticsearch", "chroma", "vector search",
                          "vector index", "vector database"}
        embedding_terms = {"sentence-transformers", "sentence transformer", "sbert",
                           "bge", "e5", "openai embeddings", "embedding model",
                           "embedding space", "dense retrieval", "dense vector"}

        has_retrieval = bool(retrieval_terms)
        has_vector_db = bool(vector_db_terms & retrieval_terms) or \
                        bool(vector_db_terms & skill_names) or \
                        any(t in all_descriptions for t in vector_db_terms)
        has_embeddings = bool(embedding_terms & retrieval_terms) or \
                         bool(embedding_terms & skill_names) or \
                         any(t in all_descriptions for t in embedding_terms)

        if has_retrieval and has_vector_db and has_embeddings:
            result.bonus += 0.12
            result.triggered_interactions.append("retrieval_system_builder")
            result.interaction_scores["retrieval_system_builder"] = 0.12

        # Interaction 2: Evaluation Framework Experience
        # JD: "Hands-on experience designing evaluation frameworks for ranking systems"
        # Trigger: production ML + evaluation metrics + A/B testing
        eval_terms = {"ndcg", "mrr", "map", "mean average precision",
                     "a/b test", "a/b testing", "offline evaluation",
                     "online evaluation", "evaluation framework",
                     "ranking evaluation", "precision at k", "recall at k"}
        has_eval = any(t in all_descriptions for t in eval_terms) or \
                   bool(eval_terms & skill_names)
        has_production = bundle.score_B >= 0.55

        if has_production and has_eval:
            result.bonus += 0.10
            result.triggered_interactions.append("evaluation_framework_exp")
            result.interaction_scores["evaluation_framework_exp"] = 0.10

        # Interaction 3: Tier 5 Candidate Discovery
        # JD: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
        # profile, but if their career history shows they built a recommendation
        # system at a product company, they're a fit."
        # Trigger: recommendation/search system + product company + scale
        rec_terms = {"recommendation system", "recommender system",
                     "collaborative filtering", "two-tower", "two tower",
                     "matrix factorization", "content-based filtering",
                     "candidate generation", "search system", "ranking system"}
        has_rec = any(t in all_descriptions for t in rec_terms) or \
                  bool(rec_terms & skill_names)
        has_product_company = bundle.has_product_company
        scale_terms = {"million", "billion", "thousand", "at scale",
                      "production", "user-facing", "real users"}
        has_scale = any(t in all_descriptions for t in scale_terms)

        if has_rec and has_product_company and has_scale:
            # This is the JD's explicitly described Tier 5 candidate
            result.bonus += 0.14
            result.triggered_interactions.append("tier5_candidate_discovery")
            result.interaction_scores["tier5_candidate_discovery"] = 0.14

        # Interaction 4: Assessment-Career Corroboration
        # When assessment scores and career evidence agree, confidence is high
        if bundle.top_assessment_score >= 65 and bundle.effective_relevant_years >= 2.0:
            result.bonus += 0.06
            result.triggered_interactions.append("assessment_career_corroborated")
            result.interaction_scores["assessment_career_corroborated"] = 0.06

        # Interaction 5: Production Deployment at Scale
        # Trigger: production vocabulary + scale indicators + real users
        prod_terms = {"production", "deployed", "shipped", "serving",
                     "real users", "user-facing", "production traffic"}
        has_prod_vocab = any(t in all_descriptions for t in prod_terms)
        has_relevant_years = bundle.effective_relevant_years >= 2.0

        if has_prod_vocab and has_relevant_years and bundle.score_B >= 0.55:
            result.bonus += 0.08
            result.triggered_interactions.append("production_at_scale")
            result.interaction_scores["production_at_scale"] = 0.08

        # ── NEGATIVE INTERACTIONS ─────────────────────────────────────────

        # Interaction 6: Pure Researcher Risk
        # JD: "If you've spent your career in pure research environments without
        # any production deployment — we will not move forward."
        research_indicators = {"research prototype", "proof of concept", "poc",
                              "academic project", "theoretical", "literature review",
                              "survey paper", "research lab", "research center"}
        has_research = sum(1 for t in research_indicators if t in all_descriptions) >= 2
        has_no_production = bundle.score_B < 0.30

        if has_research and has_no_production:
            result.penalty += 0.12
            result.triggered_interactions.append("pure_researcher_risk")
            result.interaction_scores["pure_researcher_risk"] = -0.12

        # Interaction 7: LLM-Only / Framework Enthusiast
        # JD: "If your GitHub is full of LangChain tutorials"
        llm_framework_terms = {"langchain", "llamaindex", "llama-index",
                              "haystack", "flowise", "langflow", "autogpt"}
        has_framework = bool(llm_framework_terms & skill_names) or \
                        any(t in all_descriptions for t in llm_framework_terms)
        ml_depth_terms = {"fine-tuning", "fine tuning", "lora", "qlora", "peft",
                         "training", "gradient", "backpropagation", "loss function",
                         "tokenizer", "attention", "transformer architecture"}
        has_no_ml_depth = not any(t in all_descriptions for t in ml_depth_terms)

        if has_framework and has_no_ml_depth:
            result.penalty += 0.10
            result.triggered_interactions.append("framework_enthusiast_risk")
            result.interaction_scores["framework_enthusiast_risk"] = -0.10

        # Interaction 8: Title Inflation Without Tenure
        # JD: "Title-chasers. If your career trajectory shows you optimizing for
        # 'Senior' -> 'Staff' -> 'Principal' titles by switching companies every 1.5 years"
        if len(career) >= 3:
            short_tenures = sum(
                1 for r in career if 0 < (r.get("duration_months", 24) or 24) < 18
            )
            senior_titles = sum(
                1 for t in all_titles
                if any(kw in t for kw in {"senior", "staff", "principal", "lead", "director"})
            )
            if short_tenures >= 2 and senior_titles >= 2:
                result.penalty += 0.08
                result.triggered_interactions.append("title_inflation_risk")
                result.interaction_scores["title_inflation_risk"] = -0.08

        # Interaction 9: Skills-Career Mismatch
        # When skills list is large but career descriptions don't corroborate
        if len(skills) > 20:
            skill_words_in_career = sum(
                1 for s in skills
                if (s.get("name") or "").lower() in all_descriptions
            )
            corroboration_rate = skill_words_in_career / max(1, len(skills))
            if corroboration_rate < 0.30 and bundle.score_B < 0.40:
                result.penalty += 0.10
                result.triggered_interactions.append("skills_career_mismatch")
                result.interaction_scores["skills_career_mismatch"] = -0.10

        # Clamp bonus and penalty to bounds
        result.bonus = min(0.15, result.bonus)
        result.penalty = min(0.15, result.penalty)

        return result
