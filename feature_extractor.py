"""
feature_extractor.py - Extracts all scoring features from candidate profiles.

This is the computational core of the ranking system. It produces a FeatureBundle
for each candidate containing all sub-scores (A through H) and intermediate signals
needed for reasoning generation.

Performance target: <120 seconds for 100K candidates on CPU.
Implementation approach: vectorized where possible, avoid per-candidate Python loops
on hot paths.

The BM25-style scoring here is the key technical decision. Career descriptions
are tokenized and scored against the JD vocabulary with cluster-weighted term
importance. This is O(N x D x V) where:
    N = number of candidates (100K)
    D = average descriptions length in tokens (~200 per candidate)
    V = vocabulary size (~200 terms)
Which is approximately 4 billion operations - too slow naively.

Optimization: For each candidate, we build a term presence set from the career
descriptions (O(D) per candidate), then score against the JD vocabulary using
set intersection (O(V) per candidate). Total: O(N x (D + V)) ≈ 40M operations.
This runs in well under 60 seconds.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set

from config import Config
from jd_analyzer import JDContext

logger = logging.getLogger("feature_extractor")


@dataclass
class FeatureBundle:
    """
    All computed features and sub-scores for a single candidate.
    Stored for use by scorer.py and reasoning.py.
    """
    candidate_id: str

    # Component scores (0.0 to 1.0 each)
    score_A: float = 0.0  # Core technical relevance
    score_B: float = 0.0  # Production ML depth
    score_C: float = 0.0  # Assessment score match
    score_D: float = 0.0  # Career quality and trajectory
    score_E: float = 0.0  # Behavioral availability
    score_F: float = 0.0  # Location and logistics
    score_G: float = 0.0  # Education signal
    score_H: float = 0.0  # GitHub activity

    # Sub-scores for D (career quality components)
    score_D1: float = 0.0  # Company quality
    score_D2: float = 0.0  # Career progression
    score_D3: float = 0.0  # Tenure stability
    score_D4: float = 0.0  # Domain relevance

    # Raw signals for reasoning generation
    top_assessment_skill: Optional[str] = None
    top_assessment_score: float = 0.0
    production_tier: int = 0  # 1-5 (1=strongest)
    effective_relevant_years: float = 0.0
    career_terms_found: List[str] = field(default_factory=list)
    retrieval_terms_found: List[str] = field(default_factory=list)
    ml_terms_found: List[str] = field(default_factory=list)
    days_since_active: int = 365
    has_product_company: bool = False
    location_country: str = ""
    notice_period_days: int = 60


class FeatureExtractor:
    """
    Extracts feature bundles for all candidates.

    Designed for batch processing - all candidates in one pass.
    """

    # Company quality tiers
    TIER_1_PRODUCT_COMPANIES: Set[str] = {
        "google", "meta", "microsoft", "amazon", "apple", "netflix",
        "openai", "anthropic", "deepmind", "stripe", "airbnb",
        "swiggy", "zomato", "flipkart", "razorpay", "cred",
        "meesho", "ola", "phonepe", "paytm", "zepto",
        "groww", "sharechat", "freshworks", "zoho", "postman",
        "redrob", "myntra", "bigbasket",
    }

    TIER_2_PRODUCT_COMPANIES: Set[str] = {
        "startup", "scale-up", "series a", "series b", "series c",
        "product company",  # generic positive indicator
    }

    CONSULTING_FIRMS: Set[str] = {
        "tcs", "tata consultancy", "infosys", "wipro",
        "accenture", "cognizant", "capgemini", "tech mahindra",
        "hcl", "mphasis", "hexaware", "mindtree",
        "l&t infotech", "niit technologies", "cyient",
        "genpact", "genpact ai",
    }

    # Education degree level scoring
    DEGREE_SCORES = {
        "phd": 0.95, "doctor": 0.95,
        "master": 0.85, "msc": 0.85, "mtech": 0.85, "ms ": 0.85, "m.tech": 0.85,
        "bachelor": 0.75, "btech": 0.75, "be ": 0.75, "b.tech": 0.75,
        "bsc": 0.70, "b.sc": 0.70,
        "diploma": 0.50,
    }

    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology", "it",
        "machine learning", "artificial intelligence", "data science",
        "statistics", "mathematics", "math", "computational",
        "electrical engineering", "electronics", "software",
    }

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context
        self._today = Config.REFERENCE_DATE
        # Pre-compute sorted term list for efficiency
        self._jd_terms_sorted = sorted(
            self.jd.term_weights.keys(),
            key=len,
            reverse=True,  # Longer terms first to match multi-word before single-word
        )
        # Pre-compile a single combined regex for all single-word terms
        # (massively faster than per-term search)
        single_terms = [term for term in self._jd_terms_sorted if len(term.split()) == 1]
        if single_terms:
            joined = '|'.join(re.escape(t) for t in single_terms)
            self._single_term_re = re.compile(r'\b(' + joined + r')\b')
            self._single_word_terms = set(single_terms)
        else:
            self._single_term_re = None
            self._single_word_terms = set()

    def extract_all(self, candidates: List[dict]) -> Dict[str, FeatureBundle]:
        """
        Extract features for all candidates.

        Returns {candidate_id: FeatureBundle}
        """
        bundles: Dict[str, FeatureBundle] = {}

        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            try:
                bundles[cid] = self._extract_candidate(candidate)
            except Exception as e:
                logger.warning(f"Feature extraction error for {cid}: {e}")
                # Return a neutral bundle rather than crashing
                bundles[cid] = FeatureBundle(candidate_id=cid)

            if (i + 1) % 10_000 == 0:
                logger.info(f"  Feature extraction progress: {i + 1:,} candidates processed")

        return bundles

    def _extract_candidate(self, candidate: dict) -> FeatureBundle:
        """Extract all features for a single candidate."""
        bundle = FeatureBundle(candidate_id=candidate["candidate_id"])
        career = candidate.get("career_history", [])
        signals = candidate["redrob_signals"]
        profile = candidate["profile"]
        skills = candidate.get("skills", [])
        education = candidate.get("education", [])

        # ── Component A: Core Technical Relevance ───────────────────────────
        bundle.score_A, bundle.career_terms_found, bundle.retrieval_terms_found, \
            bundle.ml_terms_found = self._score_A(career)

        # ── Component B: Production ML Depth ────────────────────────────────
        bundle.score_B, bundle.production_tier, \
            bundle.effective_relevant_years = self._score_B(career, skills)

        # ── Component C: Assessment Score Match ─────────────────────────────
        bundle.score_C, bundle.top_assessment_skill, \
            bundle.top_assessment_score = self._score_C(signals)

        # ── Component D: Career Quality & Trajectory ─────────────────────────
        bundle.score_D, bundle.score_D1, bundle.score_D2, \
            bundle.score_D3, bundle.score_D4, \
            bundle.has_product_company = self._score_D(career, profile)

        # ── Component E: Behavioral Availability ─────────────────────────────
        bundle.score_E, bundle.days_since_active = self._score_E(signals)

        # ── Component F: Location & Logistics ───────────────────────────────
        bundle.score_F, bundle.location_country, \
            bundle.notice_period_days = self._score_F(profile, signals)

        # ── Component G: Education Signal ────────────────────────────────────
        bundle.score_G = self._score_G(education)

        # ── Component H: GitHub Activity ─────────────────────────────────────
        bundle.score_H = self._score_H(signals, career)

        return bundle

    # ─────────────────────────────────────────────────────────────────────────
    # Component A: Core Technical Relevance
    # ─────────────────────────────────────────────────────────────────────────

    def _score_A(
        self, career: List[dict]
    ):
        """
        Score semantic relevance of career descriptions to JD requirements.

        Approach: Weighted term presence scoring.
        - Build a bag-of-terms from all career descriptions, weighted by recency
        - Score each term's presence against the JD vocabulary
        - Weight term score by cluster importance and recency of the role it appears in

        Recency weighting: Most recent role counts 1.0x, each prior role decays by 0.15x.
        This reflects that recent experience is more indicative of current capability.

        Returns: (score, all_terms_found, retrieval_terms, ml_terms)
        """
        if not career:
            return 0.0, [], [], []

        # Sort career by recency (most recent first)
        sorted_career = sorted(
            career,
            key=lambda r: r.get("start_date") or "2000-01-01",
            reverse=True,
        )

        all_terms_found: List[str] = []
        retrieval_terms: List[str] = []
        ml_terms: List[str] = []

        total_weighted_score = 0.0
        max_possible_score = 0.0

        for role_idx, role in enumerate(sorted_career):
            # Recency decay: most recent role = 1.0, each prior = 0.15 less
            recency_weight = max(0.2, 1.0 - role_idx * 0.15)

            description = (role.get("description") or "").lower()
            if not description:
                continue

            # Build term presence set from this description
            terms_in_this_role = self._find_jd_terms_in_text(description)
            all_terms_found.extend(terms_in_this_role)

            # Categorize and score each found term
            role_score = 0.0
            for term in terms_in_this_role:
                term_weight = self.jd.term_weights.get(term, 0.3)
                role_score += term_weight

                # Categorize for reasoning
                if term in self.jd.retrieval_search_terms:
                    if term not in retrieval_terms:
                        retrieval_terms.append(term)
                elif term in self.jd.ml_production_terms:
                    if term not in ml_terms:
                        ml_terms.append(term)

            # Check for anti-production language (reduces score)
            anti_production_count = sum(
                1 for ap_term in self.jd.anti_production_terms
                if ap_term in description
            )
            if anti_production_count > 0:
                role_score *= max(0.5, 1.0 - anti_production_count * 0.15)

            total_weighted_score += role_score * recency_weight

        # Normalize: a perfect candidate would match ~15 high-weight terms
        # Cap at a reasonable ceiling to avoid score > 1.0
        normalization_ceiling = 12.0
        score_A = min(1.0, total_weighted_score / normalization_ceiling)

        # Deduplicate term lists
        all_terms_found = list(dict.fromkeys(all_terms_found))  # Preserve order, deduplicate
        retrieval_terms = list(dict.fromkeys(retrieval_terms))
        ml_terms = list(dict.fromkeys(ml_terms))

        return score_A, all_terms_found, retrieval_terms, ml_terms

    def _find_jd_terms_in_text(self, text: str) -> List[str]:
        """
        Find all JD vocabulary terms present in a text.

        Uses substring matching with a single combined regex for word-boundary
        matching of single-word terms. Much faster than per-term regex search.
        """
        found_terms = []
        # Multi-word terms first (sorted by length desc, so longer matches first)
        for term in self._jd_terms_sorted:
            if len(term.split()) > 1:
                if term in text:
                    found_terms.append(term)
        # Single-word terms: single combined regex pass
        if self._single_term_re:
            for match in self._single_term_re.finditer(text):
                t = match.group(1)
                if t in self._single_word_terms and t not in found_terms:
                    found_terms.append(t)
        return found_terms

    # ─────────────────────────────────────────────────────────────────────────
    # Component B: Production ML Depth Score
    # ─────────────────────────────────────────────────────────────────────────

    def _score_B(
        self, career: List[dict], skills: List[dict]
    ):
        """
        Classify candidate into production ML depth tiers (1-5) and score.

        Tier 1 = Strongest (0.75-1.0)
        Tier 5 = Weakest (0.0-0.1)

        The tier is determined by analyzing career descriptions for:
        - ML/AI role titles (necessary but not sufficient)
        - Production deployment language
        - Operational scale indicators
        - Evaluation framework references
        - Sustained multi-role evidence

        Returns: (score, production_tier, effective_relevant_years)
        """
        all_descriptions = " ".join(
            (r.get("description") or "").lower() for r in career
        )
        all_titles = [
            (r.get("title") or "").lower() for r in career
        ]

        # ML title vocabulary
        ml_title_keywords = {
            "machine learning", "ml engineer", "ai engineer", "data scientist",
            "nlp engineer", "search engineer", "recommendation", "retrieval",
            "ranking engineer", "applied scientist", "research scientist",
            "applied ml", "applied ai",
        }

        # Count ML/AI-titled roles
        ml_role_count = sum(
            1 for title in all_titles
            if any(kw in title for kw in ml_title_keywords)
        )

        # Production vocabulary strength
        strong_production_terms = {
            "production", "deployed to production", "serving", "real users",
            "user-facing", "a/b test", "latency", "sla", "qps",
            "million", "billion", "at scale",
        }
        evaluation_terms = {
            "ndcg", "mrr", "mean reciprocal rank", "a/b test",
            "offline evaluation", "online evaluation",
            "evaluation framework", "ranking metrics",
        }
        operational_terms = {
            "monitoring", "drift", "incident", "alert", "on-call",
            "index refresh", "retrieval regression", "model refresh",
        }

        production_score = sum(
            1 for term in strong_production_terms if term in all_descriptions
        )
        evaluation_score = sum(
            1 for term in evaluation_terms if term in all_descriptions
        )
        operational_score = sum(
            1 for term in operational_terms if term in all_descriptions
        )

        # Compute effective relevant years (career months in ML/AI roles)
        effective_months = 0
        for role in career:
            title_lower = (role.get("title") or "").lower()
            if any(kw in title_lower for kw in ml_title_keywords):
                effective_months += role.get("duration_months", 0) or 0
        effective_relevant_years = effective_months / 12.0

        # Tier classification
        if (
            ml_role_count >= 2
            and production_score >= 3
            and (evaluation_score >= 1 or operational_score >= 1)
            and effective_relevant_years >= 2.0
        ):
            # Tier 1: Strong production ML, directly relevant
            tier = 1
            base_score = 0.75 + min(0.25, effective_relevant_years * 0.03)

        elif (
            ml_role_count >= 1
            and production_score >= 2
            and effective_relevant_years >= 1.0
        ):
            # Tier 2: Solid ML engineering, some production
            tier = 2
            base_score = 0.55 + min(0.20, effective_relevant_years * 0.04)

        elif ml_role_count >= 1 and effective_relevant_years >= 0.5:
            # Tier 3: Genuine ML experience, research or prototype level
            tier = 3
            base_score = 0.30 + min(0.25, production_score * 0.08)

        elif self._has_adjacent_ml_exposure(all_descriptions, all_titles):
            # Tier 4: Adjacent exposure (data pipelines, analytics, BI)
            tier = 4
            base_score = 0.10 + min(0.20, production_score * 0.05)

        else:
            # Tier 5: No ML production evidence
            tier = 5
            base_score = 0.02

        score_B = min(1.0, base_score)
        return score_B, tier, effective_relevant_years

    def _has_adjacent_ml_exposure(
        self, all_descriptions: str, all_titles: List[str]
    ) -> bool:
        """Check for roles adjacent to ML without being ML themselves."""
        adjacent_title_keywords = {
            "data engineer", "data analyst", "business intelligence",
            "bi developer", "analytics", "data platform",
        }
        adjacent_description_keywords = {
            "data pipeline", "etl", "data warehouse", "analytics dashboard",
        }
        has_adjacent_title = any(
            kw in title for title in all_titles
            for kw in adjacent_title_keywords
        )
        has_adjacent_desc = any(
            kw in all_descriptions for kw in adjacent_description_keywords
        )
        return has_adjacent_title or has_adjacent_desc

    # ─────────────────────────────────────────────────────────────────────────
    # Component C: Assessment Score Match
    # ─────────────────────────────────────────────────────────────────────────

    def _score_C(self, signals: dict):
        """
        Score the candidate's platform-administered assessment scores.

        These are the highest-trust quantitative signals in the dataset.
        Assessment relevance is determined by the JD requirements.
        """
        assessment_scores = signals.get("skill_assessment_scores") or {}
        if not assessment_scores:
            return 0.5, None, 0.0  # Neutral prior for no assessments

        # Assessment relevance weights matching the JD
        VERY_RELEVANT = {
            "faiss", "pinecone", "weaviate", "embeddings", "retrieval systems",
            "retrieval", "recommendation systems", "nlp", "natural language processing",
            "llm fine-tuning", "rag", "mlflow", "learning to rank", "dense retrieval",
            "vector databases", "vector search", "sentence transformers",
        }
        MODERATELY_RELEVANT = {
            "machine learning", "data science", "pytorch", "tensorflow",
            "python", "kubernetes", "elasticsearch", "deep learning",
        }
        WEAKLY_RELEVANT = {
            "data pipelines", "sql", "spark", "aws", "gcp", "azure",
            "data engineering",
        }

        weighted_score_sum = 0.0
        total_weight = 0.0
        top_skill = None
        top_score = 0.0

        for skill_name, raw_score in assessment_scores.items():
            skill_lower = skill_name.lower()
            raw_normalized = (raw_score or 0) / 100.0

            # Determine relevance weight
            if any(rv in skill_lower for rv in VERY_RELEVANT):
                weight = 1.0
            elif any(mr in skill_lower for mr in MODERATELY_RELEVANT):
                weight = 0.6
            elif any(wr in skill_lower for wr in WEAKLY_RELEVANT):
                weight = 0.2
            else:
                weight = 0.0  # Irrelevant assessments ignored

            if weight > 0:
                weighted_score_sum += raw_normalized * weight
                total_weight += weight
                if raw_score > top_score:
                    top_score = raw_score
                    top_skill = skill_name

        if total_weight == 0:
            return 0.5, None, 0.0  # Neutral - no relevant assessments

        score_C = weighted_score_sum / total_weight
        return score_C, top_skill, top_score

    # ─────────────────────────────────────────────────────────────────────────
    # Component D: Career Quality & Trajectory
    # ─────────────────────────────────────────────────────────────────────────

    def _score_D(
        self, career: List[dict], profile: dict
    ):
        """
        Multi-dimensional career quality analysis.

        D1: Company quality (5% total)
        D2: Career progression (4% total)
        D3: Tenure stability (2% total)
        D4: Domain relevance of career arc (2% total)
        """
        if not career:
            return 0.3, 0.3, 0.3, 0.3, 0.3, False

        # D1: Company quality
        score_D1, has_product_company = self._score_D1_company_quality(career)

        # D2: Career progression
        score_D2 = self._score_D2_progression(career)

        # D3: Tenure stability
        score_D3 = self._score_D3_tenure(career)

        # D4: Domain relevance arc
        score_D4 = self._score_D4_domain(career)

        # Weighted combination (D1 carries most weight at the total level)
        # Weights here are relative within component D
        score_D = (
            score_D1 * 0.38   # Maps to 5%/13% of total
            + score_D2 * 0.31  # Maps to 4%/13%
            + score_D3 * 0.15  # Maps to 2%/13%
            + score_D4 * 0.15  # Maps to 2%/13%
        )

        return score_D, score_D1, score_D2, score_D3, score_D4, has_product_company

    def _score_D1_company_quality(self, career: List[dict]):
        """Score company quality across career history."""
        has_product_company = False
        role_scores = []

        for role in career:
            company_lower = (role.get("company") or "").lower()
            company_size = (role.get("company_size") or "").lower()

            # Check for tier-1 product companies
            if any(pc in company_lower for pc in self.TIER_1_PRODUCT_COMPANIES):
                role_scores.append(1.0)
                has_product_company = True
            # Check for consulting firms
            elif any(cf in company_lower for cf in self.CONSULTING_FIRMS):
                role_scores.append(0.35)
            # Mid-size product companies (inferred from size and not consulting)
            elif company_size in ("201-500", "501-1000", "1001-5000"):
                role_scores.append(0.65)
                # Likely a product company if not consulting
                has_product_company = True
            # Early-stage startups
            elif company_size in ("1-10", "11-50"):
                role_scores.append(0.60)  # Could be great, uncertainty
                has_product_company = True  # Most small companies are product
            # Very large unknown companies
            elif company_size in ("5001-10000", "10001+"):
                role_scores.append(0.55)
            else:
                role_scores.append(0.50)  # Unknown - neutral

        if not role_scores:
            return 0.5, False

        # Weight by recency
        weighted = sum(
            score * max(0.2, 1.0 - idx * 0.15)
            for idx, score in enumerate(role_scores)
        )
        total_weight = sum(
            max(0.2, 1.0 - idx * 0.15) for idx in range(len(role_scores))
        )
        return weighted / total_weight, has_product_company

    def _score_D2_progression(self, career: List[dict]) -> float:
        """Score career progression pattern."""
        if len(career) < 2:
            return 0.6  # Single role - can't assess progression, neutral

        title_level_map = {
            "intern": 0, "trainee": 0,
            "junior": 1, "associate": 1,
            "engineer": 2, "analyst": 2, "developer": 2,
            "senior": 3, "specialist": 3,
            "lead": 4, "staff": 4, "tech lead": 4,
            "principal": 5, "architect": 5,
            "manager": 4, "director": 6,
            "head": 6, "vp": 7,
        }

        sorted_career = sorted(
            career,
            key=lambda r: r.get("start_date") or "2000-01-01",
        )

        levels = []
        for role in sorted_career:
            title_lower = (role.get("title") or "").lower()
            level = max(
                (v for k, v in title_level_map.items() if k in title_lower),
                default=2,
            )
            levels.append(level)

        if not levels:
            return 0.5

        # Check if career is trending upward
        upward_moves = sum(
            1 for i in range(1, len(levels)) if levels[i] > levels[i - 1]
        )
        lateral_moves = sum(
            1 for i in range(1, len(levels)) if levels[i] == levels[i - 1]
        )
        downward_moves = sum(
            1 for i in range(1, len(levels)) if levels[i] < levels[i - 1]
        )

        total_moves = len(levels) - 1
        if total_moves == 0:
            return 0.6

        progression_score = (
            (upward_moves * 1.0 + lateral_moves * 0.5 + downward_moves * 0.1)
            / total_moves
        )
        return min(1.0, progression_score)

    def _score_D3_tenure(self, career: List[dict]) -> float:
        """Score tenure stability."""
        if not career:
            return 0.5

        durations = [
            max(0, r.get("duration_months", 0) or 0) for r in career
        ]
        if not durations:
            return 0.5

        avg_tenure_months = sum(durations) / len(durations)
        short_tenures = sum(1 for d in durations if 0 < d < 8)
        very_short_tenures = sum(1 for d in durations if 0 < d < 4)

        # Base score from average tenure
        if avg_tenure_months >= 30:
            base = 0.90
        elif avg_tenure_months >= 24:
            base = 0.80
        elif avg_tenure_months >= 18:
            base = 0.70
        elif avg_tenure_months >= 12:
            base = 0.60
        else:
            base = 0.40

        # Penalize for instability patterns
        penalty = short_tenures * 0.05 + very_short_tenures * 0.08
        return max(0.1, min(1.0, base - penalty))

    def _score_D4_domain(self, career: List[dict]) -> float:
        """Score the domain relevance arc of the career."""
        ml_keywords = {
            "machine learning", "ai", "artificial intelligence", "data science",
            "nlp", "recommendation", "search", "retrieval", "ranking",
            "deep learning", "neural", "ml engineer", "ai engineer",
        }
        tech_keywords = {
            "software", "backend", "data", "engineer", "developer",
            "architect", "platform", "infrastructure",
        }

        # Compute ML-in-titles ratio
        all_titles_lower = [
            (r.get("title") or "").lower() for r in career
        ]
        ml_title_count = sum(
            1 for title in all_titles_lower
            if any(kw in title for kw in ml_keywords)
        )
        tech_title_count = sum(
            1 for title in all_titles_lower
            if any(kw in title for kw in tech_keywords)
        )

        total_roles = len(career)
        if total_roles == 0:
            return 0.3

        ml_ratio = ml_title_count / total_roles
        tech_ratio = tech_title_count / total_roles

        if ml_ratio >= 0.5:
            return min(1.0, 0.7 + ml_ratio * 0.3)
        elif tech_ratio >= 0.6:
            return 0.60  # Strong tech background, could transition
        elif ml_ratio >= 0.2:
            return 0.50  # Mixed career
        else:
            return 0.25  # Primarily non-tech

    # ─────────────────────────────────────────────────────────────────────────
    # Component E: Behavioral Availability
    # ─────────────────────────────────────────────────────────────────────────

    def _score_E(self, signals: dict):
        """
        Score behavioral availability and platform engagement.

        This measures whether the candidate is actually reachable and
        whether they are actively looking. Used as a multiplier modifier
        to final score, never to drop Tier 1 below Tier 4.
        """
        today = self._today

        # Parse last active date
        try:
            last_active = datetime.strptime(
                signals["last_active_date"][:10], "%Y-%m-%d"
            ).date()
            days_since_active = (today - last_active).days
        except (ValueError, TypeError, KeyError):
            days_since_active = 365

        # Sub-scores with weights from the architecture spec
        # Last active (30%)
        if days_since_active <= 30:
            s_active = 1.0
        elif days_since_active <= 90:
            s_active = 0.7
        elif days_since_active <= 180:
            s_active = 0.4
        else:
            s_active = 0.2

        # Open to work (15%)
        s_open = 1.0 if signals.get("open_to_work_flag", False) else 0.4

        # Recruiter response rate (20%)
        rr = signals.get("recruiter_response_rate", 0.5) or 0.5
        if rr >= 0.7:
            s_response = 1.0
        elif rr >= 0.4:
            s_response = 0.75
        elif rr >= 0.2:
            s_response = 0.5
        else:
            s_response = 0.25

        # Response time (10%)
        rt = signals.get("avg_response_time_hours", 48.0) or 48.0
        if rt < 24:
            s_rt = 1.0
        elif rt < 72:
            s_rt = 0.8
        elif rt < 168:
            s_rt = 0.6
        else:
            s_rt = 0.3

        # Interview completion (15%)
        icr = signals.get("interview_completion_rate", 0.7) or 0.7
        if icr >= 0.8:
            s_icr = 1.0
        elif icr >= 0.6:
            s_icr = 0.8
        elif icr >= 0.4:
            s_icr = 0.6
        else:
            s_icr = 0.3

        # Applications submitted (5%)
        apps = signals.get("applications_submitted_30d", 0) or 0
        if apps >= 5:
            s_apps = 1.0
        elif apps >= 1:
            s_apps = 0.7
        else:
            s_apps = 0.4

        # Recruiter demand signal (5%) — saved_by_recruiters_30d reflects
        # genuine human recruiter judgment: recruiters actively bookmarking a
        # candidate for future roles is a high-trust demand signal.
        # PDF: recruiter signals = high trust, "represents human judgment from
        # a domain expert." Capped at 5% so it cannot override technical quality.
        saved = signals.get("saved_by_recruiters_30d", 0) or 0
        if saved >= 15:
            s_saved = 1.0
        elif saved >= 8:
            s_saved = 0.8
        elif saved >= 3:
            s_saved = 0.6
        elif saved >= 1:
            s_saved = 0.4
        else:
            s_saved = 0.2

        score_E = (
            s_active * 0.30
            + s_open * 0.15
            + s_response * 0.20
            + s_rt * 0.10
            + s_icr * 0.15
            + s_apps * 0.05
            + s_saved * 0.05
        )

        return score_E, days_since_active

    # ─────────────────────────────────────────────────────────────────────────
    # Component F: Location & Logistics
    # ─────────────────────────────────────────────────────────────────────────

    def _score_F(self, profile: dict, signals: dict):
        """
        Score location, notice period, and work mode fit.

        Hard constraint: Role does not sponsor visas.
        """
        country = (profile.get("country") or "").strip()
        location_lower = (profile.get("location") or "").lower()
        willing_to_relocate = signals.get("willing_to_relocate", False)
        notice_period = signals.get("notice_period_days", 60) or 60
        work_mode = (signals.get("preferred_work_mode") or "flexible").lower()

        # Location score
        is_india = (
            "india" in country.lower()
            or any(loc in location_lower for loc in self.jd.target_locations)
        )
        in_target_city = any(
            city in location_lower
            for city in {"noida", "pune", "delhi", "gurgaon", "gurugram",
                         "hyderabad", "bengaluru", "bangalore", "mumbai"}
        )

        if is_india and (in_target_city or willing_to_relocate):
            s_location = 1.0
        elif is_india and not willing_to_relocate:
            s_location = 0.7
        elif not is_india and willing_to_relocate:
            s_location = 0.3  # No visa sponsorship
        else:
            s_location = 0.1

        # Notice period score — logistics signal only, not quality.
        # PDF: "A senior candidate with a 90-day notice period is actually a mild
        # positive quality signal." We must not penalize long notice heavily.
        # Score reflects pure scheduling friction: under-30 is ideal (immediate
        # availability), 30-90 is acceptable, 90+ is a logistical note only.
        if notice_period <= 30:
            s_notice = 1.0
        elif notice_period <= 60:
            s_notice = 0.92
        elif notice_period <= 90:
            s_notice = 0.85
        elif notice_period <= 150:
            s_notice = 0.75
        else:
            s_notice = 0.65

        # Work mode score
        if work_mode in ("hybrid", "flexible"):
            s_work = 1.0
        elif work_mode == "onsite":
            s_work = 0.80
        elif work_mode == "remote":
            s_work = 0.60
        else:
            s_work = 0.80  # Unknown - assume compatible

        score_F = (s_location + s_notice + s_work) / 3.0
        return score_F, country, notice_period

    # ─────────────────────────────────────────────────────────────────────────
    # Component G: Education Signal
    # ─────────────────────────────────────────────────────────────────────────

    def _score_G(self, education: List[dict]) -> float:
        """
        Score education as a weak positive modifier.

        For senior engineering roles, education is a 3% weight signal.
        Never let it override career evidence.
        """
        if not education:
            return 0.5  # Neutral - no education data

        best_score = 0.0
        for edu in education:
            degree_lower = (edu.get("degree") or "").lower()
            field_lower = (edu.get("field_of_study") or "").lower()
            tier = (edu.get("tier") or "unknown").lower()

            # Degree level score
            degree_score = 0.5  # Default for unrecognized degrees
            for degree_key, score in self.DEGREE_SCORES.items():
                if degree_key in degree_lower:
                    degree_score = score
                    break

            # Field relevance modifier
            field_relevant = any(
                rf in field_lower for rf in self.RELEVANT_FIELDS
            )
            if field_relevant:
                degree_score = min(1.0, degree_score * 1.05)

            # Institution tier modifier
            tier_modifiers = {
                "tier_1": 1.10, "tier_2": 1.0,
                "tier_3": 0.95, "tier_4": 0.90, "unknown": 0.90,
            }
            tier_modifier = tier_modifiers.get(tier, 0.90)
            degree_score = min(1.0, degree_score * tier_modifier)

            if degree_score > best_score:
                best_score = degree_score

        return best_score

    # ─────────────────────────────────────────────────────────────────────────
    # Component H: GitHub Activity
    # ─────────────────────────────────────────────────────────────────────────

    def _score_H(self, signals: dict, career: List[dict]) -> float:
        """
        Score GitHub activity.

        Special case: Large company employees often cannot open-source work.
        For candidates from companies with 10001+ employees, use neutral 0.5.
        """
        github_score = signals.get("github_activity_score", -1)

        # Check if from a large company (cannot share code publicly)
        from_large_company = any(
            (r.get("company_size") or "") == "10001+" for r in career
            if r.get("is_current", False)
        )

        if github_score == -1:
            return 0.5  # No GitHub linked - neutral, not penalized

        if from_large_company:
            return 0.5  # Large company engineer - don't penalize lack of public activity

        if github_score <= 20:
            return 0.30
        elif github_score <= 50:
            return 0.60
        elif github_score <= 75:
            return 0.80
        else:
            return 1.00