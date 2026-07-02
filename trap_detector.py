"""
trap_detector.py - Detects recruiting trap patterns specific to this JD.

Unlike honeypots (data integrity violations), traps are real candidates who are
bad fits for this specific role. They receive graduated multiplier penalties,
not near-zero scores.

The trap detectors are designed around the JD's explicit disqualifiers and the
known failure patterns described in the competition documentation.

Trust calibration notes:
    Trap detection is MEDIUM-HIGH CONFIDENCE for structural traps
    (consulting_only, pure_researcher) and MEDIUM CONFIDENCE for behavioral
    traps (langchain_only, recent_hype_pivot) which require more inference.
"""

import logging
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Set

from config import Config

logger = logging.getLogger("trap_detector")


class TrapDetector:
    """
    Detects the following trap patterns:
        - consulting_only: Entire career at tier-1 IT services firms
        - langchain_only: AI experience only through framework wrappers, no depth
        - pure_researcher: Career entirely in academia, no production deployment
        - behavioral_dead: Not reachable through the platform
        - keyword_stuffer: Skills list far exceeds career evidence
        - recent_hype_pivot: AI experience only after 2022 hype cycle, thin depth
        - title_chaser: Senior->Staff->Principal by switching every 18 months
        - domain_mismatch: CV/speech/robotics primary, no NLP/IR exposure
    """

    # Consulting/services firm names (lowercase for matching)
    CONSULTING_FIRMS: Set[str] = {
        "tcs", "tata consultancy", "infosys", "wipro",
        "accenture", "cognizant", "capgemini", "tech mahindra",
        "hcl technologies", "hcl tech", "hcl", "mphasis",
        "hexaware", "mindtree", "l&t infotech", "niit technologies",
        "cyient", "zensar", "persistent systems", "ltimindtree",
        "birlasoft", "coforge", "kpit technologies", "sasken",
        "cts",  # Cognizant Technology Solutions abbreviation
        "genpact", "genpact ai",
    }

    # Academic/research institution indicators
    RESEARCH_INDICATORS: Set[str] = {
        "iit", "iim", "iisc", "iiser", "nit", "bits",
        "university", "college", "institute of technology",
        "research lab", "research center", "research institute",
        "academia", "national lab", "mit", "stanford", "cmu",
        "microsoft research", "google research", "deepmind",
        "openai", "anthropic", "fair", "ai research",
    }

    # Production environment keywords - presence in descriptions signals real deployment
    PRODUCTION_KEYWORDS: Set[str] = {
        "production", "deployed", "shipped", "serving",
        "real users", "user-facing", "a/b test", "latency",
        "million", "billion", "scale", "sla", "qps",
        "monitoring", "alerting", "on-call", "incident",
        "kubernetes", "docker", "ci/cd", "pipeline",
    }

    # LangChain-framework vocabulary (surface-level AI work)
    LANGCHAIN_FRAMEWORKS: Set[str] = {
        "langchain", "llamaindex", "llama-index", "haystack",
        "flowise", "langflow", "autogpt", "agentgpt",
    }

    # Deep ML vocabulary that distinguishes genuine engineers from framework users
    ML_DEPTH_VOCABULARY: Set[str] = {
        "faiss", "hnsw", "ann", "approximate nearest neighbor",
        "embedding model", "fine-tuning", "fine tuning",
        "vector index", "dense retrieval", "bi-encoder", "cross-encoder",
        "sentence-transformers", "training", "evaluation framework",
        "ndcg", "mrr", "a/b test", "learning to rank",
        "gradient", "backpropagation", "loss function",
        "tokenizer", "attention", "transformer architecture",
        "model serving", "inference", "latency optimization",
        "quantization", "onnx", "pytorch", "tensorflow",
        "recommendation system", "collaborative filtering",
    }

    # Keywords that suggest primarily CV/speech/robotics without NLP/IR
    NON_NLP_DOMAIN_KEYWORDS: Set[str] = {
        "object detection", "image classification", "yolo",
        "opencv", "computer vision", "speech recognition",
        "speech synthesis", "asr", "tts",
        "robotics", "ros", "robot operating system",
        "lidar", "point cloud", "slam",
    }

    # Senior title patterns for title-chaser detection
    SENIOR_TITLE_PATTERNS = [
        r"senior", r"staff", r"principal", r"lead",
        r"director", r"architect", r"head of",
    ]

    def detect_all(self, candidates: List[dict]) -> Dict[str, Dict[str, bool]]:
        """
        Run all trap detectors on all candidates.

        Args:
            candidates: List of candidate dicts

        Returns:
            {candidate_id: {trap_name: bool}}
        """
        results: Dict[str, Dict[str, bool]] = {}
        today = Config.REFERENCE_DATE

        for candidate in candidates:
            cid = candidate["candidate_id"]
            try:
                results[cid] = self._detect_candidate(candidate, today)
            except Exception as e:
                logger.warning(
                    f"Trap detection error for {cid}: {e} - defaulting to no traps"
                )
                results[cid] = {}

        return results

    def _detect_candidate(
        self, candidate: dict, today: date
    ) -> Dict[str, bool]:
        """Run all trap checks on a single candidate."""
        career = candidate.get("career_history", [])
        signals = candidate["redrob_signals"]
        profile = candidate["profile"]
        skills = candidate.get("skills", [])

        # Pre-compute text aggregates once
        all_descriptions = " ".join(
            (r.get("description") or "").lower() for r in career
        )
        all_skill_names = {
            (s.get("name") or "").lower() for s in skills
        }
        all_companies = [
            (r.get("company") or "").lower() for r in career
        ]
        all_titles = [
            (r.get("title") or "").lower() for r in career
        ]

        return {
            "consulting_only": self._is_consulting_only(all_companies),
            "langchain_only": self._is_langchain_only(
                all_skill_names, all_descriptions
            ),
            "pure_researcher": self._is_pure_researcher(
                all_companies, all_descriptions
            ),
            "behavioral_dead": self._is_behavioral_dead(signals, today),
            "keyword_stuffer": self._is_keyword_stuffer(
                skills, all_descriptions
            ),
            "recent_hype_pivot": self._is_recent_hype_pivot(
                career, all_descriptions
            ),
            "title_chaser": self._is_title_chaser(career),
            "domain_mismatch": self._is_domain_mismatch(
                all_skill_names, all_descriptions, all_titles,
                current_title=(profile.get("current_title") or ""),
                assessment_scores=(
                    signals.get("skill_assessment_scores") or {}
                ),
            ),
            "certification_padder": self._is_certification_padder(skills),
            "experience_gap": self._is_experience_gap(
                career,
                float(profile.get("years_of_experience", 0) or 0),
            ),
        }

    def _is_consulting_only(self, company_names_lower: List[str]) -> bool:
        """
        Returns True if the ENTIRE career is at tier-1 consulting/services firms.

        Critical: This must return False if there is even ONE non-consulting role.
        A candidate who spent 3 years at Infosys and then joined Swiggy is not
        consulting-only - the Swiggy experience is meaningful product experience.
        """
        if not company_names_lower:
            return False
        for company in company_names_lower:
            # If any company is NOT in the consulting list, they are not consulting-only
            is_consulting = any(
                firm in company for firm in self.CONSULTING_FIRMS
            )
            if not is_consulting:
                return False
        return True

    def _is_langchain_only(
        self, skill_names: Set[str], all_descriptions: str
    ) -> bool:
        """
        Returns True if the candidate's AI work is entirely through high-level
        framework wrappers with no evidence of underlying ML depth.

        The combination of (has framework skills) AND (no depth vocabulary in
        career descriptions) is the key signal. Skills alone are not enough -
        a genuine ML engineer might use LangChain as a tool.
        """
        has_langchain_framework = any(
            fw in skill_names for fw in self.LANGCHAIN_FRAMEWORKS
        )
        if not has_langchain_framework:
            return False

        # Check if career descriptions show any ML depth
        has_ml_depth = any(
            term in all_descriptions for term in self.ML_DEPTH_VOCABULARY
        )
        return not has_ml_depth

    def _is_pure_researcher(
        self, company_names: List[str], all_descriptions: str
    ) -> bool:
        """
        Returns True if the candidate has only worked in academic/research
        institutions AND the career descriptions show no production deployment evidence.

        Two conditions must both be true:
        1. All companies are research/academic institutions
        2. Career descriptions lack production deployment vocabulary
        """
        if not company_names:
            return False

        # Check if all companies are research/academic
        all_research = all(
            any(ri in company for ri in self.RESEARCH_INDICATORS)
            for company in company_names
        )
        if not all_research:
            return False

        # Check for any production evidence in descriptions
        has_production_evidence = any(
            kw in all_descriptions for kw in self.PRODUCTION_KEYWORDS
        )
        return not has_production_evidence

    def _is_behavioral_dead(self, signals: dict, today: date) -> bool:
        """
        Returns True if the candidate is likely unreachable through the platform.

        Requires ALL three conditions:
        - Last active more than 6 months ago (stale account)
        - Very low recruiter response rate (doesn't respond)
        - Not marked open to work (not actively looking)

        Any one condition alone is insufficient. Together they indicate a
        fundamentally unreachable candidate.
        """
        try:
            last_active = datetime.strptime(
                signals["last_active_date"][:10], "%Y-%m-%d"
            ).date()
            days_since_active = (today - last_active).days
        except (ValueError, TypeError):
            days_since_active = 365  # Treat unparseable as old

        return (
            days_since_active > 180
            and signals.get("recruiter_response_rate", 0.5) < 0.15
            and not signals.get("open_to_work_flag", False)
        )

    def _is_keyword_stuffer(
        self, skills: List[dict], all_descriptions: str
    ) -> bool:
        """
        Returns True if the skills list is dramatically larger than what the
        career descriptions can plausibly support.

        Improved (B5): Multi-signal detection requiring corroborating evidence.
        Single-signal flagging risks penalizing genuine polymaths.

        A candidate is flagged as a keyword stuffer only when ALL of:
          1. Skills list is large (>25 skills)
          2. Career corroboration rate is low (<40%)
          3. No production deployment evidence in descriptions
             (production vocabulary indicates real work, not stuffing)

        This prevents false positives on genuine polymaths who have many
        skills AND production evidence AND career corroboration.
        """
        from config import Config

        if len(skills) < 20:
            return False  # Normal skills list - not a stuffer

        # Count skills that have some corroboration in career descriptions
        corroborated_skills = 0
        for skill in skills:
            skill_name = (skill.get("name") or "").lower()
            # Check if any word from the skill name appears in descriptions
            # Use word boundary matching to avoid false positives
            skill_words = [w for w in skill_name.split() if len(w) > 3]
            if any(word in all_descriptions for word in skill_words):
                corroborated_skills += 1

        total_skills = len(skills)
        if total_skills == 0:
            return False

        corroboration_rate = corroborated_skills / total_skills

        # Primary check: large skills list + low corroboration
        if total_skills <= 25 or corroboration_rate >= 0.40:
            return False

        # B5 improved: require additional supporting signal before flagging.
        # Check for production deployment evidence - if present, the candidate
        # likely has genuine work even with many skills.
        if Config.USE_IMPROVED_KEYWORD_STUFFER:
            production_indicators = {
                "production", "deployed", "shipped", "serving", "real users",
                "user-facing", "a/b test", "latency", "sla", "qps",
                "million", "billion", "scale",
            }
            has_production_evidence = any(
                term in all_descriptions for term in production_indicators
            )

            # If candidate has production evidence AND some corroboration (>20%),
            # do not flag - they may be a genuine polymath
            if has_production_evidence and corroboration_rate >= 0.20:
                return False

            # Require either high assessment scores or production evidence
            # to override the stuffing flag
            # (assessment check is done by caller via trap_flags context)

        # Flag as keyword stuffer: large skills + low corroboration + no production evidence
        return True


    def _is_recent_hype_pivot(
        self, career: List[dict], all_descriptions: str
    ) -> bool:
        """
        Returns True if the candidate's AI/ML experience is entirely post-2022
        with no pre-LLM-era ML foundation.

        This detects the pattern of: "was doing something else until 2022-2023,
        then pivoted to 'AI Engineer' after the ChatGPT hype cycle."

        Two conditions:
        1. The first ML/AI-titled role starts after 2022-01-01
        2. Career descriptions before 2022 show no ML vocabulary
        """
        if not career:
            return False

        ml_title_keywords = {
            "machine learning", "ml", "ai", "data science", "deep learning",
            "nlp", "recommendation", "search", "retrieval", "ranking",
        }

        # Sort career by start date to analyze chronologically
        sorted_career = sorted(
            career,
            key=lambda r: r.get("start_date") or "2000-01-01",
        )

        # Find the earliest role with an ML/AI title
        first_ml_role_year = None
        for role in sorted_career:
            title_lower = (role.get("title") or "").lower()
            if any(kw in title_lower for kw in ml_title_keywords):
                try:
                    start_str = role.get("start_date") or ""
                    first_ml_role_year = int(start_str[:4])
                    break
                except (ValueError, IndexError):
                    continue

        if first_ml_role_year is None:
            # No ML/AI-titled role found at all
            return False

        # If the first ML role started before 2022, this is a genuine trajectory
        if first_ml_role_year < 2022:
            return False

        # First ML role is 2022 or later - check if descriptions show any pre-2022 ML work
        pre_2022_ml_vocabulary = {
            "tensorflow", "pytorch", "scikit-learn", "sklearn",
            "neural network", "gradient boosting", "random forest",
            "deep learning", "word2vec", "bert", "transformer",
            "recommendation", "collaborative filtering",
            "xgboost", "lightgbm",
        }
        # Pre-2022 descriptions
        pre_2022_descriptions = " ".join(
            (r.get("description") or "").lower()
            for r in sorted_career
            if (r.get("start_date") or "2025") < "2022"
        ).lower()

        has_pre_2022_ml_foundation = any(
            kw in pre_2022_descriptions for kw in pre_2022_ml_vocabulary
        )

        # Recent hype pivot: first ML role is 2022+, no pre-2022 ML foundation
        return not has_pre_2022_ml_foundation

    def _is_title_chaser(self, career: List[dict]) -> bool:
        """
        Detects the "title-chaser" pattern: advancing seniority by switching
        companies every 1.5 years or less, with titles escalating faster than
        career maturity normally allows.

        This pattern is explicitly listed as a disqualifier in the JD.
        """
        if len(career) < 3:
            return False

        # Sort by start date
        sorted_career = sorted(
            career,
            key=lambda r: r.get("start_date") or "2000-01-01",
        )

        # Count short-tenure roles with title escalation
        short_escalation_count = 0
        title_level_map = {
            "junior": 1, "associate": 1, "engineer": 2, "analyst": 2,
            "senior": 3, "lead": 4, "staff": 5, "principal": 6,
            "director": 7, "vp": 8, "head": 7, "chief": 9, "cto": 9,
        }

        for i in range(1, len(sorted_career)):
            prev_role = sorted_career[i - 1]
            curr_role = sorted_career[i]

            # Check tenure at previous role
            prev_duration = prev_role.get("duration_months", 24)
            if prev_duration > 18:
                continue  # Not a short tenure

            # Check if title escalated
            prev_title = (prev_role.get("title") or "").lower()
            curr_title = (curr_role.get("title") or "").lower()

            prev_level = max(
                (v for k, v in title_level_map.items() if k in prev_title),
                default=2,
            )
            curr_level = max(
                (v for k, v in title_level_map.items() if k in curr_title),
                default=2,
            )

            # Check if they moved to a different company
            prev_company = (prev_role.get("company") or "").lower()
            curr_company = (curr_role.get("company") or "").lower()
            is_different_company = prev_company != curr_company

            if is_different_company and curr_level > prev_level:
                short_escalation_count += 1

        # 2+ instances of short-tenure title escalation = title chaser pattern
        return short_escalation_count >= 2

    # Assessments that confirm genuine retrieval/NLP expertise when scored >= 55
    RETRIEVAL_ASSESSMENT_NAMES: Set[str] = {
        "semantic search", "information retrieval", "vector search", "faiss",
        "pinecone", "weaviate", "qdrant", "milvus", "embeddings", "openai",
        "rag", "sentence transformer", "sentence transformers", "haystack",
        "llamaindex", "learning to rank", "bm25", "dense retrieval",
        "hybrid search", "recommendation systems",
    }

    def _is_domain_mismatch(
        self,
        skill_names: Set[str],
        all_descriptions: str,
        all_titles: List[str],
        current_title: str = "",
        assessment_scores: Optional[Dict] = None,
    ) -> bool:
        """
        Detects candidates whose primary expertise is in CV/speech/robotics
        with no NLP or information retrieval exposure.

        This is an explicit disqualifier in the JD.

        Two pathways:
        1. Title-based (strong signal): Current title is "Computer Vision Engineer"
           or similar with no redemptive retrieval assessment score >= 55.
           Description vocabulary is unreliable (dataset uses shared templates that
           may contain "recommendation"/"ranking" even for CV-titled roles).
        2. Description-based (original): High non-NLP signals + low NLP/IR signals
           + CV/speech career titles (legacy check, kept for non-title-flagged cases).
        """
        # ── Pathway 1: Current title is a direct CV/speech domain indicator ──────
        # The dataset career descriptions are synthetic templates — a CV Engineer
        # may have been assigned the "recommendation-style features" template which
        # contains "ranking" and "recommendation", preventing the description-based
        # check from firing. Current title is more reliable than description content.
        CV_TITLE_KEYWORDS = {
            "computer vision", "cv engineer", "vision engineer",
            "speech engineer", "speech scientist", "asr engineer",
        }
        current_title_lower = current_title.lower()
        is_cv_titled = any(kw in current_title_lower for kw in CV_TITLE_KEYWORDS)

        if is_cv_titled:
            # Redemption check: strong retrieval-specific assessment scores (>=55)
            # indicate the candidate has genuine NLP/IR expertise despite the CV title.
            has_retrieval_assessment = False
            if assessment_scores:
                for skill_name, score in assessment_scores.items():
                    if score is None:
                        continue
                    skill_lower = skill_name.lower()
                    if (
                        any(rk in skill_lower for rk in self.RETRIEVAL_ASSESSMENT_NAMES)
                        and float(score) >= 55
                    ):
                        has_retrieval_assessment = True
                        break
            if not has_retrieval_assessment:
                return True  # CV title + no retrieval assessment = domain mismatch

        # ── Pathway 2: Description-based check (legacy) ───────────────────────────
        # Count non-NLP domain signals
        non_nlp_signals = sum(
            1 for kw in self.NON_NLP_DOMAIN_KEYWORDS
            if kw in all_descriptions or kw in skill_names
        )

        # Count NLP/retrieval signals
        nlp_ir_signals = sum(
            1 for kw in {
                "nlp", "natural language", "text", "retrieval", "search",
                "recommendation", "ranking", "embedding", "language model",
                "bert", "transformer", "sentiment", "ner", "entity",
            }
            if kw in all_descriptions or kw in skill_names
        )

        # Check titles for CV/speech/robotics focus
        cv_speech_titles = sum(
            1 for title in all_titles
            if any(kw in title for kw in {
                "computer vision", "cv engineer", "speech", "robotics",
                "vision engineer", "image", "autonomous",
            })
        )

        # Domain mismatch: strong non-NLP signal with weak NLP/IR signal
        return (
            non_nlp_signals >= 3
            and nlp_ir_signals <= 1
            and cv_speech_titles >= 1
        )

    def _is_certification_padder(self, skills: list) -> bool:
        """
        Returns True if candidate pads skill list with low-effort entries.

        10+ skills all at beginner level with minimal duration = quantity
        padding to game keyword matchers. Genuine engineers have a mix of
        proficiency levels reflecting real learning curves.
        """
        if len(skills) < 10:
            return False
        beginner_short = sum(
            1 for s in skills
            if s.get("proficiency", "").lower() in ("beginner", "novice")
            and (s.get("duration_months", 0) or 0) <= 3
        )
        return beginner_short >= 10

    def _is_experience_gap(
        self, career: list, claimed_years: float
    ) -> bool:
        """
        Returns True if claimed experience vastly exceeds verifiable career.

        A candidate claiming 8+ years but with career history totaling < 3
        years has a massive unexplained gap. This is either fabricated
        experience or an unreliable profile.
        """
        if claimed_years < 8:
            return False
        if len(career) < 1:
            return False
        total_months = sum(
            max(0, r.get("duration_months", 0) or 0) for r in career
        )
        total_years = total_months / 12.0
        return total_years > 0 and total_years < 3.0