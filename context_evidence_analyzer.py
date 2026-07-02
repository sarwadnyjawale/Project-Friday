"""
context_evidence_analyzer.py

Analyzes the linguistic context around technical terms in career descriptions.
Distinguishes between "built and deployed X" vs "evaluated X but chose Y"
vs "familiar with X from a course."

This addresses the fundamental limitation of term-presence matching:
a technical term appearing in a negative context should not receive the
same credit as the same term in a strong production context.

Integration:
    - Called after EvidenceConsistencyEngine in rank.py
    - Receives candidates list and jd_context
    - Returns {candidate_id: ContextEvidenceBundle}
    - feature_extractor.py's FeatureBundle is extended with a context_score field
    - ScoringEngine uses context_score as an additional A-component adjustment

Why this improves NDCG@10:
    Positive technical contexts in career descriptions are the strongest
    signal that a candidate has genuine hands-on experience. The ground truth
    top candidates will have heavy positive contexts. This module directly
    measures that signal.

Why this does not replace the existing BM25 approach:
    BM25 correctly finds candidates whose careers are relevant.
    Context analysis correctly distinguishes depth within those candidates.
    They are complementary, not competing.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from jd_analyzer import JDContext

logger = logging.getLogger("context_evidence_analyzer")


# ─────────────────────────────────────────────────────────────────────────────
# Context window configuration
# ─────────────────────────────────────────────────────────────────────────────

# Characters to look at before and after a technical term
CONTEXT_WINDOW_CHARS = 120

# Context scoring weights
CONTEXT_SCORES = {
    "strong_positive": 1.00,   # Built/deployed/shipped + scale/outcome
    "positive": 0.70,           # Technical action verb without scale
    "neutral": 0.35,            # Mentioned without action context
    "weak": 0.15,               # Passing mention or tangential
    "negative": 0.05,           # Negative/exploratory context
}

# Strong positive context indicators
STRONG_POSITIVE_INDICATORS = [
    # Action + scale patterns
    (r'\b(deployed|shipped|launched|built|designed|implemented)\b', r'\b(million|billion|k\s*qps|at\s*scale|production)\b'),
    # Improvement + metric patterns
    (r'\b(improved|increased|reduced|optimized|achieved)\b', r'\b(\d+\s*%|ndcg|mrr|recall|latency)\b'),
    # Ownership patterns
    (r'\b(owned|led|architected|responsible\s+for)\b', r'\b(system|pipeline|infrastructure|platform|service)\b'),
]

# Positive context: technical action verb without necessarily having scale info
POSITIVE_VERB_PATTERNS = re.compile(
    r'\b(built|developed|implemented|designed|deployed|shipped|optimized|'
    r'trained|fine-tuned|evaluated|benchmarked|maintained|operated|'
    r'architected|migrated|integrated|scaled|launched|improved|created)\b',
    re.IGNORECASE
)

# Negative/exploratory context patterns
NEGATIVE_CONTEXT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\b(evaluated\s+but|decided\s+against|chose\s+not\s+to|moved\s+away\s+from)\b',
        r'\b(tutorial|online\s+course|coursera|udemy|workshop|academic\s+project)\b',
        r'\b(proof[\s-]of[\s-]concept|poc|prototype\s+only|experimental)\b',
        r'\b(basic\s+understanding|familiar\s+with|exposure\s+to|introductory)\b',
        r'\b(read\s+about|learned\s+about|studied|explored)\b',
    ]
]

# Quantification patterns: numbers that indicate real scale
QUANTIFICATION_PATTERNS = re.compile(
    r'\b(\d+[\.,]?\d*\s*('
    r'ms|milliseconds|seconds|'
    r'k|m|b|'
    r'million|billion|thousand|'
    r'%|percent|'
    r'qps|rps|tps|'
    r'gb|tb|pb|'
    r'users|requests|queries|documents|records|items|candidates'
    r'))\b',
    re.IGNORECASE
)


@dataclass
class TermContext:
    """Context analysis result for one occurrence of a JD term."""
    term: str
    sentence: str
    context_type: str        # strong_positive/positive/neutral/weak/negative
    context_score: float     # From CONTEXT_SCORES
    has_quantification: bool
    has_action_verb: bool
    is_negative: bool
    recency_weight: float


@dataclass
class ContextEvidenceBundle:
    """
    Context analysis results for all JD-relevant terms in a candidate's profile.
    """
    candidate_id: str

    # Per-term context analysis: {term: [list of TermContext]}
    term_contexts: Dict[str, List[TermContext]] = field(default_factory=dict)

    # Aggregate context signals
    strong_positive_term_count: int = 0   # Terms with at least one strong positive context
    positive_term_count: int = 0          # Terms with at least one positive context
    negative_context_term_count: int = 0  # Terms appearing only in negative contexts
    quantified_claim_count: int = 0       # Technical claims with numbers/metrics

    # Context quality scores per cluster
    retrieval_context_score: float = 0.0  # Quality of retrieval-term contexts
    ml_production_context_score: float = 0.0

    # Overall context quality (0.0-1.0)
    overall_context_quality: float = 0.0

    # Adjustment for ScoringEngine
    context_score_adjustment: float = 0.0  # Additive adjustment to score_A (-0.12 to +0.12)

    # Top evidence sentences for reasoning
    top_evidence_sentences: List[str] = field(default_factory=list)


class ContextEvidenceAnalyzer:
    """
    Analyzes linguistic context around JD technical terms in career descriptions.

    Key design decision: We only analyze terms from the high-weight JD clusters
    (retrieval and ML production). Analyzing all JD terms would be too noisy
    and too slow for the 5-minute budget.
    """

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context
        # Focus on the highest-weight terms only for context analysis
        # This covers the JD's must-have requirements without over-reaching
        self._focus_terms = self._build_focus_terms()
        self._strong_positive_compiled = [
            (re.compile(p1, re.IGNORECASE), re.compile(p2, re.IGNORECASE))
            for p1, p2 in STRONG_POSITIVE_INDICATORS
        ]

    def analyze_all(
        self, candidates: List[dict]
    ) -> Dict[str, ContextEvidenceBundle]:
        """
        Run context analysis on all candidates.
        Returns {candidate_id: ContextEvidenceBundle}
        """
        results: Dict[str, ContextEvidenceBundle] = {}

        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]
            try:
                results[cid] = self._analyze_candidate(candidate)
            except Exception as e:
                logger.warning(f"Context analysis error for {cid}: {e}")
                results[cid] = ContextEvidenceBundle(candidate_id=cid)

            if (i + 1) % 10_000 == 0:
                logger.info(
                    f"  Context analysis: {i+1:,} candidates processed"
                )

        return results

    def _analyze_candidate(self, candidate: dict) -> ContextEvidenceBundle:
        """Context analysis for one candidate."""
        bundle = ContextEvidenceBundle(candidate_id=candidate["candidate_id"])
        career = candidate.get("career_history", [])

        if not career:
            return bundle

        # Build sorted career by recency
        sorted_career = sorted(
            career,
            key=lambda r: r.get("start_date") or "2000-01-01",
            reverse=True,
        )

        # Concatenated text per role with recency weights
        role_texts: List[Tuple[str, float]] = []
        for rank_idx, role in enumerate(sorted_career):
            desc = (role.get("description") or "").strip()
            if desc:
                recency_weight = max(0.25, 1.0 - rank_idx * 0.15)
                role_texts.append((desc.lower(), recency_weight))

        # ── Analyze each focus term ──────────────────────────────────────────
        for term in self._focus_terms:
            term_contexts = self._analyze_term_contexts(
                term, role_texts
            )
            if term_contexts:
                bundle.term_contexts[term] = term_contexts

        # ── Compute aggregate signals ────────────────────────────────────────
        self._compute_aggregates(bundle)

        # ── Compute adjustment factor ────────────────────────────────────────
        self._compute_adjustment(bundle)

        # ── Extract top evidence sentences ───────────────────────────────────
        bundle.top_evidence_sentences = self._extract_top_sentences(bundle)

        return bundle

    def _analyze_term_contexts(
        self,
        term: str,
        role_texts: List[Tuple[str, float]],
    ) -> List[TermContext]:
        """
        Find all occurrences of a term and analyze their context.
        Returns list of TermContext objects.
        """
        contexts: List[TermContext] = []

        # Build a single regex for this term (handles both single and multi-word)
        term_pattern = re.compile(
            r'\b' + re.escape(term) + r'\b' if len(term.split()) == 1
            else re.escape(term),
            re.IGNORECASE,
        )

        for text, recency_weight in role_texts:
            for m in term_pattern.finditer(text):
                pos = m.start()
                ctx_start = max(0, pos - CONTEXT_WINDOW_CHARS)
                ctx_end = min(len(text), pos + len(term) + CONTEXT_WINDOW_CHARS)
                context_window = text[ctx_start:ctx_end]

                # Get the containing sentence for better analysis
                sentence = self._get_containing_sentence(text, pos)

                # Classify context
                ctx_type, ctx_score, has_quant, has_verb, is_neg = (
                    self._classify_context(context_window, sentence)
                )

                contexts.append(TermContext(
                    term=term,
                    sentence=sentence,
                    context_type=ctx_type,
                    context_score=ctx_score * recency_weight,
                    has_quantification=has_quant,
                    has_action_verb=has_verb,
                    is_negative=is_neg,
                    recency_weight=recency_weight,
                ))

        return contexts

    def _classify_context(
        self, context_window: str, sentence: str
    ) -> Tuple[str, float, bool, bool, bool]:
        """
        Classify the context type of a term occurrence.

        Returns: (context_type, context_score, has_quantification,
                  has_action_verb, is_negative)
        """
        # Check for negative context first (highest priority)
        is_negative = any(p.search(sentence) for p in NEGATIVE_CONTEXT_PATTERNS)
        if is_negative:
            return "negative", CONTEXT_SCORES["negative"], False, False, True

        has_quant = bool(QUANTIFICATION_PATTERNS.search(context_window))
        has_verb = bool(POSITIVE_VERB_PATTERNS.search(context_window))

        # Check for strong positive patterns (two complementary signals)
        for verb_pattern, scale_pattern in self._strong_positive_compiled:
            if verb_pattern.search(context_window) and scale_pattern.search(context_window):
                return "strong_positive", CONTEXT_SCORES["strong_positive"], has_quant, has_verb, False

        # Positive: action verb present
        if has_verb and has_quant:
            return "positive", CONTEXT_SCORES["positive"], has_quant, has_verb, False
        elif has_verb:
            return "positive", CONTEXT_SCORES["positive"] * 0.80, has_quant, has_verb, False

        # Neutral or weak
        if has_quant:
            return "neutral", CONTEXT_SCORES["neutral"], has_quant, has_verb, False
        else:
            return "weak", CONTEXT_SCORES["weak"], has_quant, has_verb, False

    def _get_containing_sentence(self, text: str, pos: int) -> str:
        """Extract the sentence containing the position."""
        # Find sentence boundaries
        sentence_end_patterns = r'[.;\n]'
        # Look backwards for start
        prev_boundary = max(0, pos - 300)
        text_before = text[prev_boundary:pos]
        boundaries = list(re.finditer(sentence_end_patterns, text_before))
        if boundaries:
            sent_start = prev_boundary + boundaries[-1].end()
        else:
            sent_start = prev_boundary

        # Look forwards for end
        text_after = text[pos:pos + 300]
        end_match = re.search(sentence_end_patterns, text_after)
        sent_end = pos + (end_match.start() if end_match else len(text_after))

        return text[sent_start:sent_end].strip()

    def _compute_aggregates(self, bundle: ContextEvidenceBundle) -> None:
        """Compute bundle-level aggregate context signals."""
        retrieval_scores = []
        ml_scores = []

        for term, contexts in bundle.term_contexts.items():
            if not contexts:
                continue

            # Best context for this term (max score)
            best_ctx = max(contexts, key=lambda c: c.context_score)

            if best_ctx.context_type == "strong_positive":
                bundle.strong_positive_term_count += 1
                bundle.positive_term_count += 1
            elif best_ctx.context_type == "positive":
                bundle.positive_term_count += 1
            elif all(c.is_negative for c in contexts):
                bundle.negative_context_term_count += 1

            if any(c.has_quantification for c in contexts):
                bundle.quantified_claim_count += 1

            # Categorize by cluster
            if term in self.jd.retrieval_search_terms:
                retrieval_scores.append(best_ctx.context_score)
            elif term in self.jd.ml_production_terms:
                ml_scores.append(best_ctx.context_score)

        bundle.retrieval_context_score = (
            sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        )
        bundle.ml_production_context_score = (
            sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
        )

        # Overall: weighted towards retrieval (Cluster 1 weight)
        all_scores = retrieval_scores + ml_scores
        bundle.overall_context_quality = (
            sum(all_scores) / len(all_scores) if all_scores else 0.0
        )

    def _compute_adjustment(self, bundle: ContextEvidenceBundle) -> None:
        """
        Compute the context_score_adjustment for score_A.

        Strong positive contexts -> positive adjustment (up to +0.12)
        Negative contexts dominating -> negative adjustment (down to -0.10)
        """
        if bundle.strong_positive_term_count >= 4:
            bundle.context_score_adjustment = min(
                0.12, bundle.strong_positive_term_count * 0.02
            )
        elif bundle.strong_positive_term_count >= 2:
            bundle.context_score_adjustment = 0.06
        elif bundle.negative_context_term_count >= 3:
            bundle.context_score_adjustment = max(
                -0.10, -bundle.negative_context_term_count * 0.03
            )
        elif bundle.quantified_claim_count >= 3:
            bundle.context_score_adjustment = 0.04
        else:
            bundle.context_score_adjustment = 0.0

    def _extract_top_sentences(
        self, bundle: ContextEvidenceBundle
    ) -> List[str]:
        """
        Extract the highest-quality evidence sentences for reasoning generation.
        Returns up to 3 strong positive evidence sentences.
        """
        strong_sentences = []
        for term, contexts in bundle.term_contexts.items():
            for ctx in contexts:
                if ctx.context_type == "strong_positive" and ctx.has_quantification:
                    strong_sentences.append((ctx.context_score, ctx.sentence))

        strong_sentences.sort(key=lambda x: -x[0])
        return [s for _, s in strong_sentences[:3]]

    def _build_focus_terms(self) -> List[str]:
        """
        Build the list of terms to analyze context for.
        Focus on highest-weight terms only - analyzing all 200+ JD terms
        would be too slow and too noisy.
        """
        # Only analyze terms with weight >= 0.65 (retrieval + ML production clusters)
        focus = [
            term for term, weight in self.jd.term_weights.items()
            if weight >= 0.65
        ]
        # Sort by length (longer terms first to avoid partial matches)
        return sorted(focus, key=len, reverse=True)