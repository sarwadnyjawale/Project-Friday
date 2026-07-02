"""
bm25_scorer.py - Mathematically correct BM25 scoring for candidate descriptions.

Replaces the linear term-presence scoring in feature_extractor.py with
a proper BM25 implementation featuring:
    - IDF (Inverse Document Frequency): downweights common terms
    - TF saturation: (k1+1)*tf / (k1+tf) caps after ~3 occurrences
    - Document length normalization: prevents verbosity bias

This is a drop-in replacement controlled by Config.USE_NEW_BM25 flag.
When the flag is False, the original feature_extractor scoring is used.

Mathematical Foundation:
    BM25(doc, query) = sum over terms t in query of:
        IDF(t) * TF_score(t, doc) * length_norm(doc)

    where:
        IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        TF_score = (k1 + 1) * tf / (k1 * (1 - b + b * |doc| / avg_doc_len) + tf)
        k1 = 1.2 (TF saturation parameter)
        b = 0.75 (length normalization parameter)

Cluster weighting from the JD vocabulary is preserved as a multiplier
on the IDF score, encoding JD-specific term importance.

Performance: O(N * D * V) where N=candidates, D=avg desc length, V=vocab size.
Optimized with pre-computed IDF and set-based term lookup.
Target: <30 seconds for 100K candidates.
"""

import logging
import math
import re
from typing import Dict, List, Set, Tuple

from jd_analyzer import JDContext

logger = logging.getLogger("bm25_scorer")


class BM25Scorer:
    """
    BM25 scorer with IDF, TF saturation, and document length normalization.

    Usage:
        # Phase 1: Build the index (compute IDF over all documents)
        scorer = BM25Scorer(jd_context)
        scorer.build_index(all_descriptions_per_candidate)

        # Phase 2: Score each candidate
        for cid, description in candidate_descriptions:
            score, terms_found = scorer.score(description)
    """

    # BM25 parameters (standard values from IR literature)
    K1: float = 1.2  # TF saturation: higher = more saturation
    B: float = 0.75  # Length normalization: 0 = none, 1 = full

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context
        self._idf_cache: Dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0
        self._built: bool = False
        # Pre-compute sorted term list for efficient matching
        self._jd_terms_sorted = sorted(
            self.jd.term_weights.keys(),
            key=len,
            reverse=True,  # Longer terms first to match multi-word before single-word
        )

    def build_index(self, documents: List[str]) -> None:
        """
        Build the BM25 index by computing IDF for all JD terms.

        Args:
            documents: List of all candidate description texts (concatenated
                       career descriptions per candidate).
        """
        self._total_docs = len(documents)
        if self._total_docs == 0:
            logger.warning("BM25 index built with 0 documents")
            self._built = True
            return

        # Compute average document length (in words)
        doc_lengths = [len(doc.split()) for doc in documents if doc]
        self._avg_doc_len = sum(doc_lengths) / max(1, len(doc_lengths))

        # Compute document frequency for each JD term
        df_counts: Dict[str, int] = {}
        for term in self._jd_terms_sorted:
            df = 0
            for doc in documents:
                if not doc:
                    continue
                doc_lower = doc.lower() if not doc.islower() else doc
                if len(term.split()) > 1:
                    if term in doc_lower:
                        df += 1
                else:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, doc_lower):
                        df += 1
            df_counts[term] = df

        # Compute IDF for each term using BM25's probabilistic IDF
        # IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5))
        # This ensures non-negative IDF (unlike standard IDF for very common terms)
        for term, df in df_counts.items():
            if df == 0:
                self._idf_cache[term] = 0.0
            else:
                self._idf_cache[term] = math.log(
                    1.0 + (self._total_docs - df + 0.5) / (df + 0.5)
                )

        self._built = True
        logger.info(
            f"BM25 index built: {self._total_docs} docs, "
            f"avg_len={self._avg_doc_len:.0f} words, "
            f"{len(self._idf_cache)} terms indexed"
        )

    def score(self, description: str) -> Tuple[float, List[str]]:
        """
        Score a single candidate description against the JD vocabulary.

        Returns:
            Tuple of (bm25_score, list_of_matched_terms)
        """
        if not self._built:
            logger.warning("BM25 scorer used before build_index() - returning 0")
            return 0.0, []

        if not description:
            return 0.0, []

        text = description.lower()
        doc_len = len(text.split())
        if doc_len == 0:
            return 0.0, []

        # Length normalization factor
        length_norm = 1.0 - self.B + self.B * (doc_len / max(1, self._avg_doc_len))

        total_score = 0.0
        terms_found: List[str] = []

        for term in self._jd_terms_sorted:
            # Check if term appears in document
            if len(term.split()) > 1:
                # Multi-word term: direct substring match
                if term not in text:
                    continue
                tf = text.count(term)
            else:
                # Single-word term: word boundary match
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.findall(pattern, text)
                if not matches:
                    continue
                tf = len(matches)

            if tf == 0:
                continue

            terms_found.append(term)

            # Get IDF (importance of this term in the corpus)
            idf = self._idf_cache.get(term, 0.0)
            if idf == 0.0:
                continue

            # Get cluster weight (importance of this term in the JD)
            cluster_weight = self.jd.term_weights.get(term, 0.3)

            # TF saturation: prevents keyword stuffing from inflating score
            # (k1 + 1) * tf / (k1 * length_norm + tf)
            tf_score = (self.K1 + 1.0) * tf / (self.K1 * length_norm + tf)

            # Final term contribution
            term_score = idf * tf_score * cluster_weight
            total_score += term_score

        return total_score, terms_found

    def score_candidate(self, career_history: List[dict]) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Score a candidate's full career history against the JD.

        Applies recency weighting: most recent role counts 1.0x,
        each prior role decays by 0.15x (minimum 0.2x).

        Returns:
            Tuple of (score, all_terms, retrieval_terms, ml_terms)
        """
        if not career_history:
            return 0.0, [], [], []

        # Sort by recency (most recent first)
        sorted_career = sorted(
            career_history,
            key=lambda r: r.get("start_date") or "2000-01-01",
            reverse=True,
        )

        all_terms: List[str] = []
        retrieval_terms: List[str] = []
        ml_terms: List[str] = []
        total_weighted_score = 0.0

        for role_idx, role in enumerate(sorted_career):
            recency_weight = max(0.2, 1.0 - role_idx * 0.15)
            description = (role.get("description") or "").lower()
            if not description:
                continue

            role_score, terms_in_role = self.score(description)
            all_terms.extend(terms_in_role)

            # Categorize terms for reasoning
            for term in terms_in_role:
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

        # Deduplicate term lists (preserve order)
        all_terms = list(dict.fromkeys(all_terms))
        retrieval_terms = list(dict.fromkeys(retrieval_terms))
        ml_terms = list(dict.fromkeys(ml_terms))

        # Normalize: cap at a reasonable ceiling
        # With proper IDF, a strong candidate matching 10 high-IDF terms
        # typically scores 15-40. Normalize to [0, 1].
        normalization_ceiling = 15.0
        score = min(1.0, total_weighted_score / normalization_ceiling)

        return score, all_terms, retrieval_terms, ml_terms
