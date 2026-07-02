"""
config.py - Feature flags and deterministic configuration for ICRS.

All feature flags default to False, preserving the original engine behavior.
Flags are toggled via environment variables or direct modification.

REFERENCE_DATE replaces date.today() everywhere, ensuring deterministic
outputs regardless of when the pipeline runs. This is critical for
Stage 3 code reproduction compliance (spec section 10.3).

Usage:
    from config import Config
    if Config.USE_NEW_BM25:
        score = bm25_scorer.score(...)
    else:
        score = legacy_term_matching(...)
"""

import os
from datetime import date


class Config:
    """
    Feature flag system for incremental ranking improvements.

    Every flag is independently toggleable. When False, the original
    engine behavior is preserved exactly. This allows:
      - A/B comparison of old vs new behavior
      - Isolation of regressions to specific features
      - Safe rollback without code changes
      - Compliance with "do not rewrite, evolve" mandate
    """

    # ── Phase A: Safe Improvements ──────────────────────────────────────
    # Deterministic reference date - replaces date.today() everywhere.
    # Uses the most recent data point in a typical hackathon dataset.
    # This ensures behavioral scores (days_since_active) are stable.
    REFERENCE_DATE: date = date(2026, 6, 27)

    # ── Phase B: Ranking Improvements ───────────────────────────────────

    # B1: Use mathematically correct BM25 instead of linear term sum.
    # When True: bm25_scorer.py computes IDF + TF saturation + length norm.
    # When False: feature_extractor.py uses original cluster-weighted term presence.
    USE_NEW_BM25: bool = os.environ.get("USE_NEW_BM25", "false").lower() == "true"

    # B2: Hybrid retrieval - add pre-computed semantic embedding similarity.
    # When True: semantic_scorer.py loads pre-computed embeddings and
    #   produces a hybrid score = alpha * lexical + beta * semantic.
    # When False: only lexical (BM25 or original) scoring is used.
    # Requires: pre-computed embeddings (run precompute_embeddings.py first).
    # Gracefully degrades to lexical-only if embeddings file is missing.
    USE_HYBRID_SEMANTIC: bool = os.environ.get("USE_HYBRID_SEMANTIC", "false").lower() == "true"

    # B3: Feature interaction engine.
    # When True: interaction_engine.py detects multi-feature combinations
    #   (e.g., retrieval + vector_db + embeddings = strong bonus).
    # When False: features are scored independently (original behavior).
    USE_INTERACTION_ENGINE: bool = os.environ.get("USE_INTERACTION_ENGINE", "true").lower() == "true"

    # B4: New confidence mode - tie-breaker only, no global score adjustment.
    # When True: confidence_estimator.py does NOT apply +/-10% adjustment.
    #   Confidence is used ONLY for tie-breaking in ranker.py.
    # When False: confidence applies +/-10% to final_score (original behavior).
    USE_NEW_CONFIDENCE: bool = os.environ.get("USE_NEW_CONFIDENCE", "true").lower() == "true"

    # B5: Improved keyword stuffing detection.
    # When True: trap_detector.py requires multiple supporting signals
    #   (skills/career mismatch + low assessment + no production evidence).
    # When False: original keyword_stuffer logic (threshold-based only).
    USE_IMPROVED_KEYWORD_STUFFER: bool = os.environ.get("USE_IMPROVED_KEYWORD_STUFFER", "true").lower() == "true"

    # B6: Strengthened honeypot detection for high-confidence checks.
    # When True: single high-confidence signal (inverted salary, expert+0dur,
    #   impossible breadth, overlapping FT roles) triggers 0.30x penalty.
    # When False: requires 2+ signals for any penalty (original behavior).
    USE_STRENGTHENED_HONEYPOT: bool = os.environ.get("USE_STRENGTHENED_HONEYPOT", "true").lower() == "true"

    # B7: Top-100 verification layer.
    # When True: top100_verifier.py runs pre-export checks (duplicates,
    #   honeypot rate, score distribution, semantic diversity).
    # When False: no pre-export verification beyond validator.py.
    USE_TOP100_VERIFICATION: bool = os.environ.get("USE_TOP100_VERIFICATION", "true").lower() == "true"

    # ── Experimental (off by default) ────────────────────────────────────

    # RRF experiment - use Reciprocal Rank Fusion instead of linear weighted sum.
    # OFF by default. The linear sum is the proven default.
    # When True: ranker.py fuses multiple signal ranks via RRF formula.
    USE_RRF_EXPERIMENT: bool = os.environ.get("USE_RRF_EXPERIMENT", "false").lower() == "true"

    # ── Honeypot multiplier overrides (when USE_STRENGTHENED_HONEYPOT=True) ──
    HONEYPOT_1_HIGH_CONFIDENCE_MULT: float = 0.30
    HONEYPOT_2_SIGNALS_MULT: float = 0.05
    HONEYPOT_3PLUS_MULT: float = 0.01

    # ── Trap multiplier overrides ───────────────────────────────────────
    KEYWORD_STUFFER_MULT: float = 0.20
    BEHAVIORAL_DEAD_MULT: float = 0.50  # Compromise: 0.70 original -> 0.50

    # ── Semantic similarity configuration ───────────────────────────────
    # Alpha = weight for lexical score in hybrid fusion
    # Beta = weight for semantic score in hybrid fusion
    # Alpha + Beta should = 1.0
    HYBRID_ALPHA: float = 0.60  # Lexical (BM25)
    HYBRID_BETA: float = 0.40   # Semantic (embeddings)

    # Embedding model for pre-computation
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # Paths for pre-computed artifacts
    EMBEDDINGS_DIR: str = "embeddings"
    EMBEDDINGS_FILE: str = "embeddings/candidate_embeddings.npy"
    JD_EMBEDDING_FILE: str = "embeddings/jd_embedding.npy"
    SIMILARITIES_FILE: str = "embeddings/similarities.npy"

    # ── Enhanced pipeline configuration ────────────────────────────────
    # Number of top candidates to run expensive enhanced analysis on.
    # Reduced from 2000 to 500 for compute safety (B5 from design doc).
    ENHANCED_CANDIDATE_LIMIT: int = 2000

    # ── Top-100 honeypot safety threshold ────────────────────────────────
    # If honeypot rate in top 100 exceeds this fraction, demote honeypots.
    # Set below the 10% disqualification threshold for safety margin.
    TOP100_HONEYPOT_RATE_THRESHOLD: float = 0.05  # 5% internal threshold

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable summary of active feature flags."""
        flags = [
            ("USE_NEW_BM25", cls.USE_NEW_BM25),
            ("USE_HYBRID_SEMANTIC", cls.USE_HYBRID_SEMANTIC),
            ("USE_INTERACTION_ENGINE", cls.USE_INTERACTION_ENGINE),
            ("USE_NEW_CONFIDENCE", cls.USE_NEW_CONFIDENCE),
            ("USE_IMPROVED_KEYWORD_STUFFER", cls.USE_IMPROVED_KEYWORD_STUFFER),
            ("USE_STRENGTHENED_HONEYPOT", cls.USE_STRENGTHENED_HONEYPOT),
            ("USE_TOP100_VERIFICATION", cls.USE_TOP100_VERIFICATION),
            ("USE_RRF_EXPERIMENT", cls.USE_RRF_EXPERIMENT),
        ]
        active = [name for name, on in flags if on]
        inactive = [name for name, on in flags if not on]
        lines = ["Feature Flag Summary:"]
        lines.append("  Active:   " + (", ".join(active) if active else "(none - original engine)"))
        lines.append(f"  Inactive: {', '.join(inactive)}" if inactive else "")
        lines.append(f"  REFERENCE_DATE: {cls.REFERENCE_DATE.isoformat()}")
        return "\n".join(lines)
