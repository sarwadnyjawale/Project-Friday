"""
semantic_scorer.py - Hybrid retrieval: lexical + semantic embedding similarity.

This module provides semantic understanding that pure lexical matching cannot:
    - Finds Tier 5 candidates who use different vocabulary (e.g., "recommendation
      engine" instead of "recommendation system")
    - Captures semantic relationships between career descriptions and JD requirements
    - Complements (not replaces) BM25 lexical matching

The hybrid score is:
    hybrid_score = alpha * lexical_score + beta * semantic_score

Pre-computed embeddings are loaded at startup (spec section 10.3 allows
pre-computation outside the 5-minute window). If embeddings are not available,
the module gracefully degrades to lexical-only scoring.

Compliance:
    - No network calls during ranking (embeddings are pre-computed)
    - No GPU during ranking (embeddings are loaded from .npy files)
    - Pre-computation script (precompute_embeddings.py) is included in the repo
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Config

logger = logging.getLogger("semantic_scorer")


class SemanticScorer:
    """
    Loads pre-computed embedding similarities and produces hybrid scores.

    Usage:
        scorer = SemanticScorer()
        scorer.load()  # Load pre-computed similarities

        hybrid_score = scorer.hybrid_score(candidate_id, lexical_score)
    """

    def __init__(self) -> None:
        self._similarities: Dict[str, float] = {}
        self._loaded: bool = False
        self._alpha = Config.HYBRID_ALPHA
        self._beta = Config.HYBRID_BETA

    def load(self, similarities_path: Optional[Path] = None) -> bool:
        """
        Load pre-computed embedding similarities from disk.

        Returns True if loaded successfully, False if not available
        (caller should fall back to lexical-only scoring).
        """
        path = Path(similarities_path or Config.SIMILARITIES_FILE)

        if not path.exists():
            logger.info(
                f"Embeddings file not found at {path} - "
                f"semantic scoring disabled (lexical-only mode)"
            )
            self._loaded = False
            return False

        try:
            # Try numpy first (fastest)
            try:
                import numpy as np

                # Load as numpy array of (candidate_id, similarity) pairs
                data = np.load(path, allow_pickle=True)
                if data.ndim == 2 and data.shape[1] == 2:
                    # Format: [[candidate_id, similarity], ...]
                    for row in data:
                        cid = str(row[0])
                        sim = float(row[1])
                        self._similarities[cid] = max(0.0, min(1.0, sim))
                elif data.ndim == 1:
                    # Format: array of similarities, index = candidate index
                    # Need a mapping file
                    mapping_path = Path(Config.EMBEDDINGS_DIR) / "id_mapping.json"
                    if mapping_path.exists():
                        import json
                        with open(mapping_path) as f:
                            id_mapping = json.load(f)
                        for idx, cid in id_mapping.items():
                            self._similarities[cid] = max(0.0, min(1.0, float(data[int(idx)])))
                    else:
                        logger.warning("Similarities array found but no ID mapping")
                        self._loaded = False
                        return False
                else:
                    logger.warning(f"Unexpected similarities format: shape={data.shape}")
                    self._loaded = False
                    return False

                logger.info(
                    f"Loaded {len(self._similarities)} pre-computed similarities from {path}"
                )
                self._loaded = True
                return True

            except ImportError:
                # numpy not available - try JSON fallback
                import json

                if path.suffix == ".json":
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self._similarities = {
                            k: max(0.0, min(1.0, float(v)))
                            for k, v in data.items()
                        }
                    elif isinstance(data, list):
                        # List of [candidate_id, similarity] pairs
                        for item in data:
                            cid = str(item[0])
                            sim = float(item[1])
                            self._similarities[cid] = max(0.0, min(1.0, sim))

                    logger.info(
                        f"Loaded {len(self._similarities)} similarities from JSON: {path}"
                    )
                    self._loaded = True
                    return True

                logger.warning("numpy not available and file is not JSON")
                self._loaded = False
                return False

        except Exception as e:
            logger.warning(f"Failed to load embeddings from {path}: {e}")
            self._loaded = False
            return False

    @property
    def is_available(self) -> bool:
        """Returns True if semantic similarities are loaded and ready."""
        return self._loaded and len(self._similarities) > 0

    def get_similarity(self, candidate_id: str) -> float:
        """
        Get the pre-computed semantic similarity for a candidate.
        Returns 0.0 if not available (graceful degradation).
        """
        return self._similarities.get(candidate_id, 0.0)

    def hybrid_score(self, candidate_id: str, lexical_score: float) -> float:
        """
        Compute hybrid score combining lexical and semantic signals.

        hybrid = alpha * lexical + beta * semantic

        If semantic similarity is not available for this candidate,
        returns the lexical score unchanged (graceful degradation).
        """
        if not self._loaded:
            return lexical_score

        semantic = self._similarities.get(candidate_id, 0.0)
        if semantic == 0.0:
            # No embedding for this candidate - use lexical only
            return lexical_score

        return self._alpha * lexical_score + self._beta * semantic

    def hybrid_score_normalized(
        self, candidate_id: str, lexical_score: float
    ) -> Tuple[float, float]:
        """
        Compute hybrid score and return both the hybrid score and semantic component.

        Returns:
            (hybrid_score, semantic_similarity)
        """
        if not self._loaded:
            return lexical_score, 0.0

        semantic = self._similarities.get(candidate_id, 0.0)
        hybrid = self.hybrid_score(candidate_id, lexical_score)
        return hybrid, semantic

    def stats(self) -> Dict:
        """Return statistics about loaded similarities."""
        if not self._loaded:
            return {"loaded": False, "count": 0}
        sims = list(self._similarities.values())
        return {
            "loaded": True,
            "count": len(sims),
            "min": min(sims) if sims else 0,
            "max": max(sims) if sims else 0,
            "mean": sum(sims) / len(sims) if sims else 0,
            "alpha": self._alpha,
            "beta": self._beta,
        }
