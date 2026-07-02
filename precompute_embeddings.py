"""
precompute_embeddings.py - Pre-compute semantic embeddings for all candidates.

This script runs OFFLINE (before the ranking step). It:
    1. Loads all candidate career descriptions
    2. Embeds them using BGE-small-en-v1.5 (384-dim)
    3. Embeds the JD text
    4. Computes cosine similarity for all candidates
    5. Saves results to embeddings/ directory

This is allowed by spec section 10.3:
    "pre-computation may exceed the 5-minute window, but the ranking step
     that produces the CSV must complete within it."

Usage:
    python precompute_embeddings.py --candidates DATA/candidates.jsonl
    python precompute_embeddings.py --candidates DATA/candidates.jsonl.gz

Requirements:
    pip install sentence-transformers torch numpy

Output:
    embeddings/candidate_embeddings.npy  (100K x 384 float32)
    embeddings/jd_embedding.npy           (384 float32)
    embeddings/similarities.npy           (100K float32)
    embeddings/id_mapping.json            (index -> candidate_id)
"""

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


def load_candidate_descriptions(candidates_path: Path) -> List[Tuple[str, str]]:
    """
    Load all candidate career descriptions.

    Returns list of (candidate_id, concatenated_description) tuples.
    Concatenates the 5 most recent career descriptions per candidate,
    weighted by recency (most recent first).
    """
    candidates = []

    opener = gzip.open if candidates_path.suffix == ".gz" else open
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = obj.get("candidate_id")
            if not cid:
                continue

            # Concatenate career descriptions (most recent 5)
            career = obj.get("career_history", [])
            if career:
                sorted_career = sorted(
                    career,
                    key=lambda r: r.get("start_date") or "2000-01-01",
                    reverse=True,
                )
                descriptions = []
                for idx, role in enumerate(sorted_career[:5]):
                    desc = (role.get("description") or "").strip()
                    if desc:
                        descriptions.append(desc)
                text = " ".join(descriptions)
            else:
                # Fallback: use summary if no career history
                profile = obj.get("profile", {})
                text = (profile.get("summary") or "") + " " + (profile.get("headline") or "")

            # Also include skills as they provide signal
            skills = obj.get("skills", [])
            if skills:
                skill_names = " ".join(s.get("name", "") for s in skills if s.get("name"))
                text = text + " " + skill_names

            candidates.append((cid, text.strip()))

            if (line_num + 1) % 10000 == 0:
                logger.info(f"  Loaded {line_num + 1:,} candidates...")

    logger.info(f"Loaded {len(candidates):,} candidate descriptions")
    return candidates


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 256,
) -> "np.ndarray":
    """Compute embeddings for a list of texts using sentence-transformers."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "Required packages not installed. Run:\n"
            "  pip install sentence-transformers torch numpy"
        )
        sys.exit(1)

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Computing embeddings for {len(texts):,} texts...")
    t = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    elapsed = time.perf_counter() - t
    logger.info(
        f"Embeddings computed in {elapsed:.1f}s | "
        f"Shape: {embeddings.shape} | "
        f"Dtype: {embeddings.dtype}"
    )

    return embeddings


def compute_similarities(
    candidate_embeddings: "np.ndarray",
    jd_embedding: "np.ndarray",
) -> "np.ndarray":
    """Compute cosine similarity between each candidate and the JD."""
    import numpy as np

    # Both are L2-normalized, so cosine similarity = dot product
    similarities = np.dot(candidate_embeddings, jd_embedding)

    # Clip to [0, 1] (negative similarities are irrelevant for ranking)
    similarities = np.clip(similarities, 0.0, 1.0)

    return similarities


def save_results(
    candidate_ids: List[str],
    embeddings,
    jd_embedding,
    similarities,
    output_dir: Path,
):
    """Save all pre-computed artifacts."""
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    emb_path = output_dir / "candidate_embeddings.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    logger.info(f"Saved embeddings: {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")

    # Save JD embedding
    jd_path = output_dir / "jd_embedding.npy"
    np.save(jd_path, jd_embedding.astype(np.float32))
    logger.info(f"Saved JD embedding: {jd_path}")

    # Save similarities
    sim_path = output_dir / "similarities.npy"
    np.save(sim_path, similarities.astype(np.float32))
    logger.info(f"Saved similarities: {sim_path}")

    # Save ID mapping (index -> candidate_id)
    mapping = {str(i): cid for i, cid in enumerate(candidate_ids)}
    mapping_path = output_dir / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    logger.info(f"Saved ID mapping: {mapping_path} ({len(mapping):,} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute semantic embeddings for Redrob hackathon"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(Config.EMBEDDINGS_DIR),
        help=f"Output directory (default: {Config.EMBEDDINGS_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Config.EMBEDDING_MODEL,
        help=f"Embedding model (default: {Config.EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PRE-COMPUTE EMBEDDINGS")
    logger.info("=" * 60)
    logger.info(f"Candidates: {args.candidates}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")

    # Step 1: Load candidate descriptions
    t = time.perf_counter()
    candidates = load_candidate_descriptions(args.candidates)
    logger.info(f"Loaded {len(candidates):,} candidates in {time.perf_counter() - t:.1f}s")

    candidate_ids = [c[0] for c in candidates]
    candidate_texts = [c[1] for c in candidates]

    # Step 2: Load JD text (from rank.py)
    sys.path.insert(0, str(Path(__file__).parent))
    from rank import JOB_DESCRIPTION

    # Step 3: Compute candidate embeddings
    candidate_embeddings = compute_embeddings(
        candidate_texts, args.model, args.batch_size
    )

    # Step 4: Compute JD embedding
    jd_embedding = compute_embeddings([JOB_DESCRIPTION], args.model, 1)[0]

    # Step 5: Compute cosine similarities
    similarities = compute_similarities(candidate_embeddings, jd_embedding)

    logger.info(
        f"Similarity stats: min={similarities.min():.4f}, "
        f"max={similarities.max():.4f}, "
        f"mean={similarities.mean():.4f}"
    )

    # Step 6: Save results
    save_results(
        candidate_ids, candidate_embeddings, jd_embedding,
        similarities, args.output_dir,
    )

    logger.info("=" * 60)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"To use semantic scoring, ensure USE_HYBRID_SEMANTIC=true in config.py "
        f"or set environment variable USE_HYBRID_SEMANTIC=true"
    )


if __name__ == "__main__":
    main()
