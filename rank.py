"""
rank.py - Redrob Hackathon: Intelligent Candidate Ranking System
Main orchestrator. Runs end-to-end in under 5 minutes on CPU, 16 GB RAM.
Zero network calls during ranking.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv --dry-run
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv --explain CAND_0000031

Design philosophy:
    Trust calibration over keyword matching.
    Career history descriptions are evidence. Skills lists are hypotheses.
    Behavioral signals measure reachability. Honeypot signals mean ignore entirely.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

from loader import CandidateLoader
from jd_analyzer import JDAnalyzer
from honeypot import HoneypotDetector
from feature_extractor import FeatureExtractor
from trap_detector import TrapDetector
from scorer import ScoringEngine
from ranker import Ranker
from reasoning import ReasoningGenerator
from writer import SubmissionWriter
from validator import SubmissionValidator

# New modules (feature-flagged, gracefully degrade if disabled)
from config import Config
from bm25_scorer import BM25Scorer
from semantic_scorer import SemanticScorer
from interaction_engine import InteractionEngine
from top100_verifier import Top100Verifier

# Enhanced pipeline modules (graceful degradation - failures don't break pipeline)
from evidence_consistency_engine import EvidenceConsistencyEngine
from context_evidence_analyzer import ContextEvidenceAnalyzer
from corroborated_skill_scorer import CorroboratedSkillScorer
from confidence_estimator import ConfidenceEstimator
from score_calibrator import ScoreCalibrator

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rank")


# ─────────────────────────────────────────────────────────────────────────────
# JD - embedded directly, no file I/O, no network call
# ─────────────────────────────────────────────────────────────────────────────

JOB_DESCRIPTION = """
Role: Senior AI Engineer - Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India. Hybrid. Open to relocation from Tier-1 Indian cities.
Experience: 5-9 years

MUST HAVE:
Production experience with embeddings-based retrieval systems including sentence-transformers,
BGE, E5, OpenAI embeddings deployed to real users. Not research, not tutorials.
Production deployment with embedding drift, index refresh, retrieval-quality regression.
Production experience with vector databases and hybrid search infrastructure including
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar with
operational experience not just API calls.
Strong Python code quality.
Hands-on experience designing evaluation frameworks for ranking systems including NDCG,
MRR, MAP, offline-to-online correlation, A/B testing.

NICE TO HAVE:
LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank models XGBoost-based or neural LTR.
Prior HR-tech recruiting-tech marketplace exposure.
Distributed systems or large-scale inference optimization.
Open-source contributions in AI/ML.

DISQUALIFIERS:
Pure research background no production deployments.
AI experience consists entirely of LangChain tutorials calling OpenAI with no pre-LLM
era ML production experience.
Senior engineer who has not written production code in 18 months moved to architecture only.
Entire career at consulting firms TCS Infosys Wipro Accenture Cognizant Capgemini.
Primary expertise in CV speech or robotics with no NLP or IR exposure.
Title-chaser pattern Senior to Staff to Principal by switching companies every 1.5 years.

IMPLICIT:
Product company experience not services company.
Ability to ship fast and think deeply.
Located in or willing to relocate to Noida or Pune India.
Notice period ideally under 30 days accept up to 90 days higher bar above 90.
Active on job market responsive to recruiter outreach.
Plans to stay 3 years minimum.

CONTEXT:
The right answer is not finding candidates whose skills section contains the most AI keywords.
A candidate who built a recommendation system at a product company is a fit even without
mentioning RAG or Pinecone. A Marketing Manager with all AI keywords listed is not a fit.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon Candidate Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl.gz or candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for submission.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 1000 candidates for fast testing",
    )
    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        metavar="CANDIDATE_ID",
        help="Print detailed score breakdown for a specific candidate ID",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    pipeline_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("REDROB HACKATHON - Candidate Ranking System")
    logger.info("=" * 70)
    if args.dry_run:
        logger.warning("DRY RUN MODE - processing first 1000 candidates only")

    # ── Step 1: Analyze JD ──────────────────────────────────────────────────
    logger.info("Step 1/10: Analyzing job description...")
    t = time.perf_counter()
    jd_analyzer = JDAnalyzer(JOB_DESCRIPTION)
    jd_context = jd_analyzer.analyze()
    logger.info(
        f"  JD analyzed in {time.perf_counter() - t:.2f}s | "
        f"{len(jd_context.must_have_terms)} must-have terms | "
        f"{len(jd_context.nice_to_have_terms)} nice-to-have terms"
    )

    # Log active feature flags
    logger.info(Config.summary())

    # ── Step 2: Load candidates ─────────────────────────────────────────────
    logger.info("Step 2/10: Loading candidates...")
    t = time.perf_counter()
    loader = CandidateLoader(args.candidates, limit=1000 if args.dry_run else None)
    candidates = loader.load()
    logger.info(
        f"  Loaded {len(candidates):,} candidates in {time.perf_counter() - t:.2f}s"
    )

    # ── Step 3: Honeypot detection ──────────────────────────────────────────
    logger.info("Step 3/10: Running honeypot detection...")
    t = time.perf_counter()
    honeypot_detector = HoneypotDetector()
    honeypot_scores = honeypot_detector.detect_all(candidates)
    flagged = sum(1 for v in honeypot_scores.values() if v >= 2)
    logger.info(
        f"  Honeypot detection done in {time.perf_counter() - t:.2f}s | "
        f"{flagged:,} candidates flagged (2+ signals)"
    )

    # ── Step 4: Trap detection ──────────────────────────────────────────────
    logger.info("Step 4/10: Running trap detection...")
    t = time.perf_counter()
    trap_detector = TrapDetector()
    trap_flags = trap_detector.detect_all(candidates)
    logger.info(f"  Trap detection done in {time.perf_counter() - t:.2f}s")
    for trap_name in ["consulting_only", "langchain_only", "pure_researcher",
                      "behavioral_dead", "keyword_stuffer", "recent_hype_pivot",
                      "title_chaser", "domain_mismatch",
                      "certification_padder", "experience_gap"]:
        count = sum(1 for v in trap_flags.values() if v.get(trap_name, False))
        if count > 0:
            logger.info(f"    {trap_name}: {count:,} candidates")

    # ── Step 5: Feature extraction ──────────────────────────────────────────
    logger.info("Step 5/10: Extracting features...")
    t = time.perf_counter()
    feature_extractor = FeatureExtractor(jd_context)
    # Returns {candidate_id: FeatureBundle}
    feature_bundles = feature_extractor.extract_all(candidates)
    logger.info(
        f"  Feature extraction done in {time.perf_counter() - t:.2f}s"
    )

    # ── Step 5b: BM25 scoring (if enabled) ─────────────────────────────────
    bm25_scores: Dict[str, float] = {}
    if Config.USE_NEW_BM25:
        logger.info("Step 5b/10: Computing BM25 scores...")
        t = time.perf_counter()
        try:
            bm25 = BM25Scorer(jd_context)
            descriptions = [
                " ".join(
                    (r.get("description") or "") for r in c.get("career_history", [])
                )
                for c in candidates
            ]
            bm25.build_index(descriptions)
            logger.info(f"  BM25 index built in {time.perf_counter() - t:.2f}s")
            for c in candidates:
                cid = c["candidate_id"]
                score, terms, ret_terms, ml_terms = bm25.score_candidate(c.get("career_history", []))
                bm25_scores[cid] = score
                bundle = feature_bundles.get(cid)
                if bundle is not None:
                    bundle.score_A = score
                    bundle.career_terms_found = terms
                    bundle.retrieval_terms_found = ret_terms
                    bundle.ml_terms_found = ml_terms
            logger.info(
                f"  BM25 scores computed for {len(bm25_scores)} candidates"
            )
        except Exception as e:
            logger.warning(f"BM25 scoring failed: {e} - using original scores")
    else:
        logger.info("Step 5b/10: BM25 scoring disabled (USE_NEW_BM25=False)")

    # ── Step 5c: Semantic scoring (if enabled) ─────────────────────────────
    semantic_scores: Dict[str, float] = {}
    if Config.USE_HYBRID_SEMANTIC:
        logger.info("Step 5c/10: Loading semantic embeddings...")
        t = time.perf_counter()
        try:
            sem = SemanticScorer()
            sem.load()
            if sem.is_available:
                logger.info(f"  Embeddings loaded in {time.perf_counter() - t:.2f}s | {sem.stats()}")
                for cid in feature_bundles:
                    lexical = bm25_scores.get(cid, 0.0)
                    hybrid = sem.hybrid_score(candidate_id=cid, lexical_score=lexical)
                    semantic_scores[cid] = hybrid
                    bundle = feature_bundles.get(cid)
                    if bundle is not None:
                        bundle.score_A = hybrid
                logger.info(
                    f"  Semantic scores blended for {len(semantic_scores)} candidates"
                )
            else:
                logger.warning("  Embeddings not available - skipping hybrid scoring")
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e} - skipping")
    else:
        logger.info("Step 5c/10: Semantic scoring disabled (USE_HYBRID_SEMANTIC=False)")

    # ── Step 5d: Interaction engine (deferred to enhanced pipeline for speed) ──
    if Config.USE_INTERACTION_ENGINE:
        logger.info("Step 5d/10: Interaction engine enabled (will run on top candidates in enhanced pipeline)")
    else:
        logger.info("Step 5d/10: Interaction engine disabled (USE_INTERACTION_ENGINE=False)")

    # ── Step 6: Scoring ─────────────────────────────────────────────────────
    logger.info("Step 6/10: Computing final scores...")
    t = time.perf_counter()
    scoring_engine = ScoringEngine()
    scored_candidates = scoring_engine.score_all(
        candidates=candidates,
        feature_bundles=feature_bundles,
        honeypot_scores=honeypot_scores,
        trap_flags=trap_flags,
    )
    logger.info(
        f"  Scoring done in {time.perf_counter() - t:.2f}s | "
        f"Score range: [{min(s['final_score'] for s in scored_candidates.values()):.4f}, "
        f"{max(s['final_score'] for s in scored_candidates.values()):.4f}]"
    )

    # ── Step 6a-6c: Enhanced pipeline (evidence, corroboration, confidence, calibration) ──
    logger.info("Step 6a-6c/10: Running enhanced scoring modules...")
    t = time.perf_counter()
    try:
        scored_candidates, evidence_bundles, context_bundles, confidence_bundles = (
            run_enhanced_pipeline(
                candidates=candidates,
                feature_bundles=feature_bundles,
                scored_candidates=scored_candidates,
                jd_context=jd_context,
                honeypot_scores=honeypot_scores,
                trap_flags=trap_flags,
                scoring_engine=scoring_engine,
            )
        )
        logger.info(
            f"  Enhanced pipeline done in {time.perf_counter() - t:.2f}s | "
            f"Calibrated score range: ["
            f"{min(s.get('calibrated_score', 0) for s in scored_candidates.values()):.4f}, "
            f"{max(s.get('calibrated_score', 0) for s in scored_candidates.values()):.4f}]"
        )
    except Exception as e:
        logger.warning(f"Enhanced pipeline failed entirely: {e} - using original scores for ranking")
        evidence_bundles = {}
        context_bundles = {}
        confidence_bundles = {}
        for sd in scored_candidates.values():
            sd["calibrated_score"] = sd.get("final_score", 0.0)
            sd["confidence_adjusted_score"] = sd.get("final_score", 0.0)

    # ── Step 7: Explain mode (optional) ────────────────────────────────────
    if args.explain:
        cid = args.explain.upper().strip()
        _explain_candidate(
            cid, candidates, scored_candidates, feature_bundles,
            honeypot_scores, trap_flags, evidence_bundles, context_bundles,
            confidence_bundles
        )
        # In explain mode, still continue to generate submission

    # ── Step 8: Ranking ─────────────────────────────────────────────────────
    logger.info("Step 7/10: Ranking candidates and selecting top 100...")
    t = time.perf_counter()
    ranker = Ranker()
    top_100 = ranker.rank(scored_candidates, candidates, honeypot_scores)
    last_entry = top_100[-1]
    logger.info(
        f"  Ranking done in {time.perf_counter() - t:.2f}s | "
        f"Top calibrated score: {top_100[0].get('calibrated_score', top_100[0]['final_score']):.4f} | "
        f"Rank-{len(top_100)} calibrated score: {last_entry.get('calibrated_score', last_entry['final_score']):.4f}"
    )

    # Log top 10 for manual sanity check
    logger.info("  TOP 10 CANDIDATES (sanity check):")
    for entry in top_100[:10]:
        cid = entry["candidate_id"]
        cand = next(c for c in candidates if c["candidate_id"] == cid)
        profile = cand["profile"]
        display_score = entry.get("calibrated_score", entry["final_score"])
        logger.info(
            f"    Rank {entry['rank']:3d} | {cid} | "
            f"{profile['current_title'][:40]:<40} | "
            f"{profile['current_company'][:25]:<25} | "
            f"Score: {display_score:.4f}"
        )

    # ── Top-100 Verification (if enabled) ─────────────────────────────────
    if Config.USE_TOP100_VERIFICATION:
        logger.info("Step 7b/10: Running top-100 pre-export verification...")
        t = time.perf_counter()
        try:
            verifier = Top100Verifier()
            verify_report = verifier.verify(top_100, candidates, honeypot_scores)
            if verify_report.passed:
                logger.info(
                    f"  Top-100 verification PASSED in {time.perf_counter() - t:.2f}s"
                )
            else:
                logger.warning(
                    f"  Top-100 verification found issues:"
                )
                for err in verify_report.errors:
                    logger.warning(f"    - ERROR: {err}")

            # Act on demotions: remove honeypot candidates and backfill
            if verify_report.demotions:
                demote_set = set(verify_report.demotions)
                clean = [e for e in top_100 if e["candidate_id"] not in demote_set]
                demoted = [e for e in top_100 if e["candidate_id"] in demote_set]
                if len(clean) < 100:
                    # Backfill from the full ranked list
                    already = {e["candidate_id"] for e in clean}
                    sortable_full = sorted(
                        scored_candidates.items(),
                        key=lambda x: -(x[1].get("calibrated_score")
                                        or x[1].get("confidence_adjusted_score")
                                        or x[1].get("final_score", 0.0))
                    )
                    for cid, sd in sortable_full:
                        if len(clean) >= 100:
                            break
                        if cid not in already and honeypot_scores.get(cid, 0) == 0:
                            score = (sd.get("calibrated_score")
                                     or sd.get("confidence_adjusted_score")
                                     or sd.get("final_score", 0.0))
                            clean.append({
                                "candidate_id": cid,
                                "final_score": score,
                                "score_dict": sd,
                            })
                            already.add(cid)
                    # Re-rank
                    for rank_idx, entry in enumerate(clean, start=1):
                        entry["rank"] = rank_idx
                    top_100 = clean
                    logger.info(
                        f"  Demoted {len(demoted)} honeypot candidates, "
                        f"backfilled to {len(top_100)}"
                    )
            for warn in verify_report.warnings:
                logger.warning(f"    - WARNING: {warn}")
        except Exception as e:
            logger.warning(f"Top-100 verification failed: {e} - skipping")

    # ── Step 8: Reasoning generation ────────────────────────────────────────
    logger.info("Step 8/10: Generating candidate-specific reasoning...")
    t = time.perf_counter()
    reasoning_generator = ReasoningGenerator(jd_context)
    top_100_with_reasoning = reasoning_generator.generate_all(
        top_100, candidates, feature_bundles, honeypot_scores, trap_flags,
        evidence_bundles=evidence_bundles,
        context_bundles=context_bundles,
        confidence_bundles=confidence_bundles,
    )
    logger.info(f"  Reasoning generation done in {time.perf_counter() - t:.2f}s")

    # ── Step 9: Write output ─────────────────────────────────────────────────
    logger.info("Step 9/10: Writing submission CSV...")
    t = time.perf_counter()
    writer = SubmissionWriter(args.out)
    writer.write(top_100_with_reasoning)
    logger.info(f"  Written to {args.out} in {time.perf_counter() - t:.2f}s")

    # ── Final validation ─────────────────────────────────────────────────────
    logger.info("Running internal submission validation...")
    validator = SubmissionValidator(args.out)
    validation_result = validator.validate()
    if validation_result.passed:
        logger.info(f"  Validation PASSED - {args.out}")
    else:
        # For small samples (<100 candidates), the strict 100-row check is
        # expected to fail. Log as warnings and continue — the CSV is valid
        # for sandbox/reproducibility checks. Full dataset validates strictly.
        if len(candidates) < 100:
            for err in validation_result.errors:
                logger.warning(f"    - {err}")
            logger.warning(
                "  Validation noted issues (expected for small sample <100) "
                "- submission CSV produced successfully"
            )
        else:
            logger.error(f"  Validation FAILED:")
            for err in validation_result.errors:
                logger.error(f"    - {err}")
            sys.exit(1)

    total_time = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE - Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    if total_time > 280:
        logger.warning("Pipeline exceeded 280s - close to the 300s limit")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Explain mode helper
# ─────────────────────────────────────────────────────────────────────────────

def _explain_candidate(
    candidate_id: str,
    candidates: list,
    scored_candidates: dict,
    feature_bundles: dict,
    honeypot_scores: dict,
    trap_flags: dict,
    evidence_bundles: dict | None = None,
    context_bundles: dict | None = None,
    confidence_bundles: dict | None = None,
) -> None:
    """Print a detailed score breakdown for a specific candidate."""
    cand = next((c for c in candidates if c["candidate_id"] == candidate_id), None)
    if cand is None:
        logger.error(f"Candidate {candidate_id} not found in dataset")
        return

    profile = cand["profile"]
    signals = cand["redrob_signals"]
    scores = scored_candidates.get(candidate_id, {})
    bundle = feature_bundles.get(candidate_id)
    hp_score = honeypot_scores.get(candidate_id, 0)
    traps = trap_flags.get(candidate_id, {})

    print("\n" + "=" * 70)
    print(f"CANDIDATE EXPLANATION: {candidate_id}")
    print("=" * 70)
    print(f"Name:          {profile['anonymized_name']}")
    print(f"Title:         {profile['current_title']}")
    print(f"Company:       {profile['current_company']}")
    print(f"YoE:           {profile['years_of_experience']}")
    print(f"Location:      {profile['location']}, {profile['country']}")
    print()
    print("SCORE BREAKDOWN:")
    print(f"  A. Core Technical Relevance  (35%): {scores.get('A', 0):.4f}")
    print(f"  B. Production ML Depth       (20%): {scores.get('B', 0):.4f}")
    print(f"  C. Assessment Match          (12%): {scores.get('C', 0):.4f}")
    print(f"  D. Career Quality            (13%): {scores.get('D', 0):.4f}")
    print(f"  E. Behavioral Availability   (10%): {scores.get('E', 0):.4f}")
    print(f"  F. Location & Logistics       (5%): {scores.get('F', 0):.4f}")
    print(f"  G. Education Signal           (3%): {scores.get('G', 0):.4f}")
    print(f"  H. GitHub Activity            (2%): {scores.get('H', 0):.4f}")
    print(f"  RAW WEIGHTED SCORE:               {scores.get('raw_score', 0):.4f}")
    print()
    if evidence_bundles or context_bundles or confidence_bundles:
        print("ENHANCED SCORES:")
        print(f"  C_corroborated:               {scores.get('C_corroborated', 'N/A')}")
        print(f"  C_blended:                    {scores.get('C_blended', 'N/A')}")
        print(f"  A_adjusted:                   {scores.get('A_adjusted', 'N/A')}")
        print(f"  B_adjusted:                   {scores.get('B_adjusted', 'N/A')}")
        print(f"  confidence_adjusted_score:    {scores.get('confidence_adjusted_score', 'N/A')}")
        print(f"  calibrated_score:             {scores.get('calibrated_score', 'N/A')}")
        print()

    print("MULTIPLIERS:")
    print(f"  Honeypot signals detected:    {hp_score}")
    print(f"  Consulting-only:              {traps.get('consulting_only', False)}")
    print(f"  LangChain-only:               {traps.get('langchain_only', False)}")
    print(f"  Pure researcher:              {traps.get('pure_researcher', False)}")
    print(f"  Behavioral dead:              {traps.get('behavioral_dead', False)}")
    print(f"  Keyword stuffer:              {traps.get('keyword_stuffer', False)}")
    print(f"  Recent hype pivot:            {traps.get('recent_hype_pivot', False)}")
    print(f"  FINAL SCORE:                  {scores.get('final_score', 0):.4f}")
    print()
    print("CAREER HISTORY:")
    for i, role in enumerate(cand["career_history"]):
        print(f"  [{i+1}] {role['title']} @ {role['company']}")
        print(f"       {role['start_date']} -> {role.get('end_date', 'Present')} "
              f"({role['duration_months']} months)")
        desc_preview = role['description'][:120].replace('\n', ' ')
        print(f"       {desc_preview}...")
    print()
    print("KEY SIGNALS:")
    print(f"  last_active_date:             {signals['last_active_date']}")
    print(f"  open_to_work_flag:            {signals['open_to_work_flag']}")
    print(f"  recruiter_response_rate:      {signals['recruiter_response_rate']}")
    print(f"  notice_period_days:           {signals['notice_period_days']}")
    print(f"  github_activity_score:        {signals['github_activity_score']}")
    print(f"  willing_to_relocate:          {signals['willing_to_relocate']}")
    relevant_assessments = {
        k: v for k, v in signals["skill_assessment_scores"].items()
    }
    if relevant_assessments:
        print("  skill_assessment_scores:")
        for skill, score in sorted(
            relevant_assessments.items(), key=lambda x: -x[1]
        )[:5]:
            print(f"    {skill}: {score:.1f}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced pipeline functions
# ─────────────────────────────────────────────────────────────────────────────

def run_enhanced_pipeline(
    candidates: list,
    feature_bundles: dict,
    scored_candidates: dict,
    jd_context,
    honeypot_scores: dict,
    trap_flags: dict,
    scoring_engine,
) -> dict:
    """
    Run the new enhancement modules and return an updated scored_candidates dict.

    This function is called from rank.py between feature extraction and ranking.
    It wraps all new module calls with graceful degradation.

    Args:
        candidates:        Original candidate list from loader
        feature_bundles:   {candidate_id: FeatureBundle} from FeatureExtractor
        scored_candidates: {candidate_id: score_dict} from ScoringEngine
        jd_context:        JDContext from JDAnalyzer
        honeypot_scores:   {candidate_id: int} from HoneypotDetector
        trap_flags:        {candidate_id: dict} from TrapDetector
        scoring_engine:    Existing ScoringEngine instance

    Returns:
        Updated scored_candidates with calibrated_score and confidence_adjusted_score
        added to each score_dict.
    """
    logger_enh = logging.getLogger("rank.enhanced_pipeline")

    # ── Performance optimization: only analyze top candidates ─────────────────
    # Running evidence/context modules on all 100K candidates exceeds 300s.
    # We restrict to the top N by base final_score - well above the top-100
    # output. Remaining candidates keep base scores unchanged.
    ENHANCED_CANDIDATE_LIMIT = Config.ENHANCED_CANDIDATE_LIMIT
    sorted_by_score = sorted(
        scored_candidates.items(), key=lambda x: -x[1].get("final_score", 0.0)
    )
    top_cids = {cid for cid, _ in sorted_by_score[:ENHANCED_CANDIDATE_LIMIT]}
    enhanced_candidates = [c for c in candidates if c["candidate_id"] in top_cids]
    logger_enh.info(
        f"Running enhanced analysis on top {len(enhanced_candidates)} candidates "
        f"(by base score) out of {len(candidates)} total"
    )

    # ── Step 5d-enhanced: Interaction Engine on top candidates only ────────
    if Config.USE_INTERACTION_ENGINE:
        try:
            logger_enh.info("Step 5d: Running interaction engine on top candidates...")
            t = time.perf_counter()
            interaction_results = InteractionEngine(jd_context).analyze_all(
                enhanced_candidates, feature_bundles
            )
            for cid, result in interaction_results.items():
                net = result.net_adjustment
                bundle = feature_bundles.get(cid)
                if bundle is not None:
                    bundle.score_A = max(0.0, min(1.0, bundle.score_A + net))
            triggered = sum(
                1 for r in interaction_results.values() if r.triggered_interactions
            )
            logger_enh.info(
                f"  Interaction engine done in {time.perf_counter() - t:.2f}s | "
                f"{triggered} candidates with triggered interactions"
            )
            # Re-score the enhanced candidates with updated score_A
            for cid in top_cids:
                bundle = feature_bundles.get(cid)
                if bundle is None:
                    continue
                hp_count = honeypot_scores.get(cid, 0)
                traps = trap_flags.get(cid, {})
                try:
                    new_sd = scoring_engine._score_candidate(
                        bundle=bundle, honeypot_count=hp_count, trap_flags=traps
                    )
                    scored_candidates[cid] = new_sd
                except Exception:
                    pass
        except Exception as e:
            logger_enh.warning(f"Interaction engine failed: {e} - skipping")

    # ── Step 5a: Evidence Consistency Analysis ───────────────────────────────
    evidence_bundles = {}
    try:
        logger_enh.info("Step 5a/3: Running evidence consistency analysis...")
        t = time.perf_counter()
        ece = EvidenceConsistencyEngine(jd_context)
        evidence_bundles = ece.analyze_all(enhanced_candidates)
        logger_enh.info(
            f"  Evidence consistency done in {time.perf_counter() - t:.2f}s"
        )
    except Exception as e:
        logger_enh.warning(
            f"Evidence consistency analysis failed: {e} - continuing without it"
        )

    # ── Step 5b: Context Evidence Analysis ───────────────────────────────────
    context_bundles = {}
    try:
        logger_enh.info("Step 5b/3: Running context evidence analysis...")
        t = time.perf_counter()
        cea = ContextEvidenceAnalyzer(jd_context)
        context_bundles = cea.analyze_all(enhanced_candidates)
        logger_enh.info(
            f"  Context evidence analysis done in {time.perf_counter() - t:.2f}s"
        )
    except Exception as e:
        logger_enh.warning(
            f"Context evidence analysis failed: {e} - continuing without it"
        )

    # ── Step 5c: Corroborated Skill Scoring ──────────────────────────────────
    corroborated_C_scores = {}
    try:
        logger_enh.info("Step 5c/3: Computing corroborated skill scores...")
        t = time.perf_counter()
        css = CorroboratedSkillScorer(jd_context)
        corroborated_C_scores = css.compute_all(
            feature_bundles=feature_bundles,
            evidence_bundles=evidence_bundles,
            context_bundles=context_bundles,
            candidates=enhanced_candidates,
        )
        # Inject corroborated scores into scored_candidates
        for cid, c_score in corroborated_C_scores.items():
            if cid in scored_candidates:
                old_C = scored_candidates[cid].get("C", 0.0)
                scored_candidates[cid]["C_corroborated"] = c_score
                # Blend corroborated score into the raw C for final_score recomputation
                # Weight: 60% corroborated, 40% raw (conservative blend)
                blended_C = old_C * 0.40 + c_score * 0.60
                scored_candidates[cid]["C_blended"] = blended_C
                # Recompute final_score with blended C
                _recompute_final_score_with_evidence(
                    scored_candidates[cid],
                    evidence_bundles.get(cid),
                    context_bundles.get(cid),
                    blended_C,
                )
        logger_enh.info(
            f"  Corroborated skill scoring done in {time.perf_counter() - t:.2f}s"
        )
    except Exception as e:
        logger_enh.warning(
            f"Corroborated skill scoring failed: {e} - continuing without it"
        )

    # ── Step 6a: Confidence Estimation ───────────────────────────────────────
    confidence_bundles = {}
    try:
        logger_enh.info("Step 6a/3: Estimating score confidence...")
        t = time.perf_counter()
        ce = ConfidenceEstimator()
        confidence_bundles = ce.estimate_all(
            candidates=enhanced_candidates,
            feature_bundles=feature_bundles,
            evidence_bundles=evidence_bundles,
            scored_candidates=scored_candidates,
        )
        # Apply confidence adjustment to final scores
        ce.apply_confidence_adjustment(scored_candidates, confidence_bundles)
        logger_enh.info(
            f"  Confidence estimation done in {time.perf_counter() - t:.2f}s"
        )
    except Exception as e:
        logger_enh.warning(
            f"Confidence estimation failed: {e} - using final_score as confidence_adjusted_score"
        )
        # Graceful degradation: copy final_score to confidence_adjusted_score
        for cid, sd in scored_candidates.items():
            sd["confidence_adjusted_score"] = sd.get("final_score", 0.0)

    # ── Step 6b: Score Calibration ────────────────────────────────────────────
    try:
        logger_enh.info("Step 6b/3: Calibrating score distribution...")
        t = time.perf_counter()
        calibrator = ScoreCalibrator()
        calibrator.calibrate(scored_candidates, score_key="confidence_adjusted_score")
        logger_enh.info(
            f"  Score calibration done in {time.perf_counter() - t:.2f}s"
        )
    except Exception as e:
        logger_enh.warning(
            f"Score calibration failed: {e} - using confidence_adjusted_score as calibrated_score"
        )
        for cid, sd in scored_candidates.items():
            sd["calibrated_score"] = sd.get("confidence_adjusted_score", sd.get("final_score", 0.0))

    return scored_candidates, evidence_bundles, context_bundles, confidence_bundles


def _recompute_final_score_with_evidence(
    score_dict: dict,
    evidence_bundle,
    context_bundle,
    blended_C: float,
) -> None:
    """
    Recompute final_score incorporating evidence bundle adjustments.

    This is a localized recompute - it adjusts score_A, score_B, and score_C
    using the evidence and context adjustments, then recomputes the weighted sum.
    The multipliers from the original computation are preserved.
    """
    # Read existing component scores
    A = score_dict.get("A", 0.0)
    B = score_dict.get("B", 0.0)
    D = score_dict.get("D", 0.0)
    E = score_dict.get("E", 0.0)
    F = score_dict.get("F", 0.0)
    G = score_dict.get("G", 0.0)
    H = score_dict.get("H", 0.0)

    # Apply evidence adjustments to A and B
    A_adj = 0.0
    B_adj = 0.0
    context_adj = 0.0

    if evidence_bundle is not None:
        A_adj = evidence_bundle.score_A_adjustment
        B_adj = evidence_bundle.score_B_adjustment

    if context_bundle is not None:
        context_adj = context_bundle.context_score_adjustment

    # Clamp adjustments to prevent runaway values
    A_adjusted = max(0.0, min(1.0, A + A_adj + context_adj))
    B_adjusted = max(0.0, min(1.0, B + B_adj))
    C_adjusted = blended_C

    # Recompute weighted raw score
    raw_score = (
        A_adjusted * 0.35
        + B_adjusted * 0.20
        + C_adjusted * 0.12
        + D * 0.13
        + F * 0.05
        + G * 0.03
        + H * 0.02
    ) / 0.90  # Normalize for excluded E (10% weight)

    # Apply behavioral soft multiplier (preserved from original scoring)
    behavioral_multiplier = score_dict.get("behavioral_multiplier", 1.0)
    raw_with_behavior = raw_score * behavioral_multiplier

    # Apply hard multiplier (preserved from original scoring)
    hard_multiplier = score_dict.get("multiplier", 1.0)
    new_final = min(1.0, raw_with_behavior * hard_multiplier)

    # Update scores
    score_dict["A_adjusted"] = A_adjusted
    score_dict["B_adjusted"] = B_adjusted
    score_dict["C_blended"] = C_adjusted
    score_dict["final_score"] = new_final


if __name__ == "__main__":
    main()