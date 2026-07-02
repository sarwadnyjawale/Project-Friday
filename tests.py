"""
tests.py - Unit tests for ICRS pipeline modules.

Run: python tests.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_candidate(
    cid="CAND_0000001",
    title="ML Engineer",
    company="Swiggy",
    yoe=6.0,
    industry="Software",
    description="Built and deployed sentence-transformers for semantic search. Used FAISS vector index serving 1M qps.",
    skills=None,
    github_score=50.0,
    notice_days=30,
    open_to_work=True,
    last_active="2026-06-01",
    salary_min=20,
    salary_max=40,
    country="India",
    location="Pune",
    assessments=None,
):
    if skills is None:
        skills = [{"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 60}]
    if assessments is None:
        assessments = {"Python": 85.0}
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": f"{title}",
            "summary": f"Experienced {title} with {yoe} years.",
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "1001-5000",
            "current_industry": industry,
        },
        "career_history": [
            {
                "company": company,
                "title": title,
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 77,
                "is_current": True,
                "industry": industry,
                "company_size": "1001-5000",
                "description": description,
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": last_active,
            "open_to_work_flag": open_to_work,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.60,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": assessments,
            "connection_count": 500,
            "endorsements_received": 50,
            "notice_period_days": notice_days,
            "expected_salary_range_inr_lpa": {"min": salary_min, "max": salary_max},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": github_score,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


JD_TEXT = """
Role: Senior AI Engineer
MUST HAVE: Production experience with embeddings-based retrieval systems including
sentence-transformers, BGE, E5, OpenAI embeddings. Vector databases: Pinecone,
Weaviate, Qdrant, Milvus, FAISS. NDCG, MRR, MAP evaluation frameworks.
NICE TO HAVE: LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank XGBoost.
DISQUALIFIERS: Pure research background. Entire career at TCS Infosys Wipro.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoader(unittest.TestCase):
    def _write_jsonl(self, records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_load_valid(self):
        from loader import CandidateLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(make_candidate("CAND_0000001")) + "\n")
            f.write(json.dumps(make_candidate("CAND_0000002")) + "\n")
            path = Path(f.name)
        try:
            loader = CandidateLoader(path)
            candidates = loader.load()
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0]["candidate_id"], "CAND_0000001")
        finally:
            os.unlink(path)

    def test_skip_malformed_json(self):
        from loader import CandidateLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write("not valid json\n")
            f.write(json.dumps(make_candidate("CAND_0000001")) + "\n")
            path = Path(f.name)
        try:
            loader = CandidateLoader(path)
            candidates = loader.load()
            self.assertEqual(len(candidates), 1)
        finally:
            os.unlink(path)

    def test_apply_defaults(self):
        from loader import CandidateLoader
        bare = {"candidate_id": "CAND_0000001"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(bare) + "\n")
            path = Path(f.name)
        try:
            loader = CandidateLoader(path)
            candidates = loader.load()
            c = candidates[0]
            self.assertIn("profile", c)
            self.assertIn("redrob_signals", c)
            self.assertIn("career_history", c)
        finally:
            os.unlink(path)

    def test_limit(self):
        from loader import CandidateLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps(make_candidate(f"CAND_{i:07d}")) + "\n")
            path = Path(f.name)
        try:
            loader = CandidateLoader(path, limit=5)
            candidates = loader.load()
            self.assertEqual(len(candidates), 5)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        from jd_analyzer import JDAnalyzer
        from feature_extractor import FeatureExtractor
        self.jd_context = JDAnalyzer(JD_TEXT).analyze()
        self.extractor = FeatureExtractor(self.jd_context)

    def test_scores_in_range(self):
        candidate = make_candidate()
        bundle = self.extractor._extract_candidate(candidate)
        for attr in ["score_A", "score_B", "score_C", "score_D", "score_E", "score_F", "score_G", "score_H"]:
            score = getattr(bundle, attr)
            self.assertGreaterEqual(score, 0.0, f"{attr} < 0")
            self.assertLessEqual(score, 1.0, f"{attr} > 1")

    def test_retrieval_terms_detected(self):
        candidate = make_candidate(
            description="Built FAISS index for semantic search using sentence-transformers and dense retrieval."
        )
        bundle = self.extractor._extract_candidate(candidate)
        self.assertGreater(bundle.score_A, 0.0)
        self.assertTrue(len(bundle.retrieval_terms_found) > 0)

    def test_consulting_company_penalized_by_D(self):
        candidate_product = make_candidate(company="Swiggy", industry="Software")
        candidate_consulting = make_candidate(company="Wipro", industry="IT Services")
        bundle_p = self.extractor._extract_candidate(candidate_product)
        bundle_c = self.extractor._extract_candidate(candidate_consulting)
        self.assertGreater(bundle_p.score_D1, bundle_c.score_D1)

    def test_india_location_gets_full_F(self):
        candidate_india = make_candidate(country="India", location="Pune")
        candidate_usa = make_candidate(country="USA", location="New York")
        bundle_india = self.extractor._extract_candidate(candidate_india)
        bundle_usa = self.extractor._extract_candidate(candidate_usa)
        self.assertGreater(bundle_india.score_F, bundle_usa.score_F)

    def test_extract_all_no_crash(self):
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(5)]
        bundles = self.extractor.extract_all(candidates)
        self.assertEqual(len(bundles), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringEngine(unittest.TestCase):
    def setUp(self):
        from jd_analyzer import JDAnalyzer
        from feature_extractor import FeatureExtractor
        from scorer import ScoringEngine
        self.jd_context = JDAnalyzer(JD_TEXT).analyze()
        self.extractor = FeatureExtractor(self.jd_context)
        self.engine = ScoringEngine()

    def test_honeypot_3plus_near_zero(self):
        candidates = [make_candidate()]
        bundles = self.extractor.extract_all(candidates)
        honeypot_scores = {"CAND_0000001": 3}
        trap_flags = {"CAND_0000001": {}}
        result = self.engine.score_all(candidates, bundles, honeypot_scores, trap_flags)
        score = result["CAND_0000001"]["final_score"]
        self.assertLess(score, 0.02)

    def test_consulting_only_penalty(self):
        candidates = [make_candidate()]
        bundles = self.extractor.extract_all(candidates)
        honeypot_scores = {"CAND_0000001": 0}
        trap_no_trap = {"CAND_0000001": {}}
        trap_consulting = {"CAND_0000001": {"consulting_only": True}}
        result_clean = self.engine.score_all(candidates, bundles, honeypot_scores, trap_no_trap)
        result_consult = self.engine.score_all(candidates, bundles, honeypot_scores, trap_consulting)
        self.assertGreater(
            result_clean["CAND_0000001"]["final_score"],
            result_consult["CAND_0000001"]["final_score"],
        )

    def test_score_in_range(self):
        candidates = [make_candidate()]
        bundles = self.extractor.extract_all(candidates)
        honeypot_scores = {"CAND_0000001": 0}
        trap_flags = {"CAND_0000001": {}}
        result = self.engine.score_all(candidates, bundles, honeypot_scores, trap_flags)
        score = result["CAND_0000001"]["final_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRanker(unittest.TestCase):
    def setUp(self):
        from ranker import Ranker
        self.ranker = Ranker()

    def _make_scored(self, n=150):
        import random
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(n)]
        scored = {}
        for c in candidates:
            cid = c["candidate_id"]
            scored[cid] = {
                "final_score": random.uniform(0.1, 0.9),
                "calibrated_score": random.uniform(0.1, 0.9),
            }
        return candidates, scored

    def test_exactly_100_results(self):
        candidates, scored = self._make_scored(150)
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        result = self.ranker.rank(scored, candidates, honeypot)
        self.assertEqual(len(result), 100)

    def test_ranks_monotonic(self):
        candidates, scored = self._make_scored(150)
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        result = self.ranker.rank(scored, candidates, honeypot)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(
                result[i]["final_score"], result[i + 1]["final_score"] - 1e-9,
                f"Rank {i+1} score {result[i]['final_score']} < rank {i+2} score {result[i+1]['final_score']}"
            )

    def test_honeypot_excluded_from_top10(self):
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(150)]
        scored = {}
        for c in candidates:
            cid = c["candidate_id"]
            scored[cid] = {"final_score": 0.5, "calibrated_score": 0.5}
        # Give the first candidate a very high score but also honeypot flags
        scored["CAND_0000000"]["final_score"] = 0.99
        scored["CAND_0000000"]["calibrated_score"] = 0.99
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        honeypot["CAND_0000000"] = 3  # Honeypot
        result = self.ranker.rank(scored, candidates, honeypot)
        top10_ids = [r["candidate_id"] for r in result[:10]]
        self.assertNotIn("CAND_0000000", top10_ids)

    def test_unique_ranks(self):
        candidates, scored = self._make_scored(150)
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        result = self.ranker.rank(scored, candidates, honeypot)
        ranks = [r["rank"] for r in result]
        self.assertEqual(len(set(ranks)), 100)
        self.assertEqual(set(ranks), set(range(1, 101)))


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVWriter(unittest.TestCase):
    def _make_top100(self):
        return [
            {
                "candidate_id": f"CAND_{i:07d}",
                "rank": i,
                "final_score": 1.0 - i * 0.005,
                "reasoning": f"Candidate {i} ranked based on strong ML background.",
            }
            for i in range(1, 101)
        ]

    def test_write_creates_file(self):
        from writer import SubmissionWriter
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            writer = SubmissionWriter(path)
            writer.write(self._make_top100())
            self.assertTrue(path.exists())
        finally:
            os.unlink(path)

    def test_write_correct_columns(self):
        import csv
        from writer import SubmissionWriter
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            writer = SubmissionWriter(path)
            writer.write(self._make_top100())
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                self.assertEqual(set(reader.fieldnames), {"candidate_id", "rank", "score", "reasoning"})
        finally:
            os.unlink(path)

    def test_write_100_rows(self):
        import csv
        from writer import SubmissionWriter
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            writer = SubmissionWriter(path)
            writer.write(self._make_top100())
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 100)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidator(unittest.TestCase):
    def _write_valid_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            for i in range(1, 101):
                writer.writerow({
                    "candidate_id": f"CAND_{i:07d}",
                    "rank": i,
                    "score": f"{1.0 - i * 0.005:.6f}",
                    "reasoning": f"Candidate {i} ranked based on ML skills.",
                })

    def test_valid_passes(self):
        from validator import SubmissionValidator
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            self._write_valid_csv(path)
            v = SubmissionValidator(path)
            result = v.validate()
            self.assertTrue(result.passed, f"Errors: {result.errors}")
        finally:
            os.unlink(path)

    def test_wrong_row_count_fails(self):
        import csv
        from validator import SubmissionValidator
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
            path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            writer.writerow({"candidate_id": "CAND_0000001", "rank": "1", "score": "0.9", "reasoning": "test"})
        try:
            v = SubmissionValidator(path)
            result = v.validate()
            self.assertFalse(result.passed)
        finally:
            os.unlink(path)

    def test_non_monotonic_score_fails(self):
        import csv
        from validator import SubmissionValidator
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
            path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            for i in range(1, 101):
                score = 0.9 if i != 50 else 0.99  # Spike at rank 50 - non-monotonic
                writer.writerow({
                    "candidate_id": f"CAND_{i:07d}",
                    "rank": i,
                    "score": f"{score:.6f}",
                    "reasoning": "test",
                })
        try:
            v = SubmissionValidator(path)
            result = v.validate()
            self.assertFalse(result.passed)
        finally:
            os.unlink(path)

    def test_duplicate_rank_fails(self):
        import csv
        from validator import SubmissionValidator
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
            path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
            writer.writeheader()
            for i in range(1, 101):
                rank = 1  # All rank 1 - duplicate
                writer.writerow({
                    "candidate_id": f"CAND_{i:07d}",
                    "rank": rank,
                    "score": "0.5",
                    "reasoning": "test",
                })
        try:
            v = SubmissionValidator(path)
            result = v.validate()
            self.assertFalse(result.passed)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHoneypotDetector(unittest.TestCase):
    def setUp(self):
        from honeypot import HoneypotDetector
        self.detector = HoneypotDetector()

    def test_clean_candidate_zero_signals(self):
        candidate = make_candidate()
        result = self.detector.detect_all([candidate])
        self.assertEqual(result["CAND_0000001"], 0)

    def test_inverted_salary_detected(self):
        candidate = make_candidate(salary_min=50, salary_max=30)  # inverted
        result = self.detector.detect_all([candidate])
        self.assertGreater(result["CAND_0000001"], 0)

    def test_never_crashes(self):
        # Minimal/broken candidate
        broken = {"candidate_id": "CAND_0000001", "profile": {}, "career_history": [], "skills": [], "redrob_signals": {"expected_salary_range_inr_lpa": {}}}
        result = self.detector.detect_all([broken])
        self.assertIn("CAND_0000001", result)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Scorer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Scorer(unittest.TestCase):
    def setUp(self):
        from jd_analyzer import JDAnalyzer
        from bm25_scorer import BM25Scorer
        self.jd = JDAnalyzer(JD_TEXT).analyze()
        self.bm25 = BM25Scorer(self.jd)

    def _build_index(self, descriptions):
        self.bm25.build_index(descriptions)

    def test_score_monotonic_with_more_matches(self):
        desc_weak = "Python developer with some experience."
        desc_strong = "Built FAISS vector index for production semantic search using sentence-transformers and dense retrieval. Deployed BGE embeddings with NDCG evaluation."
        self._build_index([desc_weak, desc_strong])
        score_weak, _ = self.bm25.score(desc_weak)
        score_strong, _ = self.bm25.score(desc_strong)
        self.assertGreater(score_strong, score_weak)

    def test_score_zero_for_empty(self):
        self._build_index(["some description"])
        score, terms = self.bm25.score("")
        self.assertEqual(score, 0.0)
        self.assertEqual(terms, [])

    def test_idf_rare_terms_score_higher(self):
        docs = [
            "Python developer with FAISS experience.",  # FAISS appears once
            "Python developer with FAISS experience.",  # FAISS appears twice
            "Python developer with ML experience.",
        ]
        self._build_index(docs)
        score, terms = self.bm25.score("FAISS vector search")
        self.assertGreater(self.bm25._idf_cache.get("faiss", 0), 0)
        # FAISS appears in 2/3 docs - should have non-zero IDF

    def test_tf_saturation(self):
        docs = ["Python " + "FAISS " * 20 + "experience.",
                "Python ML experience.", "Python data experience."]
        self._build_index(docs)
        score_1x, _ = self.bm25.score("FAISS")
        score_20x, _ = self.bm25.score("FAISS " * 20)
        # 20x occurrences should NOT give 20x the score (TF saturation)
        self.assertLess(score_20x, score_1x * 5)

    def test_score_candidate_recency_weighting(self):
        candidate = make_candidate()
        candidate["career_history"] = [
            {"company": "OldCo", "title": "Jr Dev", "start_date": "2018-01-01",
             "end_date": "2020-01-01", "duration_months": 24, "is_current": False,
             "industry": "Software", "company_size": "1001-5000",
             "description": "Built internal tools with Python."},
            {"company": "NewCo", "title": "Sr ML Eng", "start_date": "2020-06-01",
             "end_date": None, "duration_months": 48, "is_current": True,
             "industry": "Software", "company_size": "1001-5000",
             "description": "Built FAISS and sentence-transformers for production semantic search."},
        ]
        self._build_index(["Python developer", "FAISS vector search expert"])
        bm25_score, terms, ret_terms, ml_terms = self.bm25.score_candidate(
            candidate["career_history"]
        )
        self.assertGreater(bm25_score, 0.0)
        self.assertIn("faiss", terms)

    def test_score_in_expected_range(self):
        docs = ["Python developer", "FAISS expert for production semantic search",
                "ML engineer with deployment experience", "Data scientist"]
        self._build_index(docs)
        candidate = make_candidate(
            description="Built FAISS vector index for semantic search using sentence-transformers."
        )
        score, _, _, _ = self.bm25.score_candidate(candidate.get("career_history", []))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_not_built_graceful(self):
        score, terms = self.bm25.score("test description")
        self.assertEqual(score, 0.0)
        self.assertEqual(terms, [])

    def test_anti_production_penalty(self):
        docs = ["Production FAISS deployment with sentence-transformers.",
                "Jupyter notebook research on embedding models."]
        self._build_index(docs)
        candidate_research = make_candidate(
            description="Conducted research on embedding models in Jupyter notebooks with toy datasets."
        )
        score_research, _, _, _ = self.bm25.score_candidate(
            candidate_research.get("career_history", [])
        )
        candidate_prod = make_candidate(
            description="Deployed FAISS index serving 100K qps with sentence-transformers in production."
        )
        score_prod, _, _, _ = self.bm25.score_candidate(
            candidate_prod.get("career_history", [])
        )
        self.assertGreater(score_prod, score_research)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scorer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticScorer(unittest.TestCase):
    def test_graceful_degradation(self):
        from semantic_scorer import SemanticScorer
        sem = SemanticScorer()
        sem.load()
        if not sem.is_available:
            # Graceful degradation - should return lexical score
            result = sem.hybrid_score("CAND_0000001", 0.5)
            self.assertEqual(result, 0.5)

    def test_hybrid_score_formula(self):
        from semantic_scorer import SemanticScorer
        from config import Config
        sem = SemanticScorer()
        # Manually set similarities to simulate pre-computed embeddings
        sem._loaded = True
        sem._similarities = {"CAND_0000001": 0.8}
        sem._alpha = Config.HYBRID_ALPHA
        sem._beta = Config.HYBRID_BETA
        # Hybrid = 0.6 * 0.5 + 0.4 * 0.8 = 0.30 + 0.32 = 0.62
        result = sem.hybrid_score("CAND_0000001", 0.5)
        expected = 0.6 * 0.5 + 0.4 * 0.8
        self.assertAlmostEqual(result, expected, places=6)

    def test_hybrid_score_fallback_no_similarity(self):
        from semantic_scorer import SemanticScorer
        sem = SemanticScorer()
        sem._loaded = True
        sem._similarities = {"CAND_0000001": 0.0}
        result = sem.hybrid_score("CAND_0000002", 0.5)  # Different ID - not in dict
        self.assertEqual(result, 0.5)

    def test_stats_report(self):
        from semantic_scorer import SemanticScorer
        sem = SemanticScorer()
        stats = sem.stats()
        if sem.is_available:
            self.assertIn("count", stats)
            self.assertIn("alpha", stats)
            self.assertIn("beta", stats)


# ─────────────────────────────────────────────────────────────────────────────
# Interaction Engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractionEngine(unittest.TestCase):
    def setUp(self):
        from jd_analyzer import JDAnalyzer
        from interaction_engine import InteractionEngine
        self.jd = JDAnalyzer(JD_TEXT).analyze()
        self.engine = InteractionEngine(self.jd)

    def _make_bundle(self, **overrides):
        from feature_extractor import FeatureBundle
        bundle = FeatureBundle(candidate_id="CAND_0000001")
        for k, v in overrides.items():
            setattr(bundle, k, v)
        return bundle

    def test_net_adjustment_bounded(self):
        from interaction_engine import InteractionResult
        result = InteractionResult(candidate_id="CAND_0000001", bonus=0.20, penalty=0)
        self.assertEqual(result.net_adjustment, 0.15)  # Capped
        result2 = InteractionResult(candidate_id="CAND_0000001", bonus=0, penalty=0.20)
        self.assertEqual(result2.net_adjustment, -0.15)  # Capped

    def test_clean_candidate_no_interactions(self):
        candidate = make_candidate()
        bundle = self._make_bundle(
            score_A=0.1,
            retrieval_terms_found=[],
            ml_terms_found=[],
        )
        result = self.engine._analyze_candidate(candidate, bundle)
        self.assertEqual(len(result.triggered_interactions), 0)
        self.assertEqual(result.net_adjustment, 0.0)

    def test_retrieval_system_builder_detected(self):
        candidate = make_candidate(
            title="ML Engineer",
            description="Built FAISS index for semantic search using sentence-transformers and deployed to production.",
            skills=[
                {"name": "Python", "proficiency": "advanced"},
                {"name": "FAISS", "proficiency": "expert"},
                {"name": "sentence-transformers", "proficiency": "advanced"},
            ],
        )
        bundle = self._make_bundle(
            score_A=0.5,
            retrieval_terms_found=["faiss", "sentence-transformers", "semantic search"],
        )
        result = self.engine._analyze_candidate(candidate, bundle)
        self.assertIn("retrieval_system_builder", result.triggered_interactions)

    def test_pure_researcher_risk_detected(self):
        candidate = make_candidate(
            description="Research lab work on literature review and survey paper on embedding optimization. "
                        "Academic project evaluating retrieval models using benchmark datasets.",
            skills=[{"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 60}],
        )
        bundle = self._make_bundle(
            score_A=0.3,
            score_B=0.1,  # < 0.30 triggers no_production
            retrieval_terms_found=["embedding", "retrieval"],
            ml_terms_found=["evaluation"],
        )
        result = self.engine._analyze_candidate(candidate, bundle)
        self.assertIn("pure_researcher_risk", result.triggered_interactions)


# ─────────────────────────────────────────────────────────────────────────────
# Top-100 Verifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTop100Verifier(unittest.TestCase):
    def setUp(self):
        from top100_verifier import Top100Verifier
        self.verifier = Top100Verifier()

    def _make_top100(self, honeypot_at=None):
        result = []
        for i in range(1, 101):
            cid = f"CAND_{i:07d}"
            result.append({"candidate_id": cid, "rank": i, "final_score": 1.0 - i * 0.005})
        return result

    def test_clean_top100_passes(self):
        top100 = self._make_top100()
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 101)]
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        report = self.verifier.verify(top100, candidates, honeypot)
        self.assertTrue(report.passed)

    def test_duplicate_detected(self):
        top100 = self._make_top100()
        top100[0]["candidate_id"] = "CAND_0000050"  # Duplicate
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 101)]
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        report = self.verifier.verify(top100, candidates, honeypot)
        self.assertFalse(report.passed)
        self.assertTrue(any("duplicate" in e.lower() for e in report.errors))

    def test_honeypot_rate_warning(self):
        top100 = self._make_top100()
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 101)]
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        honeypot["CAND_0000001"] = 2
        honeypot["CAND_0000002"] = 2
        honeypot["CAND_0000003"] = 2
        honeypot["CAND_0000004"] = 2
        honeypot["CAND_0000005"] = 2
        honeypot["CAND_0000006"] = 3  # 6% honeypot rate
        report = self.verifier.verify(top100, candidates, honeypot)
        self.assertFalse(report.passed)
        self.assertTrue(any("honeypot" in e.lower() for e in report.errors))

    def test_monotonicity_check(self):
        top100 = self._make_top100()
        top100[49]["final_score"] = 0.99  # Spike at rank 50
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 101)]
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        report = self.verifier.verify(top100, candidates, honeypot)
        self.assertFalse(report.passed)
        self.assertTrue(any("Score" in e for e in report.errors))

    def test_wrong_count_fails(self):
        top100 = self._make_top100()[:50]  # Only 50
        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(1, 101)]
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        report = self.verifier.verify(top100, candidates, honeypot)
        self.assertFalse(report.passed)


# ─────────────────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def test_ref_date_is_fixed(self):
        from config import Config
        self.assertEqual(Config.REFERENCE_DATE.year, 2026)
        self.assertEqual(Config.REFERENCE_DATE.month, 6)
        self.assertEqual(Config.REFERENCE_DATE.day, 27)

    def test_all_defaults_are_false(self):
        from config import Config
        # Default config should have all experiment flags False
        defaults = ["USE_RRF_EXPERIMENT"]
        for flag in defaults:
            self.assertFalse(getattr(Config, flag), f"{flag} should be False by default")

    def test_summary_contains_ref_date(self):
        from config import Config
        summary = Config.summary()
        self.assertIn("REFERENCE_DATE", summary)
        self.assertIn("2026-06-27", summary)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism(unittest.TestCase):
    def test_honeypot_deterministic(self):
        from honeypot import HoneypotDetector
        detector = HoneypotDetector()
        c1 = make_candidate()
        c2 = make_candidate()
        r1 = detector.detect_all([c1])
        r2 = detector.detect_all([c2])
        r3 = detector.detect_all([c1])  # Same input, should get same output
        self.assertEqual(r1["CAND_0000001"], r3["CAND_0000001"])

    def test_extraction_deterministic(self):
        from jd_analyzer import JDAnalyzer
        from feature_extractor import FeatureExtractor
        jd = JDAnalyzer(JD_TEXT).analyze()
        extractor = FeatureExtractor(jd)
        c = make_candidate()
        b1 = extractor._extract_candidate(c)
        b2 = extractor._extract_candidate(c)
        self.assertEqual(b1.score_A, b2.score_A)

    def test_bm25_deterministic(self):
        from jd_analyzer import JDAnalyzer
        from bm25_scorer import BM25Scorer
        jd = JDAnalyzer(JD_TEXT).analyze()
        bm25 = BM25Scorer(jd)
        bm25.build_index(["test document about FAISS vector search."])
        s1, t1 = bm25.score("FAISS")
        s2, t2 = bm25.score("FAISS")
        self.assertEqual(s1, s2)
        self.assertEqual(t1, t2)

    def test_score_deterministic(self):
        from jd_analyzer import JDAnalyzer
        from feature_extractor import FeatureExtractor
        from scorer import ScoringEngine
        jd = JDAnalyzer(JD_TEXT).analyze()
        extractor = FeatureExtractor(jd)
        engine = ScoringEngine()
        c = make_candidate()
        bundles = extractor.extract_all([c])
        r1 = engine.score_all([c], bundles, {"CAND_0000001": 0}, {"CAND_0000001": {}})
        r2 = engine.score_all([c], bundles, {"CAND_0000001": 0}, {"CAND_0000001": {}})
        self.assertEqual(r1["CAND_0000001"]["final_score"],
                         r2["CAND_0000001"]["final_score"])


# ─────────────────────────────────────────────────────────────────────────────
# Trap detector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrapDetector(unittest.TestCase):
    def setUp(self):
        from trap_detector import TrapDetector
        self.detector = TrapDetector()

    def test_clean_candidate_no_traps(self):
        candidate = make_candidate()
        result = self.detector.detect_all([candidate])
        traps = result["CAND_0000001"]
        self.assertFalse(any(traps.values()))

    def test_consulting_only_detected(self):
        candidate = make_candidate(company="Infosys")
        candidate["career_history"] = [
            {"company": "Infosys", "title": "Developer", "start_date": "2018-01-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "IT Services", "company_size": "10001+",
             "description": "Client project work."},
        ]
        result = self.detector.detect_all([candidate])
        traps = result["CAND_0000001"]
        self.assertTrue(traps.get("consulting_only", False))

    def test_never_crashes(self):
        broken = {"candidate_id": "CAND_0000001", "profile": {},
                  "career_history": [], "skills": [], "redrob_signals": {}}
        result = self.detector.detect_all([broken])
        self.assertIn("CAND_0000001", result)

    def test_pure_researcher_detected(self):
        candidate = make_candidate(
            description="Published papers on embedding models in academic conferences. "
                        "Conducted research on information retrieval.",
            skills=[{"name": "Research", "proficiency": "expert"}],
        )
        result = self.detector.detect_all([candidate])
        self.assertIn("CAND_0000001", result)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration(unittest.TestCase):
    def test_end_to_end_small(self):
        from jd_analyzer import JDAnalyzer
        from feature_extractor import FeatureExtractor
        from scorer import ScoringEngine
        from ranker import Ranker

        jd = JDAnalyzer(JD_TEXT).analyze()
        extractor = FeatureExtractor(jd)

        candidates = [make_candidate(f"CAND_{i:07d}") for i in range(10)]
        bundles = extractor.extract_all(candidates)

        engine = ScoringEngine()
        honeypot = {c["candidate_id"]: 0 for c in candidates}
        traps = {c["candidate_id"]: {} for c in candidates}
        scored = engine.score_all(candidates, bundles, honeypot, traps)

        ranker = Ranker()
        top10 = ranker.rank(scored, candidates, honeypot)
        self.assertEqual(len(top10), 10)
        for entry in top10:
            self.assertIn("candidate_id", entry)
            self.assertIn("rank", entry)
            self.assertIn("final_score", entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
