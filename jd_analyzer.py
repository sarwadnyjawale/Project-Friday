"""
jd_analyzer.py - Job description parser and vocabulary builder.

The JDContext produced here is the reference object passed to all scoring modules.
It contains:
    - Structured term vocabularies grouped by relevance cluster
    - A flat {term: weight} dict for fast BM25-style scoring
    - Disqualifier pattern lists for trap detection
    - Ideal company and institution type indicators

Design notes:
    - All vocabulary is hand-crafted from the JD - no network calls, no model downloads
    - Synonym expansion is explicit and conservative - we expand only when we are
      confident the synonym means the same thing in this technical context
    - The cluster weights match the scoring architecture in scorer.py
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class JDContext:
    """
    Structured representation of the job description for use by all scoring modules.
    """
    # Weighted term vocabulary: {term: weight 0.0-1.0}
    # Used by BM25/TF-IDF matching against career descriptions
    term_weights: Dict[str, float] = field(default_factory=dict)

    # Raw term lists by cluster for component-level scoring
    must_have_terms: List[str] = field(default_factory=list)
    nice_to_have_terms: List[str] = field(default_factory=list)
    disqualifier_terms: List[str] = field(default_factory=list)

    # Cluster-specific vocabularies
    retrieval_search_terms: Set[str] = field(default_factory=set)
    ml_production_terms: Set[str] = field(default_factory=set)
    product_engineering_terms: Set[str] = field(default_factory=set)
    scale_infra_terms: Set[str] = field(default_factory=set)

    # Negative signal vocabulary (penalize if prominent without positive context)
    anti_production_terms: Set[str] = field(default_factory=set)

    # Consulting firm names for is_consulting_only detection
    consulting_firm_names: Set[str] = field(default_factory=set)

    # Target company characteristics
    target_product_company_indicators: Set[str] = field(default_factory=set)

    # Location constraints
    target_locations: Set[str] = field(default_factory=set)
    target_country: str = "India"


class JDAnalyzer:
    """
    Parses the job description and builds the JDContext vocabulary.

    The vocabulary construction is the most critical design decision in this system.
    The terms are organized into four clusters matching the scoring architecture.
    Each cluster has a base weight reflecting its importance in the JD.
    """

    def __init__(self, jd_text: str) -> None:
        self.jd_text = jd_text

    def analyze(self) -> JDContext:
        ctx = JDContext()
        ctx.retrieval_search_terms = self._build_retrieval_cluster()
        ctx.ml_production_terms = self._build_ml_production_cluster()
        ctx.product_engineering_terms = self._build_product_cluster()
        ctx.scale_infra_terms = self._build_scale_cluster()
        ctx.anti_production_terms = self._build_anti_production_signals()
        ctx.consulting_firm_names = self._build_consulting_firms()
        ctx.target_product_company_indicators = self._build_product_companies()
        ctx.target_locations = self._build_target_locations()
        ctx.term_weights = self._build_term_weight_dict(ctx)
        ctx.must_have_terms = list(ctx.retrieval_search_terms | ctx.ml_production_terms)
        ctx.nice_to_have_terms = list(ctx.product_engineering_terms | ctx.scale_infra_terms)
        ctx.disqualifier_terms = list(ctx.anti_production_terms)
        return ctx

    def _build_retrieval_cluster(self) -> Set[str]:
        """
        Cluster 1: Retrieval and Search Systems.
        These are the highest-weight terms - the JD's must-have requirements.
        Weight multiplier: 1.0 (full weight)
        """
        return {
            # Vector databases - specific product names
            "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
            "opensearch", "elasticsearch", "solr",
            # ANN and dense retrieval concepts
            "approximate nearest neighbor", "ann index", "hnsw", "ivf",
            "vector search", "dense retrieval", "dense vector",
            "semantic search", "hybrid search", "sparse dense hybrid",
            "bm25", "inverted index", "tf-idf", "tfidf",
            # Embedding model families
            "sentence-transformers", "sentence transformer", "sbert",
            "bge", "e5", "openai embeddings", "ada", "text-embedding",
            "bi-encoder", "cross-encoder", "dual encoder",
            # Retrieval-specific operational concerns
            "embedding drift", "index refresh", "retrieval quality",
            "retrieval regression", "vector index", "embedding index",
            "re-ranking", "reranking", "retrieval augmented",
            "rag", "retrieval-augmented generation",
            # Evaluation metrics specific to retrieval
            "ndcg", "mrr", "map", "mean average precision",
            "normalized discounted cumulative gain",
            "mean reciprocal rank", "recall at k", "precision at k",
        }

    def _build_ml_production_cluster(self) -> Set[str]:
        """
        Cluster 2: ML/AI Engineering with production signals.
        Weight multiplier: 0.85 (very high - these are production requirements)
        """
        return {
            # Model serving and deployment
            "model serving", "inference pipeline", "model deployment",
            "production deployment", "deployed to production", "serving infrastructure",
            "real-time inference", "batch inference", "online serving",
            # Recommendation systems (adjacent to retrieval - valid signal)
            "recommendation system", "recommender system", "collaborative filtering",
            "two-tower model", "two tower", "matrix factorization",
            "content-based filtering", "candidate generation",
            # Ranking and LTR
            "learning to rank", "ltr", "lambdamart", "xgboost ranking",
            "pointwise ranking", "pairwise ranking", "listwise ranking",
            # Evaluation frameworks (JD-specific requirement)
            "a/b test", "a/b testing", "online evaluation",
            "offline evaluation", "offline to online", "experiment",
            "evaluation framework", "ranking evaluation",
            # Model lifecycle
            "model monitoring", "drift detection", "data drift",
            "concept drift", "model performance", "mlflow", "experiment tracking",
            "model versioning", "model registry",
            # Fine-tuning (nice to have)
            "fine-tuning", "fine tuning", "lora", "qlora", "peft",
            "instruction tuning", "rlhf", "dpo",
            # Embeddings (repeated from retrieval cluster for emphasis)
            "embedding model", "embedding space", "embedding quality",
            # LLM work
            "llm", "large language model", "language model",
        }

    def _build_product_cluster(self) -> Set[str]:
        """
        Cluster 3: Product engineering signals.
        Weight multiplier: 0.65 (medium - validates production context)
        """
        return {
            # Production context indicators
            "shipped to production", "real users", "production traffic",
            "user-facing", "customer-facing", "end users",
            "latency", "throughput", "p99", "p95", "sla",
            "qps", "queries per second", "requests per second",
            "million users", "thousands of users", "at scale",
            # Engineering quality
            "python", "api design", "system design",
            "ci/cd", "continuous integration", "unit test", "integration test",
            "code review", "pull request",
            # HR tech / recruiting tech (nice-to-have but domain bonus)
            "hr tech", "hrtech", "recruiting", "talent", "candidate matching",
            "job matching", "resume screening", "applicant",
            "marketplace", "two-sided marketplace",
        }

    def _build_scale_cluster(self) -> Set[str]:
        """
        Cluster 4: Scale and infrastructure.
        Weight multiplier: 0.45 (moderate - good signal but not required)
        """
        return {
            "distributed system", "kubernetes", "docker",
            "spark", "kafka", "airflow",
            "large scale", "billion records", "petabyte",
            "inference optimization", "model quantization", "pruning",
            "onnx", "tensorrt", "triton",
            "cloud", "aws", "gcp", "azure",
            "mlops", "model pipeline",
        }

    def _build_anti_production_signals(self) -> Set[str]:
        """
        Terms that indicate research/tutorial context - penalize when prominent.
        These are not absolute disqualifiers but reduce production depth score.
        """
        return {
            "research prototype", "proof of concept", "poc",
            "tutorial", "academic project", "jupyter notebook",
            "notebook experiment", "exploratory analysis",
            "client demo", "workshop", "presentation",
            "theoretical", "literature review", "survey paper",
        }

    def _build_consulting_firms(self) -> Set[str]:
        """
        Tier-1 Indian consulting/services firms listed as explicit disqualifiers in JD.
        Used by is_consulting_only trap detector.
        Lowercase for case-insensitive matching.
        """
        return {
            "tcs", "tata consultancy", "infosys", "wipro",
            "accenture", "cognizant", "capgemini", "tech mahindra",
            "hcl technologies", "hcl tech", "mphasis", "hexaware",
            "mindtree", "l&t infotech", "niit technologies",
            "cyient", "zensar", "persistent systems",
            "ltimindtree", "birlasoft", "coforge",
        }

    def _build_product_companies(self) -> Set[str]:
        """
        Indian product companies that are positive signals for career quality.
        Not an exhaustive list - used as a soft boost, not a hard filter.
        """
        return {
            "swiggy", "zomato", "flipkart", "amazon", "google", "microsoft",
            "meta", "razorpay", "cred", "meesho", "ola", "oyo", "byju",
            "zepto", "blinkit", "phonepe", "paytm", "groww", "sharechat",
            "freshworks", "zoho", "postman", "druva", "browserstack",
            "hasura", "slintel", "sifted", "redrob",
        }

    def _build_target_locations(self) -> Set[str]:
        """
        Acceptable locations for this role. Lowercase for matching.
        """
        return {
            "pune", "noida", "delhi", "delhi ncr", "ncr",
            "gurgaon", "gurugram", "hyderabad", "bengaluru", "bangalore",
            "mumbai", "bombay", "india",
        }

    def _build_term_weight_dict(self, ctx: JDContext) -> Dict[str, float]:
        """
        Build a flat {term: weight} dict used for BM25-style scoring.

        Weight reflects the term's importance in the JD's requirements:
        - Retrieval cluster: 1.0 (must-have technical core)
        - ML production cluster: 0.85 (must-have production evidence)
        - Product cluster: 0.65 (validates context)
        - Scale cluster: 0.45 (nice-to-have)
        """
        term_weights: Dict[str, float] = {}

        for term in ctx.retrieval_search_terms:
            term_weights[term] = 1.0

        for term in ctx.ml_production_terms:
            # Don't overwrite if already in retrieval cluster (higher weight)
            term_weights.setdefault(term, 0.85)

        for term in ctx.product_engineering_terms:
            term_weights.setdefault(term, 0.65)

        for term in ctx.scale_infra_terms:
            term_weights.setdefault(term, 0.45)

        # Anti-production terms get a small negative weight
        # Handled separately in scorer, not here
        return term_weights