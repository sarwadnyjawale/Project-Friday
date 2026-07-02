# ICRS - Intelligent Candidate Ranking System

A high-performance candidate ranking pipeline for the **Redrob AI Hackathon**. Processes 100,000 candidates in under 5 minutes on CPU, producing a ranked top-100 shortlist with human-quality reasoning — matching the judgment of a senior recruiter with 10+ years of experience.

## Problem Statement

Given a job description for a **Senior AI Engineer** role and a dataset of 100,000 candidate profiles (with intentionally planted honeypots, traps, and data quality issues), build a system that:

1. Identifies and penalizes fraudulent/honeypot candidates
2. Detects trap patterns (consulting-only, keyword stuffers, title chasers, etc.)
3. Ranks candidates by genuine fit, not keyword matching
4. Produces unique, evidence-based reasoning for each selection
5. Runs under **300 seconds** on CPU with **16GB RAM**, no GPU, no internet

## Solution Overview

ICRS uses a **10-step orchestrated pipeline** that mirrors how an experienced recruiter evaluates candidates:

1. **JD Analysis** - Extracts must-have terms, nice-to-have terms, and disqualifiers
2. **Candidate Loading** - Streams 100K profiles from JSONL
3. **Honeypot Detection** - 11 structural checks catch fabricated profiles
4. **Trap Detection** - 10 behavioral patterns identify problematic candidates
5. **Feature Extraction** - 8 scored components (A-H) per candidate
6. **Scoring** - Weighted formula with graduated multipliers
7. **Enhanced Pipeline** - Evidence consistency, corroboration, confidence, calibration
8. **Ranking** - Top-100 selection with verification
9. **Reasoning** - Candidate-specific, evidence-based explanations
10. **Output** - Validated CSV submission

## Architecture

```
candidates.jsonl
       |
       v
  [JD Analyzer] ──> JDContext (108 must-have, 57 nice-to-have terms)
       |
       v
  [Loader] ──> 100,000 candidate dicts
       |
       ├──> [Honeypot Detector] ──> honeypot_scores {cid: signal_count}
       |         11 checks: inverted salary, YoE inflation, expert+0 duration,
       |         technology timeline, impossible breadth, domain mismatch,
       |         boilerplate summary, overlapping FT roles, assessment >100,
       |         negative salary, future start dates
       |
       ├──> [Trap Detector] ──> trap_flags {cid: {trap: bool}}
       |         10 traps: consulting_only, langchain_only, pure_researcher,
       |         behavioral_dead, keyword_stuffer, recent_hype_pivot,
       |         title_chaser, domain_mismatch, certification_padder,
       |         experience_gap
       |
       └──> [Feature Extractor] ──> feature_bundles {cid: FeatureBundle}
                8 components: A(technical), B(production), C(assessment),
                D(career), E(behavioral), F(logistics), G(education), H(github)
                     |
                     v
              [Scoring Engine]
              A(35%) + B(20%) + C(12%) + D(13%) weighted sum
              E applied as soft multiplier (0.85 + 0.15*E)
              F(5%) + G(3%) + H(2%) included in base
              Trap/honeypot multipliers compound
                     |
                     v
              [Enhanced Pipeline] (top 2,000 candidates)
              ├── Interaction Engine (Tier 5 discovery, multi-signal bonuses)
              ├── Evidence Consistency (career-skills corroboration)
              ├── Context Evidence (description-level analysis)
              ├── Corroborated Skill Scoring (blended C score)
              ├── Confidence Estimation (tie-breaking)
              └── Score Calibration (sigmoid normalization)
                     |
                     v
              [Ranker] ──> Top 100 (verified clean)
                     |
                     v
              [Reasoning Generator] ──> 100 unique explanations
                     |
                     v
              [Writer] ──> submission.csv (scores rescaled to 0.60-0.99)
                     |
                     v
              [Validator] ──> Format + integrity checks
```

## Folder Structure

```
Project-Friday/
├── rank.py                          # Main orchestrator (entry point)
├── config.py                        # Feature flags and configuration
├── loader.py                        # JSONL/gzip data loader
├── jd_analyzer.py                   # Job description analysis
├── honeypot.py                      # Honeypot detection (11 checks)
├── trap_detector.py                 # Trap pattern detection (10 traps)
├── feature_extractor.py             # 8-component feature scoring
├── scorer.py                        # Weighted scoring with multipliers
├── ranker.py                        # Top-100 selection and audit
├── reasoning.py                     # Evidence-based reasoning generation
├── writer.py                        # CSV output with score rescaling
├── validator.py                     # Submission format validation
├── interaction_engine.py            # Multi-signal interaction detection
├── evidence_consistency_engine.py   # Career-skills corroboration
├── context_evidence_analyzer.py     # Description-level evidence analysis
├── corroborated_skill_scorer.py     # Blended assessment scoring
├── confidence_estimator.py          # Score confidence estimation
├── score_calibrator.py              # Sigmoid score normalization
├── top100_verifier.py               # Pre-export verification layer
├── bm25_scorer.py                   # BM25 scoring (feature-flagged)
├── semantic_scorer.py               # Semantic similarity (feature-flagged)
├── precompute_embeddings.py         # Embedding pre-computation utility
├── tests.py                         # Unit and integration tests
├── submission_metadata.yaml         # Hackathon submission metadata
├── submission.csv                   # Generated submission output
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Sandbox Docker image
├── .dockerignore                    # Docker build context exclusions
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
├── DATA/
│   └── README.md                    # Data placement instructions
├── sandbox/
│   ├── sandbox.py                   # Sandbox runner (entry point for demo)
│   ├── sample_candidates.jsonl      # 50-candidate sample for sandbox
│   └── ICRS_Sandbox_Colab.ipynb     # Google Colab notebook
└── docs/
    ├── Architecture.md              # Detailed architecture documentation
    └── Methodology.md               # Scoring methodology and design decisions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sarwadnyjawale/Project-Friday.git
cd Project-Friday

# Create virtual environment (recommended)
python -m venv venv

# Activate the virtual environment:
#   Windows PowerShell:  venv\Scripts\activate
#   Linux/Mac:           source venv/bin/activate

# Install dependencies (stdlib only — no packages required by default)
pip install -r requirements.txt

# Place the dataset (candidates.jsonl is NOT in the repo — 487MB, excluded via .gitignore)
#   Windows PowerShell:  copy "C:\path\to\candidates.jsonl" DATA\
#   Linux/Mac:           cp /path/to/candidates.jsonl DATA/
```

### Requirements

- Python 3.10+ (uses PEP 604 union type syntax)
- No GPU required
- No internet access required during ranking
- 16GB RAM maximum
- No third-party packages required (stdlib only)
- Works on Windows, Linux, macOS

## Reproduce Your Submission

The single command to produce the submission CSV from the candidates file:

```bash
python rank.py --candidates DATA/candidates.jsonl --out submission.csv
```

This runs the full 10-step pipeline end-to-end in ~240 seconds on CPU with 16GB RAM. No pre-computation is required — all feature flags that need pre-computed artifacts (embeddings, BM25 index) are disabled by default.

### Pre-computation (optional, not required)

The default configuration uses no pre-computed artifacts. To enable optional semantic scoring:

```bash
# Install optional dependencies
pip install numpy sentence-transformers

# Pre-compute embeddings (may exceed 5-minute window — spec 10.3 allows this)
python precompute_embeddings.py --candidates DATA/candidates.jsonl

# Run with semantic scoring enabled
USE_HYBRID_SEMANTIC=true python rank.py --candidates DATA/candidates.jsonl --out submission.csv
```

The ranking step itself completes within 5 minutes regardless of configuration.

## Usage

### Run the full ranking pipeline

```bash
python rank.py --candidates DATA/candidates.jsonl --out submission.csv
```

### Quick test (first 1,000 candidates)

```bash
python rank.py --candidates DATA/candidates.jsonl --out test_submission.csv --dry-run
```

### Explain a specific candidate

```bash
python rank.py --candidates DATA/candidates.jsonl --out submission.csv --explain CAND_0081846
```

### Run tests

```bash
python -m pytest tests.py -v
```

### Validate submission

```bash
python -c "from validator import SubmissionValidator; r = SubmissionValidator('submission.csv').validate(); print('PASSED' if r.passed else r.errors)"
```

## Sandbox / Demo

A sandbox is provided to verify the ranking pipeline runs end-to-end reproducibly. It uses a bundled 50-candidate sample and completes in under 5 seconds on CPU. Two options are available:

### Option 1: Docker (recommended)

The Dockerfile builds a self-contained image with no external dependencies — the pipeline uses only the Python standard library.

```bash
# Build the image
docker build -t icrs-sandbox .

# Run (see ranked results in stdout)
docker run --rm icrs-sandbox

# Run (extract the CSV to your current directory)
docker run --rm -v "$PWD:/out" icrs-sandbox --out /out/sandbox_submission.csv
```

On Windows PowerShell, use `${PWD}:/out` instead of `"$PWD:/out"`.

### Option 2: Google Colab

Open `sandbox/ICRS_Sandbox_Colab.ipynb` in [Google Colab](https://colab.research.google.com) and run all cells. The notebook clones the repo, runs the pipeline, and displays the ranked results with score visualizations.

### Option 3: Local

```bash
# Run the sandbox directly (no Docker, no Colab)
python sandbox/sandbox.py
```

### Sandbox output

The sandbox produces a ranked CSV with `candidate_id`, `rank`, `score`, and `reasoning` columns. With the 50-candidate sample, honeypot/trap detection filters out fabricated profiles, producing a clean ranked shortlist (typically 20-30 candidates). The full 100K dataset produces the complete 100-row submission required by the hackathon spec.

## Runtime

| Step | Duration | Description |
|------|----------|-------------|
| JD Analysis | <1s | Extract terms from job description |
| Loading | ~12s | Stream 100K candidates from JSONL |
| Honeypot Detection | ~18s | 11 structural integrity checks |
| Trap Detection | ~8s | 10 behavioral pattern detectors |
| Feature Extraction | ~135s | 8-component scoring per candidate |
| Scoring | ~1s | Weighted formula + multipliers |
| Enhanced Pipeline | ~64s | Top-2000 deep analysis |
| Ranking + Verification | <1s | Top-100 selection + audit |
| Reasoning | <1s | 100 unique explanations |
| Output + Validation | <1s | CSV write + format check |
| **Total** | **~240s** | **Well under 300s limit** |

## Output Format

`submission.csv` contains exactly 100 rows:

| Column | Description |
|--------|-------------|
| `candidate_id` | Unique candidate identifier (e.g., CAND_0081846) |
| `rank` | Position 1-100 |
| `score` | Normalized score (0.60-0.99) |
| `reasoning` | Evidence-based explanation for the ranking |

## Feature Flags

The system uses feature flags (in `config.py`) for safe, incremental improvements:

| Flag | Default | Description |
|------|---------|-------------|
| `USE_INTERACTION_ENGINE` | True | Multi-signal interaction detection |
| `USE_NEW_CONFIDENCE` | True | Tie-breaker only confidence mode |
| `USE_IMPROVED_KEYWORD_STUFFER` | True | Multi-signal keyword stuffer detection |
| `USE_STRENGTHENED_HONEYPOT` | True | Single-signal honeypot penalty |
| `USE_TOP100_VERIFICATION` | True | Pre-export verification layer |
| `USE_NEW_BM25` | False | BM25 scoring (slower, higher accuracy) |
| `USE_HYBRID_SEMANTIC` | False | Semantic embedding similarity |

## Evaluation Metrics

The system is optimized for the hackathon evaluation formula:

- **NDCG@10** (50%) - Top-10 ranking quality
- **NDCG@50** (30%) - Top-50 ranking quality
- **MAP** (15%) - Mean Average Precision
- **P@10** (5%) - Precision at rank 10

## Key Design Decisions

1. **Career evidence over keyword matching** - The JD explicitly warns: "The right answer is not finding candidates whose skills section contains the most AI keywords." Our system weights career descriptions (evidence) over skills lists (claims).

2. **Graduated honeypot penalties** - 1 signal = 0.30x, 2 signals = 0.05x, 3+ signals = 0.01x. High-confidence checks (inverted salary, expert+0 duration) trigger meaningful penalty even with a single signal.

3. **Behavioral soft multiplier** - Behavioral score E shifts the final score by at most +/-15%, preventing platform activity from dominating technical evaluation.

4. **Tier 5 candidate discovery** - The Interaction Engine detects candidates who built recommendation/search systems at product companies, even without mentioning "RAG" or "Pinecone" — exactly what the JD describes.

5. **Deterministic output** - `Config.REFERENCE_DATE` replaces all `date.today()` calls, ensuring identical results regardless of when the pipeline runs.

## License

MIT License - see [LICENSE](LICENSE) for details.
