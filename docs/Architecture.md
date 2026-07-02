# Architecture

## System Overview

ICRS (Intelligent Candidate Ranking System) is a modular, pipeline-based ranking system designed for high-volume candidate evaluation. The architecture prioritizes correctness and explainability while meeting strict runtime constraints (300 seconds, 16GB RAM, CPU-only).

## Pipeline Stages

### Stage 1: JD Analysis (`jd_analyzer.py`)

Parses the embedded job description into a structured `JDContext`:
- **Must-have terms**: Core technical requirements (embeddings, retrieval, vector DB)
- **Nice-to-have terms**: Bonus qualifications (LLM fine-tuning, LTR models)
- **Disqualifiers**: Patterns that signal poor fit (pure research, consulting-only)
- **Term clusters**: Groups related terms for weighted matching

### Stage 2: Data Loading (`loader.py`)

Streams candidates from JSONL (or gzip-compressed JSONL) with:
- Memory-efficient line-by-line parsing
- Optional candidate limit for dry-run testing
- Schema-tolerant loading (missing fields default gracefully)

### Stage 3: Honeypot Detection (`honeypot.py`)

Identifies structurally impossible or fabricated profiles through 11 independent checks:

| # | Check | Signal |
|---|-------|--------|
| 1 | Inverted salary range (min > max) | Data fabrication |
| 2 | YoE inflation (claimed >> career sum) | Resume padding |
| 3 | Expert proficiency + 0 months duration | Impossible skill claim |
| 4 | Technology timeline violation (28 technologies tracked) | Fabricated history |
| 5 | Impossible skill breadth (40+ expert skills) | Synthetic profile |
| 6 | Description-domain mismatch | Wrong job domain |
| 7 | Boilerplate summary (template phrases) | Auto-generated profile |
| 8 | Overlapping full-time roles | Timeline impossibility |
| 9 | Assessment score > 100 | Structural impossibility |
| 10 | Negative salary | Data corruption |
| 11 | Future career start dates | Fabricated timeline |

Penalty schedule (when `USE_STRENGTHENED_HONEYPOT=True`):
- 1 signal: 0.30x multiplier
- 2 signals: 0.05x multiplier
- 3+ signals: 0.01x multiplier

### Stage 4: Trap Detection (`trap_detector.py`)

Identifies 10 behavioral patterns that indicate poor fit:

| Trap | Multiplier | Description |
|------|-----------|-------------|
| `consulting_only` | 0.60x | Entire career at IT services firms |
| `pure_researcher` | 0.50x | No production deployment evidence |
| `behavioral_dead` | 0.50x | Inactive 90+ days, low response rate |
| `domain_mismatch` | 0.65x | CV/speech/robotics, no NLP/IR |
| `langchain_only` | 0.75x | LLM frameworks only, no pre-LLM ML |
| `experience_gap` | 0.75x | Claims 8+ years but career shows <3 |
| `recent_hype_pivot` | 0.80x | AI experience only post-2022 |
| `certification_padder` | 0.80x | 10+ beginner-level certifications |
| `keyword_stuffer` | 0.20x | High skill count, low corroboration |
| `title_chaser` | 0.85x | Senior titles with <18-month tenures |

### Stage 5: Feature Extraction (`feature_extractor.py`)

Produces 8 normalized [0,1] component scores per candidate:

| Component | Weight | Description |
|-----------|--------|-------------|
| A - Technical Relevance | 35% | JD term matching in career descriptions |
| B - Production ML Depth | 20% | Production deployment vocabulary |
| C - Assessment Match | 12% | Skill assessment scores on relevant topics |
| D - Career Quality | 13% | Company quality, tenure stability, trajectory |
| E - Behavioral Availability | 10% | Activity recency, response rate, open-to-work |
| F - Location & Logistics | 5% | Location fit, notice period, work mode |
| G - Education Signal | 3% | Degree level and institution quality |
| H - GitHub Activity | 2% | Open-source contribution signal |

### Stage 6: Scoring (`scorer.py`)

Computes the final score using a three-phase formula:

```
1. base = (A*0.35 + B*0.20 + C*0.12 + D*0.13 + F*0.05 + G*0.03 + H*0.02) / 0.90
2. with_behavior = base * (0.85 + 0.15 * E) + career_depth_bonus
3. final = with_behavior * trap_multiplier * honeypot_multiplier
```

Key design: E (behavioral) is excluded from the base sum and applied as a soft multiplier, ensuring behavioral signals shift the score by at most +/-15%.

### Stage 7: Enhanced Pipeline (top 2,000 candidates)

Five sub-modules provide deeper analysis on the top candidates by base score:

1. **Interaction Engine** (`interaction_engine.py`): Detects multi-signal combinations (retrieval + vector DB + embeddings = retrieval system builder). Adjustments bounded to [-0.15, +0.15].

2. **Evidence Consistency** (`evidence_consistency_engine.py`): Cross-references skills claims against career description evidence. Identifies uncorroborated expert claims.

3. **Context Evidence** (`context_evidence_analyzer.py`): Analyzes description-level context for production, scale, and deployment indicators.

4. **Corroborated Skill Scoring** (`corroborated_skill_scorer.py`): Blends raw assessment scores with corroboration evidence (60% corroborated, 40% raw).

5. **Score Calibration** (`score_calibrator.py`): Applies sigmoid normalization across all 100K candidates for proper score distribution.

### Stage 8: Ranking & Verification

- **Ranker** (`ranker.py`): Selects top 100 by calibrated score. Audits top 10 for honeypot contamination.
- **Top-100 Verifier** (`top100_verifier.py`): Checks honeypot rate (<5%), score distribution, and triggers demotion/backfill if needed.

### Stage 9: Output

- **Reasoning** (`reasoning.py`): Generates unique, evidence-based explanations referencing specific career facts, assessment scores, and JD connections.
- **Writer** (`writer.py`): Rescales scores linearly to [0.60, 0.99] for the final CSV.
- **Validator** (`validator.py`): Verifies format compliance (100 rows, unique IDs, valid scores, unique reasonings).

## Feature Flag System

All improvements are gated behind independent boolean flags in `config.py`. Each flag:
- Defaults to a safe value
- Can be toggled via environment variable
- Preserves original behavior when disabled
- Enables A/B comparison of old vs new behavior

## Determinism

`Config.REFERENCE_DATE = date(2026, 6, 27)` replaces all `date.today()` calls, ensuring identical output regardless of execution date. This is critical for Stage 3 (code reproduction) evaluation.
