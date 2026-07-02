# Methodology

## Design Philosophy

> "The right answer is not finding candidates whose skills section contains the most AI keywords."
> — Job Description

ICRS treats candidate evaluation as a trust calibration problem, not a keyword matching exercise:

- **Career descriptions are evidence.** They describe what candidates actually built.
- **Skills lists are hypotheses.** They describe what candidates claim to know.
- **Behavioral signals measure reachability.** A perfect candidate who is unreachable is not a candidate.
- **Honeypot signals mean ignore entirely.** Structurally impossible data is fabricated data.

## Scoring Formula

### Base Score

The weighted sum excludes the behavioral component E:

```
base = (A*0.35 + B*0.20 + C*0.12 + D*0.13 + F*0.05 + G*0.03 + H*0.02) / 0.90
```

The division by 0.90 normalizes to a [0, 1] ceiling since E (10% weight) is handled separately.

### Behavioral Soft Multiplier

```
adjusted = base * (0.85 + 0.15 * E)
```

This ensures behavioral signals shift the score by at most +/-15%, preventing platform activity from dominating technical evaluation. A technically strong candidate with low platform engagement still ranks well.

### Career Depth Bonus

```
if product_company and relevant_years >= 3:
    bonus = 0.05 * min(1.0, relevant_years / 8.0) * tenure_stability
```

Rewards deep experience at product companies. Creates natural score differentiation among top candidates.

### Trap and Honeypot Multipliers

Multipliers compound multiplicatively:

```
final = adjusted * honeypot_mult * trap1_mult * trap2_mult * ...
```

A consulting-only (0.60x) behavioral-dead (0.50x) candidate gets 0.30x — correctly reflecting that both problems together are worse than either alone.

## Honeypot Detection Strategy

The dataset contains approximately 80 intentionally planted honeypot candidates. Our detection uses 11 independent structural checks, each testing for conditions that are impossible in legitimate profiles:

1. **Inverted salary** (min > max): A data entry that makes no semantic sense
2. **YoE inflation**: Claims 15 years but career history sums to 5
3. **Expert + 0 duration**: Claims expert proficiency in a skill with 0 months of experience
4. **Technology timeline**: Claims 3 years of GPT-4 experience when GPT-4 launched 18 months ago (28 technologies tracked with launch dates)
5. **Impossible breadth**: 40+ skills at expert level (humanly impossible)
6. **Domain mismatch**: Career descriptions discuss marketing but skills list AI terms
7. **Boilerplate summary**: Uses template phrases ("results-driven professional")
8. **Overlapping FT roles**: Two full-time positions at different companies simultaneously
9. **Assessment > 100**: Score exceeds the maximum possible value
10. **Negative salary**: Structurally impossible compensation
11. **Future start dates**: Career roles beginning after the reference date

### Graduated Penalties

Rather than binary keep/reject, we apply graduated penalties based on signal count and confidence:

- **1 high-confidence signal**: 0.30x (strong penalty, but not elimination — could be data noise)
- **2 signals**: 0.05x (near-certain honeypot)
- **3+ signals**: 0.01x (definite honeypot, effectively zero)

## Trap Detection Strategy

Traps are behavioral patterns the JD explicitly identifies as disqualifiers or risk factors. Unlike honeypots (structurally impossible data), traps are plausible but undesirable patterns:

### Consulting-Only (0.60x)
Career entirely at IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Genpact, etc.). The JD explicitly states preference for product company experience.

### LangChain-Only (0.75x)
AI experience consists entirely of LLM framework usage with no pre-LLM era ML production experience. The JD requires "production experience with embeddings-based retrieval systems" — not LangChain tutorials.

### Pure Researcher (0.50x)
Career spent in research environments without production deployment evidence. The JD states: "If you've spent your career in pure research environments without any production deployment — we will not move forward."

### Behavioral Dead (0.50x)
Inactive for 90+ days with low recruiter response rate. The JD notes: "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."

### Keyword Stuffer (0.20x)
High skill count with low career corroboration. Requires multiple supporting signals to avoid false positives: skills-career mismatch + low assessment scores + no production evidence.

## Enhanced Pipeline

The enhanced pipeline runs on the top 2,000 candidates by base score, providing deeper analysis that would be too expensive for all 100K:

### Interaction Engine
Detects multi-feature combinations that signal stronger fit:
- **Retrieval System Builder**: retrieval + vector DB + embeddings = shipped a retrieval system (+0.12)
- **Tier 5 Discovery**: recommendation system + product company + scale evidence (+0.14)
- **Evaluation Framework**: production ML + evaluation metrics + A/B testing (+0.10)

This directly addresses the JD's Tier 5 candidate description: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."

### Evidence Consistency
Cross-references skills claims against career description evidence. A candidate claiming "expert" in PyTorch whose career descriptions never mention neural networks, training, or model development gets an adjustment.

### Score Calibration
Applies sigmoid transformation to convert raw scores into a well-distributed [0, 1] range. This prevents score compression where all top candidates cluster within a 0.001 range.

## Output Calibration

The writer rescales top-100 scores linearly to [0.60, 0.99]:

```
rescaled = 0.60 + (raw - min) / (max - min) * 0.39
```

This produces a meaningful score spread for NDCG evaluation while keeping all scores in a professionally reasonable range.

## Reasoning Generation

Each candidate receives a unique, evidence-based reasoning string that:
- Cites specific career evidence (company name, role, years)
- References relevant assessment scores
- Connects to JD requirements
- Includes honest cautions (notice period, inactivity, limited evidence)
- Avoids hallucination (only references data present in the candidate profile)

## Determinism

All date-dependent computations use `Config.REFERENCE_DATE = date(2026, 6, 27)` instead of `date.today()`. This ensures:
- Identical behavioral scores regardless of execution date
- Reproducible honeypot detection (overlapping role checks, future date checks)
- Compliance with Stage 3 code reproduction evaluation
