# DATA Directory

Place the hackathon dataset here before running the pipeline.

## Required File

- `candidates.jsonl` - The candidate dataset (100,000 candidates in JSONL format)

## Supported Formats

The loader accepts both:
- `candidates.jsonl` (uncompressed)
- `candidates.jsonl.gz` (gzip-compressed)

## Usage

```bash
python rank.py --candidates DATA/candidates.jsonl --out submission.csv
```
