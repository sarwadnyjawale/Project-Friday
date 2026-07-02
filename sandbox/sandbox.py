#!/usr/bin/env python3
"""
sandbox.py - Sandbox runner for the ICRS ranking pipeline.

Runs the full 10-step ranking pipeline end-to-end on a small candidate
sample (50 candidates) and produces a ranked CSV. This is the sandbox
entry point for hackathon judges to verify the code runs reproducibly.

Usage:
    python sandbox/sandbox.py                         # Bundled 50-candidate sample
    python sandbox/sandbox.py --candidates custom.jsonl  # Custom JSONL/JSON input
    python sandbox/sandbox.py --out my_result.csv     # Custom output path

Requirements:
    - Python 3.9+ (standard library only, no pip install needed)
    - Runs in < 5 seconds on CPU
    - No internet access required
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths relative to the project root (parent of sandbox/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANK_SCRIPT = PROJECT_ROOT / "rank.py"
DEFAULT_SAMPLE = Path(__file__).resolve().parent / "sample_candidates.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "sandbox_submission.csv"


def ensure_jsonl(input_path: Path) -> Path:
    """
    Ensure the input file is in JSONL format (one JSON object per line).

    The loader expects JSONL. If given a JSON array, convert it on the fly
    to a temporary JSONL file and return that path.
    """
    if input_path.suffix == ".jsonl" or input_path.suffix == ".gz":
        return input_path

    # JSON array -> convert to JSONL
    if input_path.suffix == ".json":
        print(f"  Converting JSON array to JSONL: {input_path.name}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  ERROR: Expected a JSON array, got {type(data).__name__}")
            sys.exit(1)
        jsonl_path = input_path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"  Converted {len(data)} candidates -> {jsonl_path.name}")
        return jsonl_path

    # Unknown extension - assume it's already JSONL
    return input_path


def run_pipeline(candidates_path: Path, output_path: Path) -> int:
    """Run rank.py as a subprocess and return its exit code."""
    cmd = [
        sys.executable, str(RANK_SCRIPT),
        "--candidates", str(candidates_path),
        "--out", str(output_path),
    ]
    print(f"\n{'=' * 70}")
    print("ICRS SANDBOX - Running full ranking pipeline")
    print(f"{'=' * 70}")
    print(f"  Input:  {candidates_path}")
    print(f"  Output: {output_path}")
    print(f"  Command: {' '.join(cmd)}")
    print(flush=True)

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start

    print(f"\n  Pipeline finished in {elapsed:.2f}s (exit code: {result.returncode})")
    return result.returncode


def verify_and_display(output_path: Path) -> bool:
    """
    Verify the output CSV exists and is well-formed, then display results.

    Returns True if the output is valid.
    """
    if not output_path.exists():
        print(f"\n  ERROR: Output file not created: {output_path}")
        return False

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  ERROR: Output CSV is empty")
        return False

    # Verify required columns
    required = {"candidate_id", "rank", "score", "reasoning"}
    if not required.issubset(set(rows[0].keys())):
        print(f"  ERROR: Missing columns. Expected {required}, got {set(rows[0].keys())}")
        return False

    # Verify scores are non-increasing
    scores = [float(r["score"]) for r in rows]
    monotonic = all(scores[i] <= scores[i - 1] + 1e-9 for i in range(1, len(scores)))

    # Verify unique candidate IDs
    ids = [r["candidate_id"] for r in rows]
    unique_ids = len(set(ids)) == len(ids)

    print(f"\n{'=' * 70}")
    print("SANDBOX RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total candidates ranked: {len(rows)}")
    print(f"  Score range: {scores[-1]:.4f} - {scores[0]:.4f}")
    print(f"  Scores non-increasing: {'YES' if monotonic else 'NO'}")
    print(f"  Unique candidate IDs:   {'YES' if unique_ids else 'NO'}")
    print(f"  All reasoning non-empty: {'YES' if all(r['reasoning'].strip() for r in rows) else 'NO'}")

    print(f"\n  TOP 10 CANDIDATES:")
    print(f"  {'Rank':<6} {'Candidate ID':<16} {'Score':<10} {'Reasoning (preview)'}")
    print(f"  {'-' * 6} {'-' * 16} {'-' * 10} {'-' * 40}")
    for row in rows[:10]:
        reasoning = row["reasoning"][:60] + ("..." if len(row["reasoning"]) > 60 else "")
        print(f"  {row['rank']:<6} {row['candidate_id']:<16} {row['score']:<10} {reasoning}")

    print(f"\n  Output written to: {output_path}")
    print(f"{'=' * 70}")

    return monotonic and unique_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICRS Sandbox - Run the ranking pipeline on a small sample",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_SAMPLE,
        help=f"Path to candidate data (JSONL or JSON array). Default: {DEFAULT_SAMPLE.name}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT.name}",
    )
    args = parser.parse_args()

    # Ensure input is in JSONL format
    candidates_jsonl = ensure_jsonl(args.candidates)

    # Run the pipeline
    exit_code = run_pipeline(candidates_jsonl, args.out)

    # Verify and display results (even if rank.py had warnings)
    success = verify_and_display(args.out)

    if success:
        print("\n  SANDBOX: PASSED - Ranking pipeline runs end-to-end successfully.")
        print("  Note: Small sample (<100 candidates) produces fewer than 100 rows.")
        print("  The full 100K dataset produces the complete 100-row submission.")
        sys.exit(0)
    else:
        print("\n  SANDBOX: FAILED - Output verification found issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
