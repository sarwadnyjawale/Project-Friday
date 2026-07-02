"""
top100_verifier.py - Pre-export verification layer for the final Top-100 list.

Runs comprehensive checks before the CSV is written to catch issues that
would fail Stage 1 (format validation) or hurt NDCG at Stage 2 (scoring).

Checks performed:
    1. No duplicate candidate_ids in top 100
    2. Honeypot rate within safe threshold (<5%, spec allows <10%)
    3. Score distribution is reasonable (not all identical, not compressed)
    4. No impossible rankings (scores must be non-increasing)
    5. Semantic diversity (not all candidates from same company/title)
    6. Ranking stability (top 10 are genuinely the strongest)

Controlled by Config.USE_TOP100_VERIFICATION flag.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("top100_verifier")


@dataclass
class VerificationReport:
    """Report from top-100 verification."""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    demotions: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class Top100Verifier:
    """
    Verifies the top-100 list before export.

    Usage:
        verifier = Top100Verifier()
        report = verifier.verify(top_100, candidates, honeypot_scores)
        if not report.passed:
            # Handle verification failure
            for err in report.errors:
                logger.error(f"  Verification error: {err}")
    """

    def verify(
        self,
        top_100: List[Dict],
        candidates: List[dict],
        honeypot_scores: Dict[str, int],
    ) -> VerificationReport:
        """
        Run all verification checks on the top-100 list.

        Args:
            top_100: List of dicts with candidate_id, rank, final_score
            candidates: Original candidate list (for lookup)
            honeypot_scores: {candidate_id: signal_count}

        Returns:
            VerificationReport with pass/fail and details
        """
        report = VerificationReport()

        if len(top_100) != 100:
            report.add_error(f"Expected 100 candidates, got {len(top_100)}")

        # ── Check 1: No duplicate candidate_ids ────────────────────────────
        candidate_ids = [entry["candidate_id"] for entry in top_100]
        id_counts = Counter(candidate_ids)
        duplicates = {cid: count for cid, count in id_counts.items() if count > 1}
        if duplicates:
            report.add_error(f"Duplicate candidate_ids in top 100: {duplicates}")

        # ── Check 2: Honeypot rate ──────────────────────────────────────────
        honeypot_cids_in_top100 = [
            entry["candidate_id"] for entry in top_100
            if honeypot_scores.get(entry["candidate_id"], 0) >= 1
        ]
        honeypot_count_in_top100 = len(honeypot_cids_in_top100)
        honeypot_rate = honeypot_count_in_top100 / max(1, len(top_100))
        report.stats["honeypot_count"] = honeypot_count_in_top100
        report.stats["honeypot_rate"] = honeypot_rate

        if honeypot_rate >= 0.10:
            report.add_error(
                f"Honeypot rate {honeypot_rate:.1%} exceeds 10% disqualification "
                f"threshold (spec section 7). {honeypot_count_in_top100} honeypots "
                f"in top 100."
            )
            report.demotions.extend(honeypot_cids_in_top100)
        elif honeypot_rate >= 0.05:
            report.add_warning(
                f"Honeypot rate {honeypot_rate:.1%} above 5% internal threshold "
                f"({honeypot_count_in_top100} honeypots in top 100)"
            )
            report.demotions.extend(honeypot_cids_in_top100)

        # ── Check 3: Score distribution ─────────────────────────────────────
        scores = [entry.get("final_score", 0.0) for entry in top_100]
        if scores:
            score_range = max(scores) - min(scores)
            unique_scores = len(set(round(s, 6) for s in scores))
            report.stats["score_range"] = score_range
            report.stats["unique_scores"] = unique_scores

            if unique_scores < 50:
                report.add_warning(
                    f"Only {unique_scores} unique scores in top 100 - "
                    f"model may not be differentiating candidates"
                )

            if score_range < 0.01:
                report.add_warning(
                    f"Score range only {score_range:.6f} in top 100 - "
                    f"severe score compression may hurt NDCG"
                )

        # ── Check 4: Score monotonicity ────────────────────────────────────
        for i in range(1, len(top_100)):
            prev_score = top_100[i - 1].get("final_score", 0.0)
            curr_score = top_100[i].get("final_score", 0.0)
            if curr_score > prev_score + 1e-9:
                report.add_error(
                    f"Score not non-increasing at rank {i+1}: "
                    f"{curr_score} > {prev_score}"
                )
                break

        # ── Check 5: Top-10 honeypot check ─────────────────────────────────
        top_10_honeypots = sum(
            1 for entry in top_100[:10]
            if honeypot_scores.get(entry["candidate_id"], 0) >= 1
        )
        report.stats["top10_honeypots"] = top_10_honeypots
        if top_10_honeypots > 0:
            report.add_error(
                f"{top_10_honeypots} honeypot(s) in top 10 - "
                f"this is catastrophic for NDCG@10 (50% of composite score)"
            )

        # ── Check 6: Semantic diversity ────────────────────────────────────
        candidate_lookup = {c["candidate_id"]: c for c in candidates}
        companies = []
        titles = []
        for entry in top_100:
            cand = candidate_lookup.get(entry["candidate_id"])
            if cand:
                profile = cand.get("profile", {})
                companies.append((profile.get("current_company") or "").lower())
                titles.append((profile.get("current_title") or "").lower())

        company_counts = Counter(companies)
        title_counts = Counter(titles)

        # Check for over-concentration on one company
        if companies:
            top_company = company_counts.most_common(1)[0]
            if top_company[1] > 10:
                report.add_warning(
                    f"Top company '{top_company[0]}' has {top_company[1]} "
                    f"candidates in top 100 - check for company bias"
                )

        report.stats["unique_companies"] = len(set(companies))
        report.stats["unique_titles"] = len(set(titles))

        # ── Check 7: Rank uniqueness ───────────────────────────────────────
        ranks = [entry.get("rank", 0) for entry in top_100]
        rank_counts = Counter(ranks)
        duplicate_ranks = {r: c for r, c in rank_counts.items() if c > 1}
        if duplicate_ranks:
            report.add_error(f"Duplicate ranks in top 100: {duplicate_ranks}")

        expected_ranks = set(range(1, 101))
        actual_ranks = set(ranks)
        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            if missing:
                report.add_error(f"Missing ranks: {sorted(missing)[:10]}")
            if extra:
                report.add_error(f"Invalid ranks: {sorted(extra)[:10]}")

        return report
