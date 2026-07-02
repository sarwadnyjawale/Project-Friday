"""
honeypot.py - Honeypot detection for synthetic impossible candidate profiles.

The dataset contains ~80 honeypot profiles with structurally impossible characteristics.
These are not "bad candidates" - they are data integrity violations that signal the
profile was synthetically corrupted and should not appear in any top results.

The detection is entirely rule-based and runs in O(N x K) where K is the number
of honeypot checks (constant). No ML required.

Trust calibration notes:
    These checks are HIGH CONFIDENCE. An inverted salary range or expert skill with
    duration=0 is not an ambiguous signal - it is a structural impossibility.
    The detection threshold is deliberately conservative: 2+ signals required for
    the heavy penalty. This avoids penalizing candidates with one data-entry error.
"""

import logging
import re
from datetime import date, datetime
from typing import Dict, List, Tuple

from config import Config

logger = logging.getLogger("honeypot")


# ─────────────────────────────────────────────────────────────────────────────
# Technology launch dates for timeline impossibility check
# These are the public GA/release dates - not private beta dates.
# ─────────────────────────────────────────────────────────────────────────────
TECHNOLOGY_LAUNCH_DATES = {
    "gpt-4": date(2023, 3, 1),
    "gpt4": date(2023, 3, 1),
    "langchain": date(2023, 1, 1),
    "llama 2": date(2023, 7, 1),
    "llama2": date(2023, 7, 1),
    "llama 3": date(2024, 4, 1),
    "llama3": date(2024, 4, 1),
    "claude": date(2023, 3, 1),
    "pinecone": date(2021, 9, 1),        # Pinecone GA September 2021
    "weaviate": date(2021, 1, 1),        # Weaviate open-source but GA 2021
    "qdrant": date(2021, 6, 1),          # Qdrant first stable release 2021
    "chatgpt": date(2022, 11, 30),       # ChatGPT public launch
    "gpt-3.5": date(2022, 11, 30),
    "stable diffusion": date(2022, 8, 1),
    "midjourney": date(2022, 7, 1),
    "bard": date(2023, 3, 21),           # Google Bard public launch
    "gemini": date(2023, 12, 6),         # Gemini public announcement
    "mixtral": date(2023, 12, 1),
    "mistral": date(2023, 9, 1),         # Mistral 7B September 2023
    "llama": date(2023, 2, 1),           # LLaMA-1 release
    "qlora": date(2023, 5, 1),           # QLoRA paper May 2023
    "dspy": date(2023, 10, 1),           # DSPy public release
    "crewai": date(2023, 12, 1),         # CrewAI launch
    "autogen": date(2023, 9, 1),         # Microsoft AutoGen
    "phi-2": date(2023, 12, 1),          # Microsoft Phi-2
    "groq": date(2024, 2, 1),            # Groq public API
    "openai o1": date(2024, 9, 1),       # OpenAI o1 reasoning model
    "devin": date(2024, 3, 1),           # Devin AI launch
}

# Tolerance for date calculations (months)
# Some candidates may round experience duration generously - allow 6 months tolerance
DURATION_TOLERANCE_MONTHS = 6

# Boilerplate phrases that indicate copy-pasted or template summaries
# These are specific enough to be diagnostic without being too aggressive
BOILERPLATE_DIAGNOSTIC_PHRASES = [
    "i've spent my career in marketing manager",
    "results-driven professional with x+ years",
    "dynamic professional with x years",
    "seeking opportunities to leverage my skills",
    "passionate about leveraging cutting-edge ai to transform",
    "adept at utilizing machine learning to drive business",
    # Marker for JD-paraphrase honeypots - verbatim JD language in summary
    "retrieval quality regression",  # Very specific JD language, unlikely in organic summaries
]


class HoneypotDetector:
    """
    Detects structural impossibilities that mark a profile as a synthetic honeypot.

    Returns a {candidate_id: signal_count} dict.
    Candidates with signal_count >= 2 receive the heavy penalty multiplier.
    Candidates with signal_count >= 3 are scored near zero.
    """

    def detect_all(self, candidates: List[dict]) -> Dict[str, int]:
        """
        Run honeypot detection on all candidates.

        Args:
            candidates: List of candidate dicts from the loader

        Returns:
            {candidate_id: count_of_honeypot_signals}
        """
        results: Dict[str, int] = {}
        today = Config.REFERENCE_DATE

        for candidate in candidates:
            cid = candidate["candidate_id"]
            try:
                count, signals_found = self._detect_candidate(candidate, today)
                results[cid] = count
                if count >= 2:
                    logger.debug(
                        f"HONEYPOT FLAG: {cid} - {count} signals: {signals_found}"
                    )
            except Exception as e:
                # Never crash - log and return 0 for this candidate
                logger.warning(
                    f"Honeypot detection error for {cid}: {e} - defaulting to 0"
                )
                results[cid] = 0

        detected_2plus = sum(1 for v in results.values() if v >= 2)
        detected_3plus = sum(1 for v in results.values() if v >= 3)
        logger.info(
            f"Honeypot detection complete: "
            f"{detected_2plus} with 2+ signals, {detected_3plus} with 3+ signals"
        )
        return results

    def _detect_candidate(
        self, candidate: dict, today: date
    ) -> Tuple[int, List[str]]:
        """
        Run all honeypot checks on a single candidate.

        Returns:
            Tuple of (signal_count, list_of_signal_names_found)
        """
        count = 0
        signals_found: List[str] = []

        # ── Check 1: Inverted salary range ──────────────────────────────────
        # min > max is a structural data integrity violation.
        # From the problem statement: ~30% of candidates have this in the dataset.
        # This is the most reliable single honeypot indicator.
        sal = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
        sal_min = sal.get("min", 0)
        sal_max = sal.get("max", 0)
        if sal_max > 0 and sal_min > sal_max:
            count += 1
            signals_found.append("inverted_salary_range")

        # ── Check 2: YoE exceeds career history ─────────────────────────────
        # Self-reported YoE should not exceed 130% of verifiable career history.
        # The 30% tolerance handles rounding, part-time work, and freelancing.
        career_history = candidate.get("career_history", [])
        if career_history:
            total_career_months = sum(
                max(0, r.get("duration_months", 0)) for r in career_history
            )
            claimed_years = float(
                candidate["profile"].get("years_of_experience", 0) or 0
            )
            verifiable_years = total_career_months / 12.0
            if claimed_years > 0 and verifiable_years > 0:
                if claimed_years > verifiable_years * 1.35:
                    count += 1
                    signals_found.append(
                        f"yoe_inflated({claimed_years:.1f}yr_vs_{verifiable_years:.1f}yr)"
                    )

        # ── Check 3: Expert skill with zero duration ─────────────────────────
        # "Expert" proficiency in a skill used for 0 months is structurally impossible.
        for skill in candidate.get("skills", []):
            if (
                skill.get("proficiency", "").lower() == "expert"
                and skill.get("duration_months", 1) == 0
            ):
                count += 1
                signals_found.append(f"expert_zero_duration({skill.get('name', 'unknown')})")
                break  # One is enough to flag

        # ── Check 4: Technology timeline impossibility ───────────────────────
        # Skill duration implies usage before the technology existed.
        today_date = today  # Uses Config.REFERENCE_DATE for determinism
        for skill in candidate.get("skills", []):
            skill_name_lower = (skill.get("name") or "").lower()
            duration_months = skill.get("duration_months", 0) or 0
            if duration_months <= 0:
                continue
            for tech_name, launch_date in TECHNOLOGY_LAUNCH_DATES.items():
                if tech_name in skill_name_lower:
                    max_possible_months = (
                        (today_date.year - launch_date.year) * 12
                        + (today_date.month - launch_date.month)
                        + DURATION_TOLERANCE_MONTHS
                    )
                    if duration_months > max_possible_months:
                        count += 1
                        signals_found.append(
                            f"timeline_impossible({skill.get('name')}:"
                            f"{duration_months}mo_vs_max_{max_possible_months}mo)"
                        )
                        break  # One impossible skill is enough

        # ── Check 5: Skill count + proficiency impossibility ─────────────────
        # More than 40 skills with a majority at advanced/expert level suggests
        # an impossible breadth - no engineer can be expert in 30+ distinct tech domains.
        skills = candidate.get("skills", [])
        if len(skills) > 40:
            high_prof_count = sum(
                1 for s in skills
                if s.get("proficiency", "").lower() in ("advanced", "expert")
            )
            if high_prof_count > 25:
                count += 1
                signals_found.append(
                    f"impossible_skill_breadth({len(skills)}_skills,"
                    f"{high_prof_count}_advanced+)"
                )

        # ── Check 6: Career description domain mismatch ──────────────────────
        # Descriptions that reference non-software domains at software companies.
        # This catches "Managing brand identity" at a data analytics firm, etc.
        non_software_patterns = [
            r"\b(mechanical subsystem|structural design|civil engineering)\b",
            r"\b(brand identity|marketing campaign|advertising spend)\b",
            r"\b(inventory management|supply chain logistics|warehouse)\b",
            r"\b(tax filing|audit report|financial statement preparation)\b",
        ]
        all_descriptions = " ".join(
            (r.get("description") or "") for r in career_history
        ).lower()
        profile_industry = (
            candidate["profile"].get("current_industry") or ""
        ).lower()

        # Only flag if the mismatch is clear: non-software description + software industry claim
        software_industry_indicators = {"software", "technology", "tech", "saas", "ai", "ml"}
        is_software_industry = any(
            ind in profile_industry for ind in software_industry_indicators
        )
        if is_software_industry:
            for pattern in non_software_patterns:
                if re.search(pattern, all_descriptions, re.IGNORECASE):
                    count += 1
                    signals_found.append("description_domain_mismatch")
                    break  # One pattern match is enough

        # ── Check 7: Summary copies JD language verbatim ─────────────────────
        summary_lower = (candidate["profile"].get("summary") or "").lower()
        for phrase in BOILERPLATE_DIAGNOSTIC_PHRASES:
            if phrase.lower() in summary_lower:
                count += 1
                signals_found.append(f"summary_boilerplate({phrase[:30]}...)")
                break  # Don't stack-penalize multiple boilerplate phrases

        # ── Check 8: Overlapping full-time roles ─────────────────────────────
        # Two roles with overlapping dates and neither marked as part-time/contract
        # is a structural impossibility (for full-time employment).
        if self._has_overlapping_fulltime_roles(career_history):
            count += 1
            signals_found.append("overlapping_fulltime_roles")

        # ── Check 9: Assessment score impossibility ──────────────────────────
        # Assessment score > 100 is structurally impossible (scale is 0-100).
        assessment_scores = candidate.get("redrob_signals", {}).get(
            "skill_assessment_scores", {}
        ) or {}
        for skill_name, score_val in assessment_scores.items():
            try:
                if float(score_val) > 100:
                    count += 1
                    signals_found.append(f"impossible_assessment({skill_name}:{score_val})")
                    break
            except (ValueError, TypeError):
                continue

        # ── Check 10: Salary range absurdity ─────────────────────────────────
        # Salary min or max of 0 or negative = data corruption.
        if sal_min < 0 or sal_max < 0:
            count += 1
            signals_found.append(f"negative_salary(min={sal_min},max={sal_max})")

        # ── Check 11: Future career start dates ──────────────────────────────
        # Career roles starting after REFERENCE_DATE = fabricated timeline.
        for role in career_history:
            try:
                start_str = role.get("start_date") or ""
                if not start_str:
                    continue
                role_start = datetime.strptime(start_str[:7], "%Y-%m").date()
                if role_start > today:
                    count += 1
                    signals_found.append(f"future_start_date({start_str})")
                    break
            except (ValueError, TypeError):
                continue

        return count, signals_found

    def _has_overlapping_fulltime_roles(self, career_history: List[dict]) -> bool:
        """
        Check for impossible simultaneous full-time employment.

        Two roles overlapping by more than 2 months suggests fabricated timeline.
        We allow 2 months of overlap for transition periods and notice periods.
        """
        if len(career_history) < 2:
            return False

        # Parse role date ranges
        date_ranges: List[Tuple[date, date]] = []
        for role in career_history:
            try:
                start_str = role.get("start_date") or ""
                end_str = role.get("end_date") or ""
                if not start_str:
                    continue
                start = datetime.strptime(start_str[:7], "%Y-%m").date()
                end = (
                    Config.REFERENCE_DATE
                    if not end_str
                    else datetime.strptime(end_str[:7], "%Y-%m").date()
                )
                date_ranges.append((start, end))
            except (ValueError, TypeError):
                continue

        if len(date_ranges) < 2:
            return False

        # Sort by start date
        date_ranges.sort(key=lambda x: x[0])

        # Check for overlaps > 2 months
        for i in range(len(date_ranges) - 1):
            end_a = date_ranges[i][1]
            start_b = date_ranges[i + 1][0]
            if start_b < end_a:
                # Overlap exists - compute duration
                overlap_months = (
                    (end_a.year - start_b.year) * 12 + (end_a.month - start_b.month)
                )
                if overlap_months > 2:
                    return True

        return False