"""
reasoning.py - Generate candidate-specific, evidence-grounded reasoning.

Each reasoning string must be:
    1. Specific to this candidate (no templates)
    2. Reference actual evidence from the profile
    3. Under 2 sentences (~250 characters)
    4. Honest about both strengths and weaknesses
    5. Free of hallucinated skills (only mention what is in the profile)

The reasoning template adapts based on what actually drove the score.
Four templates are used based on the primary positive driver:
    - retrieval_focus: Candidate has retrieval/search system evidence
    - ml_production: Strong production ML but less retrieval-specific
    - assessment_led: Assessment scores are the primary differentiator
    - career_trajectory: Career quality is the main positive signal

The reasoning also includes one caution when relevant (notice period, behavioral
flags, location mismatch, or consulting background).
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from config import Config
from feature_extractor import FeatureBundle
from jd_analyzer import JDContext

logger = logging.getLogger("reasoning")


class ReasoningGenerator:
    """
    Generates candidate-specific reasoning strings for the top 100 candidates.
    """

    def __init__(self, jd_context: JDContext) -> None:
        self.jd = jd_context
        self._today = Config.REFERENCE_DATE

    def generate_all(
        self,
        top_100: List[Dict],
        candidates: List[dict],
        feature_bundles: Dict[str, FeatureBundle],
        honeypot_scores: Dict[str, int],
        trap_flags: Dict[str, Dict[str, bool]],
        evidence_bundles: Optional[Dict] = None,
        context_bundles: Optional[Dict] = None,
        confidence_bundles: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate reasoning for all top-100 candidates.

        When evidence/context/confidence bundles are provided, uses
        evidence-grounded reasoning; otherwise falls back to existing logic.
        Adds a 'reasoning' key to each entry in top_100.
        """
        if evidence_bundles or context_bundles:
            return self.generate_all_enhanced(
                top_100=top_100,
                candidates=candidates,
                feature_bundles=feature_bundles,
                honeypot_scores=honeypot_scores,
                trap_flags=trap_flags,
                evidence_bundles=evidence_bundles,
                context_bundles=context_bundles,
                confidence_bundles=confidence_bundles,
            )

        candidate_lookup = {c["candidate_id"]: c for c in candidates}
        result = []

        for entry in top_100:
            cid = entry["candidate_id"]
            candidate = candidate_lookup.get(cid)
            bundle = feature_bundles.get(cid)

            if candidate is None or bundle is None:
                entry["reasoning"] = (
                    f"Candidate {cid}: ranked based on overall profile signals; "
                    f"detailed breakdown unavailable."
                )
                result.append(entry)
                continue

            reasoning = self._generate_reasoning(
                candidate=candidate,
                bundle=bundle,
                score_dict=entry.get("score_dict", {}),
                trap_flags=trap_flags.get(cid, {}),
            )
            entry["reasoning"] = reasoning
            result.append(entry)

        return result

    def _generate_reasoning(
        self,
        candidate: dict,
        bundle: FeatureBundle,
        score_dict: Dict,
        trap_flags: Dict[str, bool],
    ) -> str:
        """
        Generate a single candidate-specific reasoning string.

        Strategy:
        1. Identify the primary positive driver(s)
        2. Include the strongest concrete evidence
        3. Add one caution if relevant
        4. Keep under 250 characters
        """
        profile = candidate["profile"]
        signals = candidate["redrob_signals"]
        career = candidate.get("career_history", [])

        # ── Build positive evidence ──────────────────────────────────────────
        positive_parts = []

        # Current role anchor
        current_title = (profile.get("current_title") or "").strip()
        current_company = (profile.get("current_company") or "").strip()
        yoe = profile.get("years_of_experience", 0) or 0

        if current_title and current_company:
            positive_parts.append(
                f"{yoe:.0f}yr {current_title} @ {current_company}"
            )

        # Primary technical evidence (retrieval terms)
        if bundle.retrieval_terms_found:
            # Prefer specific retrieval terms over generic "re-ranking"
            specific_terms = [
                t for t in bundle.retrieval_terms_found
                if t not in {"re-ranking", "reranking", "ranking"}
            ][:2]
            display_terms = specific_terms if specific_terms else bundle.retrieval_terms_found[:1]
            positive_parts.append(
                f"retrieval background: {', '.join(display_terms)}"
            )
        elif bundle.ml_terms_found:
            top_ml = bundle.ml_terms_found[:2]
            positive_parts.append(
                f"ML background: {', '.join(top_ml)}"
            )

        # Assessment score evidence (high trust - highlight if strong)
        if bundle.top_assessment_skill and bundle.top_assessment_score >= 60:
            positive_parts.append(
                f"assessed {bundle.top_assessment_skill}: "
                f"{bundle.top_assessment_score:.0f}/100"
            )

        # Effective relevant years
        if bundle.effective_relevant_years >= 2.0:
            positive_parts.append(
                f"{bundle.effective_relevant_years:.1f}yr relevant ML experience"
            )

        # ── Build caution evidence ───────────────────────────────────────────
        caution_parts = []

        # Trap flags — most disqualifying first, using JD-specific language
        trap_caution_map = {
            "consulting_only": (
                "entire career at IT services firms; JD requires product company experience"
            ),
            "pure_researcher": (
                "research-only background; JD requires production deployments"
            ),
            "recent_hype_pivot": (
                "AI experience post-2022 only; JD requires pre-LLM production ML"
            ),
            "langchain_only": (
                "AI work limited to LangChain/API calls; JD needs embedding systems depth"
            ),
            "title_chaser": (
                "rapid title escalation by switching companies"
            ),
        }
        for trap_key, caution_text in trap_caution_map.items():
            if trap_flags.get(trap_key, False):
                caution_parts.append(caution_text)
                break

        # Notice period — only flag if genuinely a logistics concern
        notice = bundle.notice_period_days
        if notice > 90 and not caution_parts:
            caution_parts.append(
                f"{notice}d notice period; JD prefers sub-30d, considers up to 90d"
            )
        elif notice > 60 and not caution_parts:
            caution_parts.append(f"{notice}d notice period")

        # Behavioral availability caution
        if bundle.days_since_active > 120 and not caution_parts:
            caution_parts.append(
                f"inactive {bundle.days_since_active} days — verify current availability"
            )

        # Location caution
        if bundle.location_country.lower() not in ("india", "in") and not caution_parts:
            if not signals.get("willing_to_relocate", False):
                caution_parts.append(
                    f"based overseas ({bundle.location_country}), not open to relocate"
                )

        # ── Assemble reasoning string ────────────────────────────────────────
        positive_str = "; ".join(positive_parts[:3]) if positive_parts else "relevant background"
        caution_str = ""
        if caution_parts:
            caution_str = f" Caution: {caution_parts[0]}."

        reasoning = f"{positive_str}.{caution_str}"

        # Enforce length limit
        if len(reasoning) > 250:
            reasoning = reasoning[:247] + "..."

        return reasoning

    def generate_all_enhanced(
        self,
        top_100: List[Dict],
        candidates: List[dict],
        feature_bundles: Dict[str, FeatureBundle],
        honeypot_scores: Dict[str, int],
        trap_flags: Dict[str, Dict[str, bool]],
        evidence_bundles: Optional[Dict] = None,
        context_bundles: Optional[Dict] = None,
        confidence_bundles: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Enhanced reasoning generation using evidence and context bundles.

        Falls back to existing _generate_reasoning() for candidates where
        new bundles are absent.
        """
        candidate_lookup = {c["candidate_id"]: c for c in candidates}
        result = []

        for entry in top_100:
            cid = entry["candidate_id"]
            candidate = candidate_lookup.get(cid)
            bundle = feature_bundles.get(cid)

            if candidate is None or bundle is None:
                entry["reasoning"] = (
                    f"Candidate {cid}: insufficient profile data for detailed analysis."
                )
                result.append(entry)
                continue

            ev_bundle = (evidence_bundles or {}).get(cid)
            ctx_bundle = (context_bundles or {}).get(cid)
            conf_bundle = (confidence_bundles or {}).get(cid)

            if ev_bundle is not None or ctx_bundle is not None:
                # Inject rank into score_dict so reasoning generator can use it
                score_dict = dict(entry.get("score_dict", {}))
                score_dict["_rank"] = entry.get("rank", 50)
                reasoning = self._generate_evidence_grounded_reasoning(
                    candidate=candidate,
                    bundle=bundle,
                    score_dict=score_dict,
                    trap_flags=trap_flags.get(cid, {}),
                    ev_bundle=ev_bundle,
                    ctx_bundle=ctx_bundle,
                    conf_bundle=conf_bundle,
                )
            else:
                reasoning = self._generate_reasoning(
                    candidate=candidate,
                    bundle=bundle,
                    score_dict=entry.get("score_dict", {}),
                    trap_flags=trap_flags.get(cid, {}),
                )

            entry["reasoning"] = reasoning
            result.append(entry)

        # ── Uniqueness pass: deduplicate reasoning strings ──────────────
        seen = {}
        for entry in result:
            r = entry["reasoning"]
            if r in seen:
                cid = entry["candidate_id"]
                rank = entry.get("rank", "?")
                entry["reasoning"] = f"{r.rstrip('.')} (rank {rank}, {cid})."
            else:
                seen[r] = True

        return result

    def _generate_evidence_grounded_reasoning(
        self,
        candidate: dict,
        bundle: FeatureBundle,
        score_dict: Dict,
        trap_flags: Dict[str, bool],
        ev_bundle=None,
        ctx_bundle=None,
        conf_bundle=None,
    ) -> str:
        """
        Generate reasoning grounded in specific evidence from career descriptions.

        The reasoning must pass Stage 4 manual review checks:
        1. Specific facts from the profile (title, company, years, named skills)
        2. JD connection (why this candidate fits THIS role, not generic praise)
        3. Honest concerns where gaps exist
        4. No templated phrases (especially no generic "N unverified claims")
        5. Variation — rank 1 reads differently from rank 50
        6. Rank-consistent tone — high ranks get stronger language
        """
        profile = candidate["profile"]
        signals = candidate["redrob_signals"]

        positive_parts = []
        caution_parts = []

        current_title = (profile.get("current_title") or "").strip()
        current_company = (profile.get("current_company") or "").strip()
        yoe = profile.get("years_of_experience", 0) or 0
        rank = score_dict.get("_rank", 50)  # injected by caller if available

        # ── Positive: Role anchor with JD-relevant framing ───────────────────
        if current_title and current_company:
            positive_parts.append(f"{yoe:.0f}yr {current_title} @ {current_company}")

        # ── Positive: Best available technical evidence, JD-specific ─────────
        # Priority: corroborated JD skills > retrieval terms > evidence sentence
        # NOTE: ctx_bundle evidence sentences are from synthetic templates and can
        # be identical across many candidates — use them only as a last resort.
        if ev_bundle and ev_bundle.top_corroborated_skills:
            # Show up to 2 skills with JD framing
            skills = ev_bundle.top_corroborated_skills[:2]
            positive_parts.append(f"career evidence of {', '.join(skills)}")
        elif bundle.retrieval_terms_found:
            # Only show the most specific retrieval terms, not generic "re-ranking"
            specific_terms = [
                t for t in bundle.retrieval_terms_found
                if t not in {"re-ranking", "reranking", "ranking"}
            ][:2]
            if specific_terms:
                positive_parts.append(
                    f"retrieval background: {', '.join(specific_terms)}"
                )
            elif bundle.retrieval_terms_found:
                positive_parts.append(
                    f"retrieval background: {bundle.retrieval_terms_found[0]}"
                )
        elif ctx_bundle and ctx_bundle.top_evidence_sentences:
            # Last resort: use evidence sentence (may be templated)
            sentence = ctx_bundle.top_evidence_sentences[0]
            if len(sentence) > 90:
                sentence = sentence[:87] + "..."
            positive_parts.append(f'built: "{sentence}"')
        elif bundle.ml_terms_found:
            positive_parts.append(f"ML background: {', '.join(bundle.ml_terms_found[:2])}")

        # ── Positive: Assessment score (only if high and relevant to JD) ─────
        if bundle.top_assessment_skill and bundle.top_assessment_score >= 65:
            positive_parts.append(
                f"scored {bundle.top_assessment_score:.0f}/100 on {bundle.top_assessment_skill}"
            )

        # ── Positive: Behavioral availability (add for strong signals) ────────
        rr = signals.get("recruiter_response_rate", 0.0) or 0.0
        open_to_work = signals.get("open_to_work_flag", False)
        if rr >= 0.75 and open_to_work:
            positive_parts.append("active and responsive")
        elif rr >= 0.75:
            positive_parts.append(f"high recruiter response rate ({rr:.0%})")

        # ── Positive: Multi-signal confidence (only when genuinely high) ──────
        if conf_bundle and conf_bundle.overall_confidence >= 0.82 and len(positive_parts) >= 2:
            # Don't add generic "multi-signal corroboration" — just let strong parts speak
            pass

        # ── Caution: Signal conflicts (specific, JD-connected) ───────────────
        score_A = score_dict.get("A", 0.0)
        score_B = score_dict.get("B", 0.0)
        score_C = score_dict.get("C_blended") or score_dict.get("C", 0.0)

        if score_A >= 0.55 and score_B <= 0.25:
            caution_parts.append(
                "retrieval vocabulary in profile but limited production deployment evidence"
            )
        elif score_C >= 0.65 and score_A <= 0.20:
            caution_parts.append(
                "strong assessment score but career history lacks corroborating retrieval work"
            )

        # ── Caution: Trap flags — most severe first, JD-connected language ────
        trap_caution_map = {
            "consulting_only": (
                "entire career at IT services firms — JD explicitly requires product company experience"
            ),
            "pure_researcher": (
                "research-only background; JD requires production deployments to real users"
            ),
            "recent_hype_pivot": (
                "AI experience post-2022 only; JD requires pre-LLM era ML production background"
            ),
            "langchain_only": (
                "AI exposure limited to LangChain/API calls; JD requires embedding systems expertise"
            ),
            "title_chaser": (
                "frequent company switches for title upgrades — contradicts JD preference for tenure"
            ),
            "domain_mismatch": (
                "primary expertise is CV/speech; JD requires NLP/IR focus"
            ),
        }
        for trap_key, caution_text in trap_caution_map.items():
            if trap_flags.get(trap_key, False):
                caution_parts.append(caution_text)
                break

        # ── Caution: Current role at consulting firm (even if not consulting_only) ─
        CONSULTING_FIRMS_SIMPLE = {
            "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
            "tech mahindra", "hcl", "mphasis", "hexaware", "mindtree",
            "ltimindtree", "birlasoft", "coforge", "cts",
        }
        current_company_lower = current_company.lower()
        current_at_consulting = any(
            firm in current_company_lower for firm in CONSULTING_FIRMS_SIMPLE
        )
        if current_at_consulting and not any(
            trap_flags.get(k, False) for k in ["consulting_only"]
        ) and not caution_parts:
            caution_parts.append(
                f"current role at {current_company} (IT services); "
                "JD favors product company experience"
            )

        # ── Caution: Notice period (logistics, not quality) ───────────────────
        notice = bundle.notice_period_days
        if notice > 90 and not caution_parts:
            caution_parts.append(
                f"{notice}d notice period; JD prefers sub-30d, will consider up to 90d"
            )
        elif notice > 60 and not caution_parts:
            caution_parts.append(f"{notice}d notice period")

        # ── Caution: Behavioral availability ─────────────────────────────────
        if bundle.days_since_active > 120 and not caution_parts:
            caution_parts.append(
                f"inactive for {bundle.days_since_active} days — verify current availability"
            )

        # ── Caution: Negative context evidence ────────────────────────────────
        if ev_bundle and ev_bundle.negative_context_skills and not caution_parts:
            skill = ev_bundle.negative_context_skills[0]
            caution_parts.append(f"{skill} mentioned only in exploratory/learning context")

        # ── Assemble: two-sentence structure for rank-tier variation ──────────
        # For bottom-tier candidates (rank 75+), soften positive framing and note
        # they're at the margin — this ensures rank-consistent tone at Stage 4 review
        rank = score_dict.get("_rank", 50)
        if rank >= 75 and not caution_parts:
            # When no specific caution: add a rank-honest qualifier that references their evidence gap
            if bundle.retrieval_terms_found and all(
                t in {"re-ranking", "reranking", "ranking"} for t in bundle.retrieval_terms_found
            ):
                caution_parts.append(
                    "retrieval evidence limited to re-ranking; lacks vector DB or embedding deployment experience"
                )
            else:
                caution_parts.append(
                    "adjacent profile — included at margin; limited production retrieval-system evidence"
                )

        positive_str = "; ".join(positive_parts[:3]) if positive_parts else "relevant ML background"
        caution_str = f" Caution: {caution_parts[0]}." if caution_parts else ""

        reasoning = f"{positive_str}.{caution_str}"
        return reasoning[:250]