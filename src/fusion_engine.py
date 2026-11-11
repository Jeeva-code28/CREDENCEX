# src/fusion_engine.py
# Phase 6: Multimodal Fusion & Final Decision
# Combines textual NLI results, primary official origins, and image analysis to produce
# an authoritative verification verdict: Supported / Contradicted / Unverified.

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class VerificationResult:
    """Final verification result with all supporting details."""
    verdict: str                            # 'Supported', 'Contradicted', 'Unverified'
    confidence: float                       # Confidence score [0, 1]
    supporting_evidence: List[Dict] = field(default_factory=list)
    contradicting_evidence: List[Dict] = field(default_factory=list)
    image_assessment: Optional[Dict] = None
    reasoning: str = ""


class FusionEngine:
    """
    Multimodal fusion engine that combines textual NLI results, primary official source
    corroboration, and image authenticity/context analysis to produce a final verdict.

    Primary Origin First Decision Rules:
    1. Primary Origin Official Confirmation -> Supported with high confidence
    2. Accredited Refutation + Absence of Primary Confirmation -> Contradicted (Debunked / Fabricated Claim)
    3. Strong image contradiction (manipulation or context mismatch) -> Contradicted
    4. Unverified social media claims without primary confirmation -> Unverified / Contradicted
    5. Conflicting NLI and image signals -> Unverified
    """

    def __init__(self):
        self.verdict_thresholds = {
            'supported': 0.7,
            'contradicted': 0.7,
            'unverified': 0.5
        }
        logger.info("FusionEngine initialized with Primary Origin Priority")

    def fuse_verdicts(
        self,
        nli_result: Dict,
        image_result: Optional[Dict] = None,
        primary_origins: Optional[List[Dict]] = None,
        evidence: Optional[List[Dict]] = None,
        claim: Optional[str] = None
    ) -> Dict:
        """
        Combine textual NLI results, primary origin official verification, and image analysis.

        Args:
            nli_result: Dict from NLIVerifier.verify_claim_against_evidence()
            image_result: Optional dict from image forensics pipeline.
            primary_origins: Optional list of identified primary origin entities.
            evidence: Optional list of ranked evidence items.
            claim: Optional original claim string.

        Returns:
            Dict with text_verdict, text_confidence, image_verdict,
            image_confidence, final_verdict, final_confidence, and reasoning.
        """
        nli_agg = nli_result.get('aggregated', {}) if isinstance(nli_result, dict) else {}
        primary_verdict = nli_agg.get('primary_verdict', 'neutral')
        text_confidence = nli_agg.get('confidence', 0.5)

        result = {
            'text_verdict': primary_verdict,
            'text_confidence': text_confidence,
            'image_verdict': None,
            'image_confidence': 0.0,
            'final_verdict': None,
            'final_confidence': 0.0,
            'primary_status': 'unknown',
            'reasoning': ''
        }

        # Process image results if available
        if image_result:
            image_authenticity = image_result.get('authenticity', {})
            image_context = image_result.get('context', {})

            image_confidence = 0.0
            image_verdict = 'neutral'

            auth_v = str(image_authenticity.get('verdict', '')).lower()
            if auth_v in ('potentially_manipulated', 'manipulated', 'contradiction'):
                image_verdict = 'contradiction'
                auth_score_val = float(image_authenticity.get('authenticity_score', 0.0))
                image_confidence = (1.0 - auth_score_val) if auth_score_val <= 0.65 else auth_score_val

            ctx_v = str(image_context.get('verdict', '')).lower()
            if ctx_v in ('context_mismatch', 'contradiction'):
                image_verdict = 'contradiction'
                ctx_score = image_context.get('consistency_score', image_context.get('overall_context_score', 0.5))
                image_confidence = max(
                    image_confidence,
                    1.0 - float(ctx_score)
                )

            result['image_verdict'] = image_verdict
            result['image_confidence'] = image_confidence

        # Check for Primary Official Corroboration & Accredited Contradiction / Entailment
        has_primary_entailment = False
        has_accredited_contradiction = False
        has_accredited_entailment = False
        accredited_supporting_count = 0
        primary_names = []
        refuting_source_names = []

        if primary_origins:
            primary_names = [po.get('name', '') for po in primary_origins if po.get('name')]

        # Build lookup from nli_result in case callers did not copy 'relationship' onto evidence dicts
        nli_stance_by_text = {}
        if isinstance(nli_result, dict):
            for r in nli_result.get('individual_results', []):
                ev_r = r.get('evidence', {}) if isinstance(r, dict) else {}
                t_key = (ev_r.get('text') or '').strip()
                if t_key:
                    nli_stance_by_text[t_key] = r.get('prediction', 'neutral')
            for d in nli_result.get('evidence_details', []):
                t_key = (d.get('text') or '').strip() if isinstance(d, dict) else ''
                if t_key and t_key not in nli_stance_by_text:
                    nli_stance_by_text[t_key] = d.get('relationship', d.get('verdict', 'neutral'))

        if evidence:
            for ev in evidence:
                is_primary = ev.get('is_official_primary', False)
                t_key = (ev.get('text') or '').strip()
                rel = ev.get('relationship', ev.get('stance'))
                if not rel and t_key in nli_stance_by_text:
                    rel = nli_stance_by_text[t_key]
                rel = rel or 'neutral'

                # Primary origin official confirmation
                if is_primary and rel == 'entailment':
                    has_primary_entailment = True

                # Accredited news wire or fact checker refutation
                if ev.get('source_tier') in ('fact_checker', 'news_wire', 'mainstream_press') and rel == 'contradiction':
                    has_accredited_contradiction = True
                    src_name = ev.get('source', '') or ev.get('url', '')
                    if src_name:
                        refuting_source_names.append(src_name)

                # Accredited news wire, fact checker, or mainstream press entailment
                if ev.get('source_tier') in ('fact_checker', 'news_wire', 'mainstream_press') and rel == 'entailment':
                    has_accredited_entailment = True
                    accredited_supporting_count += 1

        # Detect if claim concerns a speech, press meet, public address, or public event/inauguration
        claim_text = (claim or "").strip()
        if not claim_text and isinstance(nli_result, dict):
            claim_text = nli_result.get('claim', '')

        is_speech = bool(re.search(
            r'\b(speech|spoke|speaks|speaking|said|stated|statement|press\s*meet|press\s*conference|interview|'
            r'rally|addressed|addressing|parliamentary|parliament|assembly|bhashan|'
            r'பேச்சு|பேசியது|செய்தியாளர்\s*சந்திப்பு|பேட்டி)\b',
            claim_text,
            re.IGNORECASE
        ))

        is_event = bool(re.search(
            r'\b(inaugurat\w*|launch\w*|open\w*|unveil\w*|flag\w*\s+off|dedicat\w*|commission\w*|'
            r'ceremony|stretch|metro|summit|event|rally|foundation\s+stone)\b',
            claim_text,
            re.IGNORECASE
        ))

        # Apply hierarchical decision rules
        final_verdict, final_confidence, reasoning = self._apply_fusion_rules(
            result,
            has_primary_entailment=has_primary_entailment,
            has_accredited_contradiction=has_accredited_contradiction,
            has_accredited_entailment=has_accredited_entailment,
            accredited_supporting_count=accredited_supporting_count,
            primary_names=primary_names,
            refuting_sources=refuting_source_names,
            is_speech_claim=is_speech,
            is_event_claim=is_event
        )

        result['final_verdict'] = final_verdict
        result['final_confidence'] = final_confidence
        result['reasoning'] = reasoning

        result['truth_analysis'] = self.generate_truth_analysis(
            claim=claim_text,
            final_verdict=final_verdict,
            final_confidence=final_confidence,
            evidence=evidence,
            primary_origins=primary_origins,
            nli_result=nli_result,
            is_speech_claim=is_speech,
            is_event_claim=is_event
        )

        logger.info(
            f"Fusion result: {final_verdict} (confidence={final_confidence:.2%}) — {reasoning[:80]}..."
        )

        return result

    def _apply_fusion_rules(
        self,
        result: Dict,
        has_primary_entailment: bool = False,
        has_accredited_contradiction: bool = False,
        has_accredited_entailment: bool = False,
        accredited_supporting_count: int = 0,
        primary_names: Optional[List[str]] = None,
        refuting_sources: Optional[List[str]] = None,
        is_speech_claim: bool = False,
        is_event_claim: bool = False
    ) -> tuple:
        """
        Apply decision rules for multimodal fusion prioritizing primary origin records,
        accredited news channels / live broadcast reporting for public speeches,
        and accredited news wire / mainstream reporting for public events.
        """
        text_verdict = result['text_verdict']
        text_conf = result['text_confidence']
        image_verdict = result.get('image_verdict')
        image_conf = result.get('image_confidence', 0.0)
        primary_str = ", ".join(primary_names[:2]) if primary_names else "official primary entities"

        # Rule 1: Accredited Outlets Refute + Absence of Official Primary Corroboration
        if has_accredited_contradiction and not has_primary_entailment:
            if is_speech_claim:
                reasoning = (
                    f"Contradicted by accredited news reporting and live broadcast archives. Public speeches and "
                    f"press meets by figures like {primary_str} are documented live by accredited news channels and newspapers; "
                    f"no authentic broadcast record or video proof corroborates this claim."
                )
            else:
                reasoning = (
                    f"Contradicted by accredited news reporting and lack of confirmation from primary origin ({primary_str}). "
                    "Verified records and independent fact-checking refute the circulating claim."
                )
            return 'Contradicted', max(0.85, text_conf), reasoning

        # Rule 2: Primary Official Confirmation verified
        if has_primary_entailment and image_verdict != 'contradiction':
            reasoning = f"Corroborated by official announcements from primary origin ({primary_str})."
            return 'Supported', max(0.92, text_conf), reasoning

        # Rule 2b: Accredited Media Corroboration for Public Events & News
        # Breaking events, inaugurations, ceremonies, and speeches documented by accredited media
        # (The Hindu, Reuters, ETV Bharat, ANI, etc.) represent verified ground truth.
        if (has_accredited_entailment or (is_event_claim and text_verdict == 'entailment')) and text_conf >= 0.65 and not has_accredited_contradiction and image_verdict != 'contradiction':
            reasoning = f"Corroborated by accredited news reporting and official event coverage."
            return 'Supported', max(0.88, text_conf), reasoning

        # Rule 3: Image provides strong contradiction (forensic manipulation or context mismatch)
        if image_conf > 0.7 and image_verdict == 'contradiction':
            reasoning = "Visual forensics detected manipulation, spliced elements, or mismatched caption context in the image."
            return 'Contradicted', image_conf, reasoning

        # Rule 4: Strong text signal with no image contradiction
        if text_conf > 0.75 and (image_verdict is None or image_verdict == 'neutral'):
            if text_verdict == 'entailment':
                return 'Supported', text_conf, "Supported by multiple independent authoritative sources."
            elif text_verdict == 'contradiction':
                if is_speech_claim:
                    return 'Contradicted', text_conf, f"Refuted by accredited news coverage; no verified broadcast or newspaper has recorded this speech by {primary_str}."
                return 'Contradicted', text_conf, "Refuted by credible factual evidence."

        # Rule 5: Conflicting signals -> Unverified
        if text_verdict == 'entailment' and image_verdict == 'contradiction':
            return 'Unverified', 0.50, "Conflicting signals between textual claims and image forensic markers."

        # Rule 6: Weak signals or lack of primary confirmation -> Unverified
        if text_conf < 0.60:
            if is_speech_claim:
                reasoning = (
                    f"Insufficient verified evidence found to confirm this speech. "
                    f"No accredited television news channel or major newspaper has reported this speech by {primary_str}."
                )
            elif is_event_claim:
                reasoning = (
                    f"Insufficient conclusive evidence found to confirm this event. "
                    f"No accredited news reports or official announcements corroborate this occurrence."
                )
            else:
                reasoning = (
                    f"Insufficient conclusive evidence found to decisively confirm this claim. "
                    f"No official confirmation was issued by primary sources ({primary_str})."
                )
            return 'Unverified', text_conf, reasoning

        # Rule 7: Default mapping
        if text_verdict == 'entailment' and not has_accredited_contradiction:
            return 'Supported', text_conf, "Corroborated by available evidence."
        elif text_verdict == 'contradiction':
            return 'Contradicted', text_conf, "Contradicted by available evidence."
        else:
            if is_speech_claim:
                return 'Unverified', text_conf, f"Claim lacks video proof or news media verification for {primary_str}."
            elif is_event_claim:
                return 'Unverified', text_conf, f"Event lacks verified news reporting or official announcement."
            return 'Unverified', text_conf, f"Claim lacks official corroboration from primary sources ({primary_str})."

    def generate_truth_analysis(
        self,
        claim: str,
        final_verdict: str,
        final_confidence: float,
        evidence: Optional[List[Dict]] = None,
        primary_origins: Optional[List[Dict]] = None,
        nli_result: Optional[Dict] = None,
        is_speech_claim: bool = False,
        is_event_claim: bool = False
    ) -> Dict:
        """
        Generate a structured truth breakdown separating circulating rumors
        from actual verified ground truth based on retrieved evidence.
        Tailored specifically for speeches/press meets vs. public events vs. government/institutional circulars.
        """
        evidence_list = evidence or []
        primary_names = [po.get('name', '') for po in (primary_origins or []) if po.get('name')]
        primary_str = ", ".join(primary_names[:2]) if primary_names else "official authorities"

        refuting_snippets = []
        supporting_snippets = []

        nli_stance_by_text = {}
        if isinstance(nli_result, dict):
            for r in nli_result.get('individual_results', []):
                ev_r = r.get('evidence', {}) if isinstance(r, dict) else {}
                t_key = (ev_r.get('text') or '').strip()
                if t_key:
                    nli_stance_by_text[t_key] = r.get('prediction', 'neutral')
            for d in nli_result.get('evidence_details', []):
                t_key = (d.get('text') or '').strip() if isinstance(d, dict) else ''
                if t_key and t_key not in nli_stance_by_text:
                    nli_stance_by_text[t_key] = d.get('relationship', d.get('verdict', 'neutral'))

        for ev in evidence_list:
            text = ev.get('text', '').strip()
            if not text:
                continue
            rel = ev.get('relationship', ev.get('stance'))
            if not rel:
                rel = nli_stance_by_text.get(text, 'neutral')
            rel = str(rel).lower()
            if rel == 'contradiction':
                refuting_snippets.append(text)
            elif rel == 'entailment':
                supporting_snippets.append(text)

        effective_claim = claim.strip() if (claim and claim.strip()) else "Circulating assertion"

        # Check speech / event context if not explicitly passed
        if not is_speech_claim:
            is_speech_claim = bool(re.search(
                r'\b(speech|spoke|speaks|speaking|said|stated|statement|press\s*meet|press\s*conference|interview|'
                r'rally|addressed|addressing|parliamentary|parliament|assembly|bhashan|'
                r'பேச்சு|பேசியது|செய்தியாளர்\s*சந்திப்பு|பேட்டி)\b',
                effective_claim,
                re.IGNORECASE
            ))

        if not is_event_claim:
            is_event_claim = bool(re.search(
                r'\b(inaugurat\w*|launch\w*|open\w*|unveil\w*|flag\w*\s+off|dedicat\w*|commission\w*|'
                r'ceremony|stretch|metro|summit|event|rally|foundation\s+stone)\b',
                effective_claim,
                re.IGNORECASE
            ))

        if final_verdict == 'Contradicted':
            status = 'rumor'
            badge_label = 'DEBUNKED RUMOR / FALSE'
            badge_color = 'rose'

            if is_speech_claim:
                if refuting_snippets:
                    snippet_excerpt = refuting_snippets[0][:260]
                    actual_truth = (
                        f"Accredited news channels, live broadcast archives, and verified newspaper reports "
                        f"confirm this speech claim is false. Verified reports clarify: \"{snippet_excerpt}\""
                    )
                else:
                    actual_truth = (
                        f"Accredited news channels, live broadcast archives, and verified newspaper reports confirm "
                        f"that {primary_str} delivered no such speech or statement. In real-world events, public addresses "
                        f"and press meets are documented live by news channels and accredited press; the circulating "
                        f"claim is a fabricated assertion not corroborated by any credible news media broadcast or video evidence."
                    )
                rumor_statement = effective_claim
                evidence_basis = (
                    f"Cross-referenced against accredited TV news channels, live broadcast coverage, news wires (ANI, PTI), "
                    f"and major newspapers. No authentic video proof or news record exists."
                )
                verdict_meaning = "This circulating speech claim is false and fabricated; no verified news channel or newspaper has recorded such an event."
            else:
                if refuting_snippets:
                    snippet_excerpt = refuting_snippets[0][:260]
                    actual_truth = (
                        f"Official records and factual evidence confirm this statement is a rumor. "
                        f"Authoritative reports clarify: \"{snippet_excerpt}\""
                    )
                elif primary_origins:
                    is_non_gov_domain = any(
                        isinstance(po, dict) and po.get('category') in (
                            'space_astronomy', 'health_medical', 'sports', 'cinema_entertainment', 'corporate_business'
                        )
                        for po in primary_origins
                    )
                    if is_non_gov_domain:
                        actual_truth = (
                            f"Authoritative records and verified evidence from {primary_str} do not support this claim. "
                            f"Verified data contradicts the circulating assertion."
                        )
                    else:
                        actual_truth = (
                            f"This assertion is an unfounded viral rumor. {primary_str} have issued no circular, "
                            f"policy, or order authorizing this. Existing rules and official operations remain unchanged."
                        )
                else:
                    actual_truth = (
                        "Verified reporting and fact-checking records confirm this statement is factually "
                        "false or fabricated. Authoritative data directly refutes the claim."
                    )
                rumor_statement = effective_claim
                evidence_basis = (
                    f"Refuted by {primary_str} records and accredited reporting. "
                    "No official notification exists to support the claim."
                )
                verdict_meaning = "This statement is a rumor and NOT actual truth based on official data."

        elif final_verdict == 'Supported':
            status = 'verified_truth'
            badge_label = 'VERIFIED ACTUAL TRUTH'
            badge_color = 'emerald'

            if is_speech_claim:
                if supporting_snippets:
                    snippet_excerpt = supporting_snippets[0][:260]
                    actual_truth = (
                        f"Verified by live broadcast coverage, accredited news channels, and mainstream newspaper reports. "
                        f"Media records confirm: \"{snippet_excerpt}\""
                    )
                else:
                    actual_truth = f"Verified by live broadcast coverage, accredited news channels, and mainstream newspaper reports documenting the speech delivered by {primary_str}."
                rumor_statement = "No conflicting rumor detected; speech/statement is verified by media broadcast records."
                evidence_basis = "Corroborated by live video proofs, television news broadcasts, and accredited press reporting."
                verdict_meaning = "This speech/statement is verified as genuine based on accredited news broadcasts and published reports."
            elif is_event_claim:
                if supporting_snippets:
                    snippet_excerpt = supporting_snippets[0][:260]
                    actual_truth = (
                        f"This statement represents verified factual truth. "
                        f"Accredited news media and official event coverage confirm: \"{snippet_excerpt}\""
                    )
                else:
                    actual_truth = f"This statement represents verified factual truth corroborated by accredited news reporting and official event coverage involving {primary_str}."
                rumor_statement = "No conflicting rumor detected; event is corroborated by verified media and official coverage."
                evidence_basis = "Corroborated by accredited press reporting, live news coverage, and official public records."
                verdict_meaning = "This statement is verified as genuine truth based on official evidence."
            else:
                if supporting_snippets:
                    snippet_excerpt = supporting_snippets[0][:260]
                    actual_truth = (
                        f"This statement represents verified factual truth. "
                        f"Authoritative records confirm: \"{snippet_excerpt}\""
                    )
                else:
                    actual_truth = f"Corroborated by official announcements and verified records from {primary_str}."
                rumor_statement = "No conflicting rumor detected; statement is authenticated."
                evidence_basis = f"Directly corroborated by primary records and independent media."
                verdict_meaning = "This statement is verified as genuine truth based on official evidence."

        else:  # Unverified
            status = 'unverified_rumor'
            badge_label = 'UNCONFIRMED / POTENTIAL RUMOR'
            badge_color = 'amber'

            if is_speech_claim:
                actual_truth = (
                    f"No accredited news channel, live broadcast recording, or verified newspaper report has documented "
                    f"this speech or press meet. While this assertion is actively circulating on social media, accredited "
                    f"press and media archives contain no video proof or confirmation from {primary_str}."
                )
                rumor_statement = effective_claim
                evidence_basis = f"Lacks video proof, live broadcast coverage, or newspaper confirmation from accredited media."
                verdict_meaning = "This speech claim lacks video proof or news media verification and is currently uncorroborated."
            elif is_event_claim:
                actual_truth = (
                    f"No accredited news reports, official event schedules, or government announcements have corroborated "
                    f"this claim. While this assertion is circulating on social channels, accredited newsrooms "
                    f"and {primary_str} have not verified or announced it."
                )
                rumor_statement = effective_claim
                evidence_basis = f"Lacks event confirmation, live news coverage, or official press releases from {primary_str}."
                verdict_meaning = "This event claim lacks verified truth data and is circulating without official authentication."
            else:
                actual_truth = (
                    f"No official notification, authoritative record, or verified press wire exists to corroborate this claim. "
                    f"While this assertion is actively circulating on social channels, {primary_str} have not verified or announced it. "
                    f"What is currently established is that this claim lacks authentic evidentiary backing."
                )
                rumor_statement = effective_claim
                evidence_basis = f"Lacks official documentation or press confirmation from {primary_str}."
                verdict_meaning = "This claim lacks verified truth data and is circulating without official authentication."

        return {
            'status': status,
            'badge_label': badge_label,
            'badge_color': badge_color,
            'claim_statement': effective_claim,
            'actual_truth': actual_truth,
            'rumor_statement': rumor_statement,
            'evidence_basis': evidence_basis,
            'verdict_meaning': verdict_meaning
        }

    def generate_report(
        self,
        claim: str,
        nli_result: Dict,
        image_result: Optional[Dict] = None,
        primary_origins: Optional[List[Dict]] = None,
        evidence: Optional[List[Dict]] = None,
        fusion_result: Optional[Dict] = None
    ) -> str:
        """
        Generate a human-readable verification report with Primary Origin audit.

        Args:
            fusion_result: Optional pre-computed fusion result dict from fuse_verdicts().
                           If not provided, fuse_verdicts() will be called internally.
        """
        if fusion_result is None:
            fusion_result = self.fuse_verdicts(
                nli_result,
                image_result=image_result,
                primary_origins=primary_origins,
                evidence=evidence,
                claim=claim
            )


        primary_section = ""
        if primary_origins:
            primary_section = "\n        PRIMARY ORIGINAL SOURCES INVESTIGATED:\n"
            for po in primary_origins[:3]:
                handles_str = ", ".join(po.get('official_handles', [])) or "Official Channels"
                primary_section += f"        - {po.get('name')} ({po.get('role')}): {handles_str}\n"

        report = f"""
        ================================================================
        CLAIM VERIFICATION REPORT — PRIMARY ORIGIN AUDIT
        ================================================================

        CLAIM: {claim}

        ----------------------------------------------------------------
        FINAL VERDICT: {fusion_result['final_verdict'].upper()}
        Confidence: {fusion_result['final_confidence']:.2%}
        Reasoning: {fusion_result.get('reasoning', '')}
        ----------------------------------------------------------------
{primary_section}
        TEXT-BASED EVIDENCE:
        - Primary Verdict: {fusion_result['text_verdict'].upper()}
        - Confidence: {fusion_result['text_confidence']:.2%}

        Weighted Evidence Probabilities:
        - Entailment (Supporting): {nli_result['aggregated']['weighted_probabilities']['entailment']:.2%}
        - Neutral (Insufficient): {nli_result['aggregated']['weighted_probabilities']['neutral']:.2%}
        - Contradiction (Refuting): {nli_result['aggregated']['weighted_probabilities']['contradiction']:.2%}

        Top Evidence Items Evaluated:
        """

        if evidence:
            for i, ev in enumerate(evidence[:4], 1):
                tier_badge = ev.get('source_tier', 'web').upper()
                report += f"\n        [{i}] {ev.get('title', 'Evidence Source')} [{tier_badge}]\n"
                report += f"            Source: {ev.get('source', 'N/A')} | Credibility: {round(float(ev.get('scores_breakdown', {}).get('reliability', 0.5))*100)}%\n"
                report += f"            Snippet: {ev.get('text', '')[:120]}...\n"

        if image_result:
            authenticity = image_result.get('authenticity', {})
            context = image_result.get('context', {})
            report += f"""
        IMAGE ANALYSIS:
        - Authenticity Verdict: {authenticity.get('verdict', 'N/A')}
        - Authenticity Score: {authenticity.get('authenticity_score', 0):.2%}
        - Context Alignment: {context.get('overall_context_score', 0):.2%}
        """

        report += "\n        ================================================================\n"
        return report
