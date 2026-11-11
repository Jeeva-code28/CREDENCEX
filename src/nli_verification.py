# src/nli_verification.py
# Phase 4: Claim-Evidence NLI & Verification
# Uses DeBERTa-v3 for Natural Language Inference to classify
# claim-evidence relationships as entailment, neutral, or contradiction.

from typing import List, Dict, Optional

import re
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    HAS_TRANSFORMERS = False

from src.utils import setup_logger, timer

logger = setup_logger(__name__)

NLI_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'as', 'it', 'its', 'from', 'that', 'this', 'these', 'those', 'here',
    'there', 'their', 'they', 'he', 'she', 'we', 'you', 'i', 'into', 'over', 'under',
    'after', 'before', 'between', 'during', 'out', 'up', 'down', 'about'
}

NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', 'without', 'nothing', 'nowhere'
}

REFUTATION_CUES = {
    'false', 'fake', 'hoax', 'misleading', 'debunk', 'debunked', 'fabricated',
    'incorrect', 'untrue', 'denied', 'denies', 'no evidence', 'rumor', 'rumours',
    'scam', 'refuted', 'refutes', 'refute', 'unsubstantiated', 'bogus', 'no such',
    'never said', 'never announced', 'disproven', 'fact check', 'misinformation',
    'deepfake', 'morphed', 'clarified that no', 'baseless', 'rejected reports',
    'denied reports', 'falsely claimed', 'contrary to', 'myth', 'rather than',
    'instead', 'unverified', 'wrong', 'lacks', 'lack'
}

MONTH_NAMES = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr',
    'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
}


def _stem_nli_token(w: str) -> str:
    """Lightweight suffix-stripping stemmer for semantic token overlap."""
    w = w.lower()
    for suffix in ('ations', 'ation', 'ated', 'ates', 'ate', 'ments', 'ment', 'ingly', 'edly', 'ing', 'ed', 'ies', 'es', 's'):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


class NLIVerifier:
    """
    Natural Language Inference verifier that classifies the relationship
    between a claim and evidence passages as entailment (supporting),
    neutral, or contradiction (refuting).

    Uses DeBERTa-v3-base by default for strong accuracy with reasonable speed.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        """
        Initialize the NLI verifier.

        Args:
            model_name: HuggingFace model name for sequence classification.
                        Must be a model trained for NLI (3-class output).
                        Default: cross-encoder/nli-deberta-v3-base
        """
        logger.info(f"Loading NLI model: {model_name}")

        self.model_name = model_name
        self.has_neural_model = False
        self.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        self.label2id = {v: k for k, v in self.id2label.items()}

        if HAS_TRANSFORMERS and AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                if hasattr(self.model.config, 'id2label') and len(self.model.config.id2label) == 3:
                    self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
                self.label2id = {v: k for k, v in self.id2label.items()}

                self.device = torch.device('cuda' if (torch and torch.cuda.is_available()) else 'cpu')
                self.model.to(self.device)
                self.model.eval()

                import os
                if self.device.type == 'cpu' and hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(min(8, os.cpu_count() or 4))

                self.has_neural_model = True
                logger.info(f"NLI neural model loaded on device: {self.device}")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace NLI weights ({e}). Running in resilient heuristic mode.")
                self.has_neural_model = False
        else:
            logger.info("Transformers/Torch not present. Operating in resilient heuristic NLI mode.")

    @timer
    def verify_claim_against_evidence(
        self,
        claim: str,
        evidence_list: List[Dict],
        batch_size: int = 16,
        max_evidence: Optional[int] = None
    ) -> Dict:
        """
        Verify a claim against multiple evidence passages using NLI.

        For each evidence passage, classifies the claim-evidence pair as
        entailment, neutral, or contradiction, then aggregates results
        using evidence-weighted averaging.

        In NLI for fact verification:
            - Premise: Retrieved evidence passage
            - Hypothesis: The claim being verified

        Args:
            claim: The claim text to verify (hypothesis).
            evidence_list: List of evidence dicts with 'text' and optional 'score'.
            batch_size: Number of pairs to evaluate in a single batch.
            max_evidence: Maximum number of top-ranked evidence passages to evaluate.

        Returns:
            Dict with:
                - 'individual_results': Per-evidence NLI predictions
                - 'aggregated': Weighted aggregate verdict
                    - 'weighted_probabilities': {entailment, neutral, contradiction}
                    - 'primary_verdict': Most likely label
                    - 'confidence': Confidence of primary verdict
        """
        valid_evidence = [
            e for e in evidence_list
            if e.get('text') and e.get('text', '').strip()
        ]

        if max_evidence is not None and max_evidence > 0:
            valid_evidence = valid_evidence[:max_evidence]

        if not valid_evidence:
            logger.warning("No non-empty evidence provided for NLI verification.")
            aggregated = self._aggregate_nli_results([])
            return {
                'individual_results': [],
                'aggregated': aggregated
            }

        results = []

        if self.has_neural_model and hasattr(self, 'model') and self.model is not None:
            inference_ctx = torch.inference_mode if (torch and hasattr(torch, 'inference_mode')) else (torch.no_grad if torch else None)

            claim_lower = claim.lower()
            claim_tokens_raw = set(re.findall(r'\b\w+\b', claim_lower))
            claim_content = [w for w in claim_tokens_raw if w not in NLI_STOPWORDS and len(w) > 2]
            claim_stems = {_stem_nli_token(w) for w in claim_content} if claim_content else claim_tokens_raw
            claim_nums = set(re.findall(r'(?<![A-Za-z-])\b\d+(?:\.\d+)?\b(?![A-Za-z-])', claim_lower))

            # Process evidence in batches for better throughput
            for i in range(0, len(valid_evidence), batch_size):
                batch = valid_evidence[i:i + batch_size]
                premises = []
                for e in batch:
                    txt = e.get('text', '').strip()
                    title = (e.get('title') or '').strip()
                    clean_title = re.sub(r'\s*[-|—]\s*Wikipedia$', '', title, flags=re.IGNORECASE).strip()
                    if clean_title and clean_title.lower() not in txt[:min(len(txt), len(clean_title) + 40)].lower():
                        premises.append(f"{clean_title}: {txt}")
                    else:
                        premises.append(txt)
                hypotheses = [claim.strip()] * len(batch)

                # In NLI: Premise = evidence_text, Hypothesis = claim
                raw_inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                inputs = {
                    k: v.to(self.device) if hasattr(v, 'to') else v
                    for k, v in raw_inputs.items()
                }

                with inference_ctx():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for j, evidence in enumerate(batch):
                    probs = batch_probs[j]
                    prob_dict = {
                        self.id2label[k]: float(probs[k])
                        for k in range(len(probs))
                    }
                    e_p = prob_dict.get('entailment', 0.0)
                    n_p = prob_dict.get('neutral', 0.0)
                    c_p = prob_dict.get('contradiction', 0.0)

                    # Out-of-distribution safeguard for MNLI false contradictions on disjoint passages:
                    # MNLI cross-encoders often output C ~ 0.99 when two passages discuss different
                    # unrelated facts with zero refutation/negation cues or negligible lexical overlap.
                    if c_p >= 0.60 and c_p >= e_p and c_p >= n_p:
                        premise_low = premises[j].lower()
                        ev_tokens_raw = set(re.findall(r'\b\w+\b', premise_low))
                        ev_content = [w for w in ev_tokens_raw if w not in NLI_STOPWORDS and len(w) > 2]
                        ev_stems = {_stem_nli_token(w) for w in ev_content} if ev_content else ev_tokens_raw
                        stem_overlap = len(claim_stems & ev_stems) / max(len(claim_stems), 1)

                        ev_nums = set(re.findall(r'(?<![A-Za-z-])\b\d+(?:\.\d+)?\b(?![A-Za-z-])', premise_low))
                        has_num_conflict = bool(claim_nums and ev_nums and (claim_nums - ev_nums) and (ev_nums - claim_nums))

                        has_neg = bool(
                            (ev_tokens_raw & NEGATION_WORDS)
                            or re.search(r"\b\w+n['’]t\b", premise_low)
                            or any(cue in premise_low for cue in REFUTATION_CUES)
                        )
                        tier = evidence.get('source_tier', '')
                        sem_val = evidence.get('scores_breakdown', {}).get('semantic')

                        if (stem_overlap < 0.35 or (sem_val is not None and float(sem_val) < 0.25)) and not has_neg and tier != 'fact_checker':
                            shift = c_p * 0.85
                            c_p -= shift
                            n_p += shift
                        elif tier in ('general_web', 'unverified_social', '') and not has_neg and not has_num_conflict and stem_overlap < 0.60:
                            shift = c_p * 0.70
                            c_p -= shift
                            n_p += shift

                    calibrated_probs = {
                        'entailment': float(e_p),
                        'neutral': float(n_p),
                        'contradiction': float(c_p)
                    }
                    pred_label = max(calibrated_probs, key=calibrated_probs.get)
                    res = {
                        'evidence': evidence,
                        'probabilities': calibrated_probs,
                        'prediction': pred_label
                    }
                    results.append(res)
        else:
            # Resilient heuristic NLI stance detector handling fact-checking cues, negation parity,
            # conflicting numerical assertions, and prepositional entity swaps
            claim_lower = claim.lower()
            claim_words_raw = set(re.findall(r'\b\w+\b', claim_lower))
            claim_content_words = [w for w in claim_words_raw if w not in NLI_STOPWORDS and len(w) > 2]
            claim_stems = set(_stem_nli_token(w) for w in claim_content_words) if claim_content_words else claim_words_raw
            claim_has_neg = bool((claim_words_raw & NEGATION_WORDS) or re.search(r"\b\w+n['’]t\b", claim_lower))
            claim_nums = set(re.findall(r'(?<![A-Za-z-])\b\d+(?:\.\d+)?\b(?![A-Za-z-])', claim_lower))
            claim_prep_entities = {
                (m.group(1).lower(), m.group(2).lower())
                for m in re.finditer(r'\b(in|on|at|to|from)\s+(?:the\s+)?([A-Z][a-z]{2,})\b', claim)
                if m.group(2).lower() not in MONTH_NAMES
            }

            entail_cues = {
                'confirmed', 'confirms', 'announces', 'announced', 'announcement', 'official',
                'verified', 'reports that', 'accurate', 'authentic', 'gazette', 'circular',
                'press release', 'fact check: true', 'fact check : true',
                'inaugurate', 'inaugurates', 'inaugurated', 'inauguration',
                'launch', 'launches', 'launched', 'launching',
                'opened', 'opens', 'opening',
                'unveiled', 'unveils', 'unveiling',
                'flags off', 'flagged off', 'flagging off',
                'dedicated', 'dedicates', 'commissioned', 'commissions',
                'live updates', 'live for updates', 'breaking news', 'live:', 'live updates:',
                'held on', 'attended', 'addressed', 'foundation stone', 'ceremony', 'inspected'
            }

            for evidence in valid_evidence:
                ev_raw = f"{evidence.get('title', '')} {evidence.get('text', '')}".strip()
                text_lower = ev_raw.lower()
                ev_words_raw = set(re.findall(r'\b\w+\b', text_lower))
                ev_content_words = [w for w in ev_words_raw if w not in NLI_STOPWORDS and len(w) > 2]
                ev_stems = set(_stem_nli_token(w) for w in ev_content_words) if ev_content_words else ev_words_raw
                ev_has_neg = bool((ev_words_raw & NEGATION_WORDS) or re.search(r"\b\w+n['’]t\b", text_lower))
                ev_nums = set(re.findall(r'(?<![A-Za-z-])\b\d+(?:\.\d+)?\b(?![A-Za-z-])', text_lower))

                has_contra = any(cue in text_lower for cue in REFUTATION_CUES)
                has_entail = any(cue in text_lower for cue in entail_cues)

                stem_overlap = len(claim_stems & ev_stems) / max(len(claim_stems), 1)
                raw_overlap = len(claim_words_raw & ev_words_raw) / max(len(claim_words_raw), 1)
                effective_overlap = stem_overlap if len(claim_content_words) >= 2 else raw_overlap

                # Asymmetric negation parity & numeric / entity mismatch checks on shared context
                asymmetric_neg = (claim_has_neg != ev_has_neg) and effective_overlap >= 0.45
                both_neg = claim_has_neg and ev_has_neg and effective_overlap >= 0.45
                numeric_conflict = bool(
                    claim_nums and ev_nums
                    and (claim_nums - ev_nums) and (ev_nums - claim_nums)
                    and effective_overlap >= 0.45
                )

                ev_prep_entities = {
                    (m.group(1).lower(), m.group(2).lower())
                    for m in re.finditer(r'\b(in|on|at|to|from)\s+(?:the\s+)?([A-Z][a-z]{2,})\b', ev_raw)
                    if m.group(2).lower() not in MONTH_NAMES
                }
                entity_swap_conflict = False
                if effective_overlap >= 0.50 and claim_prep_entities and ev_prep_entities:
                    for prep, c_ent in claim_prep_entities:
                        if c_ent not in ev_words_raw and any(ep == prep and ee != c_ent for ep, ee in ev_prep_entities):
                            entity_swap_conflict = True
                            break

                is_high_cred = (
                    evidence.get('is_official_primary', False) or
                    evidence.get('source_tier') in ('primary_origin', 'fact_checker', 'news_wire', 'mainstream_press') or
                    float(evidence.get('authority_score', 0.5)) >= 0.80
                )

                if (has_contra and not both_neg and (effective_overlap >= 0.25 or evidence.get('source_tier') == 'fact_checker')) or asymmetric_neg:
                    probs = {'entailment': 0.05, 'neutral': 0.10, 'contradiction': 0.85}
                    pred = 'contradiction'
                elif numeric_conflict or entity_swap_conflict:
                    probs = {'entailment': 0.06, 'neutral': 0.14, 'contradiction': 0.80}
                    pred = 'contradiction'
                elif both_neg:
                    probs = {'entailment': 0.82, 'neutral': 0.12, 'contradiction': 0.06}
                    pred = 'entailment'
                elif has_entail and effective_overlap >= 0.35:
                    e_prob = 0.90 if is_high_cred else 0.85
                    probs = {'entailment': e_prob, 'neutral': round((1.0 - e_prob) * 0.65, 3), 'contradiction': round((1.0 - e_prob) * 0.35, 3)}
                    pred = 'entailment'
                elif effective_overlap >= 0.55:
                    e_prob = 0.88 if is_high_cred else 0.75
                    probs = {'entailment': e_prob, 'neutral': round((1.0 - e_prob) * 0.70, 3), 'contradiction': round((1.0 - e_prob) * 0.30, 3)}
                    pred = 'entailment'
                elif effective_overlap >= 0.40 and is_high_cred:
                    probs = {'entailment': 0.78, 'neutral': 0.16, 'contradiction': 0.06}
                    pred = 'entailment'
                elif effective_overlap >= 0.40:
                    probs = {'entailment': 0.65, 'neutral': 0.25, 'contradiction': 0.10}
                    pred = 'entailment'
                else:
                    probs = {'entailment': 0.15, 'neutral': 0.70, 'contradiction': 0.15}
                    pred = 'neutral'

                res = {
                    'evidence': evidence,
                    'probabilities': probs,
                    'prediction': pred
                }
                results.append(res)

        # Aggregate predictions across all evidence
        aggregated = self._aggregate_nli_results(results)

        logger.info(
            f"NLI verdict: {aggregated['primary_verdict'].upper()} "
            f"(confidence: {aggregated['confidence']:.2%})"
        )

        # Build evidence_details for easy lookup and backward compatibility
        evidence_details = []
        for r in results:
            ev = r.get('evidence', {})
            pred = r.get('prediction', 'neutral')
            probs = r.get('probabilities', {})
            conf = probs.get(pred, 0.5) if probs else 0.5
            evidence_details.append({
                'text': ev.get('text', ''),
                'title': ev.get('title', ''),
                'source': ev.get('source', ''),
                'url': ev.get('url', ''),
                'verdict': pred,
                'relationship': pred,
                'confidence': conf,
                'probabilities': probs,
                'score': ev.get('score', 1.0)
            })

        return {
            'individual_results': results,
            'evidence_details': evidence_details,
            'aggregated': aggregated
        }

    def _aggregate_nli_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate NLI predictions across multiple evidence pieces.

        Uses informative stance weighting: retrieved background passages that do not
        directly address the claim (neutral ~ 1.0) provide general context but must not
        drown out decisive, verified factual evidence (entailment or contradiction > 0.65).

        Args:
            results: List of individual NLI result dicts.

        Returns:
            Aggregated verdict dict with weighted_probabilities,
            primary_verdict, and confidence.
        """
        if not results:
            return {
                'weighted_probabilities': {
                    'entailment': 0.0,
                    'neutral': 1.0,
                    'contradiction': 0.0
                },
                'primary_verdict': 'neutral',
                'confidence': 0.0
            }

        if len(results) == 1:
            probs = dict(results[0].get('probabilities', {}))
            pred = max(probs, key=probs.get) if probs else 'neutral'
            conf = probs.get(pred, 0.0) if probs else 0.0
            return {
                'weighted_probabilities': probs,
                'primary_verdict': pred,
                'confidence': conf
            }

        # Tier-stratified evidentiary weighting + semantic relevance:
        # 1. Primary official origins (2.2x) and fact-checkers (2.0x) carry highest weight.
        # 2. News wires (1.6x), mainstream press (1.35x), encyclopedic records (1.15x).
        # 3. General web (0.55x) and unverified social media (0.25x) are dampened.
        # 4. Neutral passages retain a 0.25 base weight floor so 1 noisy source cannot hijack 4 neutrals.
        tier_mults = {
            'primary_origin': 2.2,
            'fact_checker': 2.0,
            'news_wire': 1.6,
            'mainstream_press': 1.35,
            'encyclopedic_record': 1.15,
            'general_web': 0.55,
            'unverified_social': 0.25,
        }

        weights = []
        for r in results:
            ev = r.get('evidence', {})
            base_score = max(0.01, float(ev.get('score', 1.0)))
            tier = ev.get('source_tier', '')
            if ev.get('is_official_primary'):
                auth_mult = tier_mults['primary_origin']
            elif tier in tier_mults:
                auth_mult = tier_mults[tier]
            else:
                a_score = float(ev.get('authority_score', 0.6))
                auth_mult = 1.5 if a_score >= 0.90 else (0.55 if a_score <= 0.60 else 1.0)

            probs_r = r.get('probabilities', {})
            signal_strength = probs_r.get('entailment', 0.0) + probs_r.get('contradiction', 0.0)
            informativeness = 0.25 + 0.75 * signal_strength

            sem_score = ev.get('scores_breakdown', {}).get('semantic')
            sem_mult = max(0.25, min(1.2, float(sem_score) * 1.5)) if sem_score is not None else 1.0

            w = base_score * auth_mult * sem_mult * informativeness
            weights.append(max(0.001, w))

        total_weight = sum(weights) or 1.0

        weighted_probs = {
            'entailment': 0.0,
            'neutral': 0.0,
            'contradiction': 0.0
        }

        for result, w in zip(results, weights):
            norm_w = w / total_weight
            for label, prob in result.get('probabilities', {}).items():
                weighted_probs[label] += prob * norm_w

        def _is_on_topic(res_item: Dict) -> bool:
            sem = res_item.get('evidence', {}).get('scores_breakdown', {}).get('semantic')
            return sem is None or float(sem) >= 0.22

        # Check for decisive signals across on-topic evidence items
        max_e = max(
            (r.get('probabilities', {}).get('entailment', 0.0) for r in results if _is_on_topic(r)),
            default=max((r.get('probabilities', {}).get('entailment', 0.0) for r in results), default=0.0)
        )
        max_c = max(
            (r.get('probabilities', {}).get('contradiction', 0.0) for r in results if _is_on_topic(r)),
            default=max((r.get('probabilities', {}).get('contradiction', 0.0) for r in results), default=0.0)
        )

        entailing_sources = [r for r in results if r.get('probabilities', {}).get('entailment', 0.0) >= 0.65 and _is_on_topic(r)]
        contradicting_sources = [r for r in results if r.get('probabilities', {}).get('contradiction', 0.0) >= 0.60 and _is_on_topic(r)]

        # Symmetric Corroboration Checks:
        # Neither Supported nor Contradicted may be decided solely by a single unvetted web blog/social post.
        authoritative_entailing = [
            r for r in entailing_sources
            if r.get('evidence', {}).get('source_tier') != 'unverified_social'
        ]
        has_entailment_corroboration = (
            len(authoritative_entailing) >= 2 or
            any(r.get('evidence', {}).get('is_official_primary') for r in entailing_sources) or
            (len(authoritative_entailing) >= 1 and weighted_probs['entailment'] >= 0.50)
        )

        has_any_tier_meta = any(
            'source_tier' in r.get('evidence', {}) or 'authority_score' in r.get('evidence', {})
            for r in results
        )
        authoritative_contradicting = [
            r for r in contradicting_sources
            if r.get('evidence', {}).get('is_official_primary')
            or r.get('evidence', {}).get('source_tier') in ('primary_origin', 'fact_checker', 'news_wire', 'mainstream_press')
            or float(r.get('evidence', {}).get('authority_score', 0.6)) >= 0.85
        ]
        has_contra_corroboration = (
            len(authoritative_contradicting) >= 1
            or len(contradicting_sources) >= 2
            or (not has_any_tier_meta and len(contradicting_sources) >= 1)
        )

        # 1. Decisive Contradiction (directly debunked/refuted by authoritative or multiple corroborated sources)
        if has_contra_corroboration and (
            (len(contradicting_sources) >= 1 and max_c >= 0.70 and max_e < 0.40) or
            (weighted_probs['contradiction'] > weighted_probs['entailment'] and weighted_probs['contradiction'] >= 0.45)
        ):
            primary_verdict = 'contradiction'
            confidence = max(weighted_probs['contradiction'], max_c * 0.85)

        # 2. Decisive Entailment (requires corroboration and no strong contradiction)
        elif has_entailment_corroboration and max_e >= 0.70 and max_c < 0.40 and weighted_probs['entailment'] > weighted_probs['contradiction']:
            primary_verdict = 'entailment'
            confidence = max(weighted_probs['entailment'], min(0.95, max_e * (0.9 if len(entailing_sources) >= 2 else 0.7)))

        # 3. Default: Fallback to majority weighted probability distribution
        else:
            primary_verdict = max(weighted_probs, key=weighted_probs.get)
            confidence = max(weighted_probs.values())

            # Symmetric uncorroborated outlier guard
            if primary_verdict == 'entailment' and not has_entailment_corroboration:
                primary_verdict = 'neutral'
                confidence = max(weighted_probs.get('neutral', 0.60), 0.65)
            elif primary_verdict == 'contradiction' and not has_contra_corroboration:
                primary_verdict = 'neutral'
                confidence = max(weighted_probs.get('neutral', 0.60), 0.65)

        return {
            'weighted_probabilities': weighted_probs,
            'primary_verdict': primary_verdict,
            'confidence': confidence
        }
