# src/claim_extraction.py
# Phase 2: Claim Extraction & Structuring
# Extracts structured claims C = (E, V, T, D) from raw text input.

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

import spacy
from dateparser import parse as parse_date
import re

from src.utils import setup_logger, timer

logger = setup_logger(__name__)


@dataclass
class Claim:
    """
    Structured claim representation.

    Attributes:
        entities: E — extracted named entities (persons, orgs, locations, etc.)
        event: V — main event/predicate extracted via dependency parsing
        temporal: T — temporal expressions (absolute and relative dates)
        domain: D — classified domain (politics, health, science, space, cinema, etc.)
        original_text: The raw input text
        claimant: Extracted originating authority or speaker
        category: Inferred universal domain category
    """
    entities: List[str] = field(default_factory=list)
    event: str = ""
    temporal: Dict = field(default_factory=dict)
    domain: str = "general"
    original_text: str = ""
    claimant: str = ""
    category: str = "general"
    primary_origins: List[Dict] = field(default_factory=list)
    is_speech_claim: bool = False


class ClaimExtractor:
    """
    Extracts and structures claims from raw text using NER,
    dependency parsing, temporal recognition, and domain classification.
    """

    def __init__(self, ner_model: str = "en_core_web_trf"):
        """
        Initialize the ClaimExtractor.

        Args:
            ner_model: spaCy model name for NER and dependency parsing.
                       Use 'en_core_web_trf' for best accuracy or
                       'en_core_web_sm' for lightweight/fast inference.
        """
        logger.info(f"Loading spaCy model: {ner_model}")
        try:
            self.nlp = spacy.load(ner_model)
        except OSError:
            logger.warning(
                f"Model '{ner_model}' not found. Falling back to 'en_core_web_sm'. "
                f"Install the transformer model with: python -m spacy download {ner_model}"
            )
            self.nlp = spacy.load("en_core_web_sm")
        self._cache = {}
        self._cache_max_size = 500
        logger.info("spaCy model loaded successfully with acceleration cache")

    def extract_entities(self, text_or_doc) -> List[Dict]:
        """
        Extract persons, organizations, locations, dates, and other
        named entities from the input text or existing spaCy Doc.

        Args:
            text_or_doc: Raw claim text or pre-parsed spaCy Doc.

        Returns:
            List of entity dicts with keys: text, label, start, end.
        """
        doc = text_or_doc if hasattr(text_or_doc, 'ents') else self.nlp(text_or_doc)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def extract_temporal_info(self, text: str) -> Dict:
        """
        Extract and normalize time expressions from text using
        fast pattern matching, strptime, and dateparser fallback.

        Args:
            text: Raw claim text.

        Returns:
            Dict with 'mentioned_dates' (parsed datetime objects)
            and 'relative_times' (unparseable time expressions).
        """
        if not text or not isinstance(text, str):
            return {'mentioned_dates': [], 'relative_times': []}

        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',                        # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',                        # DD-MM-YYYY or MM-DD-YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',                             # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', # DD Month YYYY
        ]

        fast_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        temporal_info = {'mentioned_dates': [], 'relative_times': []}

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.replace(',', ' ').strip()
                parsed = None
                # Fast path: try standard strptime formats first
                for fmt in fast_formats:
                    try:
                        parsed = datetime.strptime(clean_match if ' ' in fmt else match, fmt)
                        break
                    except (ValueError, TypeError):
                        continue

                # Fallback to dateparser if strptime didn't catch it
                if parsed is None:
                    try:
                        parsed = parse_date(match)
                    except Exception:
                        parsed = None

                if parsed is not None:
                    temporal_info['mentioned_dates'].append(parsed)
                else:
                    temporal_info['relative_times'].append(match)

        return temporal_info

    def extract_main_event(self, text_or_doc) -> str:
        """
        Extract the primary predicate/event using dependency parsing.
        Finds the ROOT token and its immediate dependents (subject, object, modifier)
        ordered naturally by token sequence.

        Args:
            text_or_doc: Raw claim text or pre-parsed spaCy Doc.

        Returns:
            String describing the main event, or first 50 chars as fallback.
        """
        if text_or_doc is None:
            return ""

        doc = text_or_doc if hasattr(text_or_doc, '__iter__') and hasattr(text_or_doc, 'ents') else self.nlp(str(text_or_doc))

        # Find ROOT token (main verb/predicate)
        for token in doc:
            if token.dep_ == "ROOT":
                relevant_deps = {"nsubj", "nsubjpass", "dobj", "pobj", "nmod", "attr", "prep", "acomp", "advmod", "prt"}
                selected_tokens = [token] + [child for child in token.children if child.dep_ in relevant_deps]
                # Sort by token position in original sentence to preserve natural phrasing
                selected_tokens.sort(key=lambda t: t.i)
                event_text = " ".join(t.text for t in selected_tokens).strip()
                if event_text:
                    return event_text

        return (doc.text if hasattr(doc, 'text') else str(text_or_doc))[:50]  # Fallback to first 50 chars

    def detect_domain(self, text: str) -> str:
        """
        Classify the claim domain based on keyword matching and universal resolver fallback.

        Supported domains: politics, health, science, business, general.

        Args:
            text: Raw claim text.

        Returns:
            Detected domain string.
        """
        if not text or not isinstance(text, str):
            return 'general'

        domain_keywords = {
            'politics': {
                'election', 'politician', 'vote', 'government', 'congress',
                'president', 'senate', 'democrat', 'republican', 'legislation',
                'minister', 'parliament', 'assembly', 'court', 'supreme', 'governor',
                'scheme', 'policy', 'cm', 'pm', 'pmo', 'bjp', 'dmk', 'aiadmk',
                'tvk', 'party', 'bill', 'ordinance', 'cabinet', 'treaty', 'sanctions',
                'diplomat', 'ambassador', 'ceasefire', 'military', 'panchayat'
            },
            'health': {
                'vaccine', 'disease', 'cure', 'doctor', 'health', 'medicine',
                'virus', 'pandemic', 'hospital', 'treatment', 'symptom',
                'covid', 'cancer', 'drug', 'infection', 'clinical', 'medical',
                'organ', 'heart', 'advisory', 'nutrition', 'ayush', 'mental',
                'surgery', 'patient', 'therapy', 'bacteria', 'cholesterol'
            },
            'science': {
                'study', 'research', 'scientist', 'experiment', 'evidence',
                'climate', 'space', 'physics', 'biology', 'discovery',
                'sun', 'moon', 'planet', 'earth', 'solar', 'star', 'stars',
                'astronomy', 'astronomer', 'orbit', 'gravity', 'telescope',
                'universe', 'quantum', 'chemistry', 'laboratory', 'nature',
                'scientific', 'mars', 'lunar', 'galaxy', 'sphere', 'atmosphere',
                'asteroid', 'comet', 'satellite', 'rocket'
            },
            'business': {
                'company', 'stock', 'profit', 'business', 'market',
                'revenue', 'ceo', 'investment', 'startup', 'economy',
                'bank', 'banking', 'rbi', 'gst', 'tax', 'finance',
                'inflation', 'gdp', 'currency', 'rupee', 'dollar', 'trade',
                'shares', 'investor', 'corporate', 'loan',
                'crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain',
                'token', 'tokens', 'airdrop', 'giveaway', 'nft', 'defi',
                'wallet', 'mining', 'exchange', 'binance', 'coinbase',
                'staking', 'web3'
            },
        }

        # Fast token-set intersection
        words = set(re.findall(r'\b\w+\b', text.lower()))
        domain_scores = {
            domain: len(words & keywords)
            for domain, keywords in domain_keywords.items()
        }

        max_score = max(domain_scores.values()) if domain_scores else 0
        if max_score > 0:
            return max(domain_scores, key=domain_scores.get)

        # Fallback to UniversalOriginResolver category inference
        try:
            from src.authority_sources import UniversalOriginResolver
            u_cat = UniversalOriginResolver.infer_category(text)
            category_to_domain = {
                'space_astronomy': 'science',
                'health_medical': 'health',
                'agriculture_farming': 'science',
                'finance_banking': 'business',
                'governance_state_national': 'politics',
                'tamil_nadu_govt': 'politics',
                'tvk_party': 'politics',
                'district_local_gov': 'politics',
                'global_affairs': 'politics',
                'cinema_entertainment': 'general',
            }
            if u_cat in category_to_domain:
                return category_to_domain[u_cat]
        except Exception:
            pass

        return 'general'

    @timer
    def extract_claim(self, text: str, ocr_data: Optional[Dict] = None) -> Claim:
        """
        Extract structured claim from text with single-pass spaCy inference
        and multi-source primary entity resolution.

        Args:
            text: Raw claim text.
            ocr_data: Optional OCR extraction dict from image/poster.

        Returns:
            Structured Claim object.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return Claim(
                entities=[],
                event="",
                temporal={'mentioned_dates': [], 'relative_times': []},
                domain="general",
                original_text=str(text or ""),
                claimant="",
                category="general",
                primary_origins=[]
            )

        clean_text = text.strip()
        if not ocr_data and clean_text in self._cache:
            cached = self._cache[clean_text]
            return Claim(
                entities=list(cached.entities),
                event=cached.event,
                temporal={
                    'mentioned_dates': list(cached.temporal.get('mentioned_dates', [])),
                    'relative_times': list(cached.temporal.get('relative_times', []))
                },
                domain=cached.domain,
                original_text=cached.original_text,
                claimant=getattr(cached, 'claimant', ''),
                category=getattr(cached, 'category', cached.domain),
                primary_origins=list(getattr(cached, 'primary_origins', [])),
                is_speech_claim=getattr(cached, 'is_speech_claim', False)
            )

        logger.info("Extracting structured claim (single-pass spaCy)...")

        # Parse text ONCE with spaCy, reuse across entity and event extraction
        doc = self.nlp(text)
        entities = self.extract_entities(doc)
        temporal = self.extract_temporal_info(text)
        event = self.extract_main_event(doc)
        domain = self.detect_domain(text)

        entity_texts = [ent['text'] for ent in entities]

        # Check if this claim concerns a speech, press meet, rally, interview, or public address
        combined_check_text = (text or "") + " " + (ocr_data.get('extracted_text', '') if ocr_data else "")
        is_speech = bool(re.search(
            r'(\b(speech|spoke|speaks|speaking|said|stated|statement|press\s*meet|press\s*conference|interview|'
            r'rally|addressed|addressing|parliamentary|parliament|assembly|public\s*address|'
            r'bhashan|stage\s*talk)\b|'
            r'(பேச்சு|பேசிய|செய்தியாளர்|கூறிய|பேட்டி|அறிக்கை))',
            combined_check_text,
            re.IGNORECASE
        ))

        # Filter candidate claimant entities (only PERSON, ORG, NORP) to avoid non-claimants (e.g. 'Sun', 'today')
        claimant_candidate_entities = [
            ent['text'] for ent in entities
            if ent.get('label') in ('PERSON', 'ORG', 'NORP')
        ]

        # Multi-Source Primary Entity Detection across ALL sectors
        claimant = ""
        category = domain
        primary_origins = []
        try:
            from src.authority_sources import UniversalOriginResolver
            primary_origins = UniversalOriginResolver.resolve_primary_origins(
                text,
                entities=entity_texts,
                ocr_data=ocr_data
            )
            c_name, _ = UniversalOriginResolver.extract_claimant_and_district(text, claimant_candidate_entities)
            if c_name:
                claimant = c_name
            elif primary_origins:
                claimant = primary_origins[0].get('name', '')

            u_cat = UniversalOriginResolver.infer_category(text, entity_texts)
            if u_cat and u_cat != 'general':
                category = u_cat
        except Exception as e:
            logger.debug(f"Universal primary origin extraction error: {e}")

        claim = Claim(
            entities=entity_texts,
            event=event,
            temporal=temporal,
            domain=domain,
            original_text=text,
            claimant=claimant,
            category=category,
            primary_origins=primary_origins,
            is_speech_claim=is_speech
        )

        # Store in cache
        if not ocr_data:
            if len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[clean_text] = claim

        logger.info(
            f"Claim extracted — entities: {entity_texts[:3]}, claimant: '{claimant}', "
            f"category: '{category}', is_speech: {is_speech}, primary_origins: {[po['name'] for po in primary_origins]}"
        )

        return claim


def generate_queries(claim: Claim) -> List[str]:
    """
    Convert a structured claim into multiple retrieval queries
    for evidence search with high-priority primary official source channels.

    Strategy:
        - For Speeches / Press Meets: Public addresses of popular figures (politicians,
          actors, leaders) are proven through TV news channels, live video broadcasts,
          accredited newspapers, and fact checks — NOT by personal social pages.
        - For Institutional / Policy / Corporate / Science claims: Primary official channels,
          government gazettes, research repositories, and official handles.

    Args:
        claim: Structured Claim object.

    Returns:
        Deduplicated list of query strings.
    """
    if not claim:
        return ["news verification"]

    queries = []

    # 0. Full original claim text (ALWAYS the primary authoritative query for live search)
    if claim.original_text and claim.original_text.strip():
        queries.append(claim.original_text.strip()[:150])

    # SPEECH / PRESS MEET CLAIMS: Target accredited TV news, newspapers, and fact-checks
    if getattr(claim, 'is_speech_claim', False):
        speaker = claim.claimant or (claim.entities[0] if claim.entities else "")
        topic_entities = [e for e in claim.entities if e.lower() not in (speaker.lower(), "speech", "photo", "shows")][:2]
        topic_str = " ".join(topic_entities)

        if speaker:
            if topic_str:
                queries.append(f'"{speaker}" {topic_str} speech news')
                queries.append(f'"{speaker}" {topic_str} press meet video')
            else:
                queries.append(f'"{speaker}" speech news video')
                queries.append(f'"{speaker}" press meet live news')

            # Accredited television channels and regional/national newspapers
            queries.append(
                f'"{speaker}" speech (site:thehindu.com OR site:puthiyathalaimurai.com OR '
                f'site:thanthitv.com OR site:polimernews.com OR site:dinamalar.com OR '
                f'site:indianexpress.com OR site:ndtv.com)'
            )
            queries.append(f'"{speaker}" speech fact check')

        if claim.event and claim.event.strip() and claim.event.strip() not in queries:
            queries.append(claim.event.strip())

        if len(queries) < 4 and len(claim.entities) >= 2:
            queries.append(' '.join(claim.entities[:3]))

        return list(dict.fromkeys([q for q in queries if q.strip()]))[:7]

    # GENERAL / INSTITUTIONAL CLAIMS:
    # 0b. High-signal keyword condensation (preserves late identifiers like K2-18b & avoids bare 'Earth NASA' queries)
    if claim.original_text and len(claim.original_text.split()) > 4:
        try:
            from src.authority_sources import UniversalOriginResolver
            kw_q = UniversalOriginResolver._extract_topic_terms(claim.original_text, limit=6)
            if kw_q and len(kw_q.split()) >= 3:
                queries.append(kw_q)
        except Exception:
            pass

    # 1. Entity combinations (only add if it actually carries 3+ words/tokens, or as lower-priority fallback)
    if len(claim.entities) >= 2:
        ent_combo = ' '.join(claim.entities[:3])
        if len(ent_combo.split()) >= 3:
            queries.append(ent_combo)

    # 2. Main entity + event queries (relational factual search)
    for entity in claim.entities:
        if claim.event and claim.event.strip():
            if entity.lower() in claim.event.lower():
                queries.append(claim.event.strip())
            else:
                queries.append(f"{claim.event} {entity}".strip())
        elif claim.domain and claim.domain != 'general':
            queries.append(f"{entity} {claim.domain}".strip())

    # 3. Temporal-aware query
    if claim.temporal and claim.temporal.get('mentioned_dates'):
        d0 = claim.temporal['mentioned_dates'][0]
        date_str = str(d0.date()) if hasattr(d0, 'date') else str(d0)[:10]
        if claim.event and claim.event.strip():
            queries.append(f"{claim.event} {date_str}")
        elif claim.entities:
            queries.append(f"{claim.entities[0]} {date_str}")

    # 4. Entity + domain context (avoids bare brand names that only return social handles)
    if len(claim.entities) == 1:
        e = claim.entities[0].strip()
        if claim.domain and claim.domain != 'general':
            queries.append(f"{e} {claim.domain}")
        else:
            queries.append(e)

    # 5. Official channels of identified Primary Origins (supplementary verification)
    if hasattr(claim, 'primary_origins') and claim.primary_origins:
        for po in claim.primary_origins:
            for sq in po.get('search_queries', []):
                if sq and sq.strip() and sq.strip() not in queries:
                    queries.append(sq.strip())
                if len(queries) >= 8:
                    break
            if len(queries) >= 8:
                break

    # Fallback if empty
    if not queries:
        fallback = f"{claim.event} {claim.domain}".strip()
        queries.append(fallback if fallback else "news verification")

    # Remove duplicates while preserving order
    return list(dict.fromkeys([q for q in queries if q.strip()]))

