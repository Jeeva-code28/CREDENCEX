# src/authority_sources.py
# Universal Dynamic Origin Resolver
# Automatically identifies primary original sources, accredited news wires,
# certified fact-checkers, and official channels for ALL news sectors:
# Cinema/Entertainment, Governance/Politics, Banking/Finance, Health/Medical,
# Space/Science, Corporate/Business, Sports, District Administration, etc.

import re
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict
from urllib.parse import urlparse

from src.utils import setup_logger

logger = setup_logger(__name__)

# -------------------------------------------------------------------------
# Universal Domain Pattern Hierarchies (Algorithmic, Zero-Hardcoding)
# -------------------------------------------------------------------------

# Peer-reviewed scientific, medical, and technical research repositories
ACADEMIC_JOURNAL_DOMAINS = {
    'ncbi.nlm.nih.gov', 'pubmed.ncbi.nlm.nih.gov', 'nature.com', 'sciencedirect.com',
    'thelancet.com', 'nejm.org', 'bmj.com', 'cell.com', 'sciencemag.org',
    'science.org', 'jamanetwork.com', 'arxiv.org', 'biorxiv.org', 'medrxiv.org',
    'ieeexplore.ieee.org', 'springer.com', 'wiley.com', 'frontiersin.org', 'doi.org'
}

# Accredited News Wire Services (Primary reporting origin for breaking news)
NEWS_WIRE_DOMAINS = {
    'ptinews.com', 'aninews.in', 'reuters.com', 'apnews.com', 'afp.com',
    'bloomberg.com', 'ians.in', 'upi.com', 'xinhuanet.com'
}

# Certified Fact-Checking Networks (IFCN signatory networks worldwide)
CERTIFIED_FACTCHECK_DOMAINS = {
    'pib.gov.in', 'youturn.in', 'boomlive.in', 'altnews.in', 'factly.in',
    'vishvasnews.com', 'factcrescendo.com', 'thequint.com', 'indiatoday.in',
    'snopes.com', 'politifact.com', 'factcheck.org', 'afp.com', 'reuters.com',
    'fullfact.org', 'checkyourfact.com', 'leadstories.com', 'newsmobile.in'
}

# Recognized Domain Keywords for Category Inference across ALL sectors
DOMAIN_CATEGORY_KEYWORDS = {
    'cinema_entertainment': [
        'movie', 'film', 'cinema', 'director', 'producer', 'production', 'actor',
        'actress', 'teaser', 'trailer', 'box office', 'collection', 'crore', 'crores',
        'worldwide', 'blockbuster', 'shooting', 'release date', 'soundtrack', 'kollywood',
        'bollywood', 'tollywood', 'hollywood', 'suriya', 'karuppu', 'kanguva', 'leo',
        'jailer', 'coolie', 'goat', 'thalapathy vijay', 'rajinikanth', 'kamal haasan',
        'ajith', 'sivakarthikeyan', 'anirudh', 'vetri maaran', 'lokesh kanagaraj',
        'dream warrior', 'lyca', 'sun pictures', '2d entertainment', 'red giant',
        'mythri', 'hombale', 'studio green', 'dharma productions', 'yrf'
    ],
    'governance_state_national': [
        'prime minister', 'chief minister', 'pmo', 'governor', 'supreme court',
        'high court', 'ordinance', 'bill', 'assembly', 'parliament', 'lok sabha',
        'rajya sabha', 'gazette', 'cabinet', 'election commission', 'eci', 'president',
        'government of india', 'union ministry', 'judiciary'
    ],
    'tamil_nadu_govt': [
        'tamil nadu', 'tamilnadu', 'tn govt', 'tn government', 'mk stalin',
        'kalaignar magalir urimai', 'magalir urimai', 'pudhumai penn', 'pongal gift',
        'tnpsc', 'tasmac', 'tnega', 'dipr', 'tnsocialwelfare'
    ],
    'tvk_party': [
        'tvk', 'tamilaga vettri kazhagam', 'thamizhaga vetri kazhagam', 'tvk party',
        'thalapathy vijay', 'tvk flag', 'tvk rally', 'vikravandi', 'tvkvijayhq'
    ],
    'finance_banking': [
        'rbi', 'reserve bank', 'central bank', 'federal reserve', 'upi', 'npci',
        'gst', 'cbic', 'income tax', 'finmin', 'repo rate', 'inflation', 'gdp',
        'currency note', '500 note', 'bank locker', 'atm', 'sebi', 'stock market',
        'sensex', 'nifty', 'demat', 'mutual fund', 'loan', 'interest rate',
        'crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'token',
        'tokens', 'airdrop', 'giveaway', 'nft', 'defi', 'wallet', 'mining',
        'exchange', 'binance', 'coinbase', 'staking', 'web3'
    ],
    'health_medical': [
        'who', 'world health organization', 'who advisory', 'who report', 'who guidelines',
        'icmr', 'cdc', 'fda', 'aiims', 'mohfw', 'lancet', 'pubmed',
        'virus', 'vaccine', 'pandemic', 'disease', 'cancer', 'symptom', 'clinical trial',
        'drug approval', 'hospital', 'medicine', 'health advisory', 'infection',
        'medical study', 'doctor', 'treatment', 'heart attack', 'cholesterol'
    ],
    'space_astronomy': [
        'isro', 'nasa', 'esa', 'jaxa', 'spacex', 'roscosmos', 'chandrayaan',
        'gaganyaan', 'artemis', 'satellite', 'rocket', 'lunar', 'moon', 'mars',
        'exoplanet', 'james webb', 'hubble', 'orbit', 'astronomer', 'telescope',
        'supernova', 'black hole', 'asteroid', 'space station'
    ],
    'agriculture_farming': [
        'icar', 'ministry of agriculture', 'agricoop', 'msp', 'minimum support price',
        'farmer', 'paddy', 'wheat', 'sugarcane', 'fertilizer', 'urea', 'subsidy',
        'monsoon crop', 'kharif', 'rabi', 'irrigation', 'apmc', 'krishi',
        'pm kisan', 'crop insurance', 'soil health', 'drip irrigation', 'tractor'
    ],
    'district_local_gov': [
        'district collector', 'collector', 'district administration', 'collectorate',
        'taluk', 'tahsildar', 'tehsildar', 'revenue divisional officer', 'rdo',
        'municipal corporation', 'municipality', 'panchayat', 'school holiday',
        'rain holiday', 'flood alert', 'red alert', 'curfew', 'section 144',
        'local body', 'zila parishad', 'district magistrate', 'dm'
    ],
    'corporate_business': [
        'ceo', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit',
        'merger', 'acquisition', 'layoffs', 'tata', 'reliance', 'infosys', 'tcs',
        'adani', 'apple', 'google', 'microsoft', 'meta', 'amazon', 'tesla', 'openai',
        'investor relations', 'press release'
    ],
    'sports': [
        'cricket', 'bcci', 'icc', 'ipl', 'fifa', 'football', 'world cup', 'test match',
        'odi', 't20', 'olympics', 'medal', 'csk', 'rcb', 'mi', 'kohli', 'dhoni', 'rohit'
    ],
    'education': [
        'ugc', 'nta', 'cbse', 'tnpsc', 'upsc', 'exam', 'results', 'neet', 'jee',
        'cutoff', 'hall ticket', 'syllabus', 'admissions', 'counseling'
    ],
    'global_affairs': [
        'united nations', 'un security council', 'unsc', 'international court of justice',
        'icj', 'icc', 'nato', 'g20', 'g7', 'brics', 'treaty', 'ceasefire', 'ambassador',
        'foreign ministry', 'bilateral', 'sanctions', 'embassy'
    ]
}

# -------------------------------------------------------------------------
# Contextual Disambiguation Engine for Homonymous / Polysemous Keywords
# -------------------------------------------------------------------------
# Many acronyms and short words have dual meanings:
#   "who" = World Health Organization (ORG) vs. English relative pronoun
#   "can" = modal verb vs. organization abbreviation
#   "mars" = planet vs. chocolate brand vs. verb "to mar"
#   "bill" = legislation vs. person name vs. restaurant check
#   "cell" = biology term vs. prison cell vs. phone/battery cell
#   "exchange" = stock/crypto exchange vs. verbal exchange
#   "mining" = crypto mining vs. coal/mineral mining
#
# This engine validates keyword matches using three orthogonal signals:
#   1. ORTHOGRAPHIC: Case-sensitive form in the original text (WHO ≠ who ≠ Who)
#   2. COLLOCATES:   Presence of domain-specific co-occurring words
#   3. EXCLUSION:    Presence of words that indicate the OTHER meaning
# -------------------------------------------------------------------------

class AmbiguousTermConfig(TypedDict):
    entity_meaning: str
    common_meanings: List[str]
    case_forms: Optional[List[str]]
    collocates: Set[str]
    min_collocates: int
    exclusion_words: Set[str]


AMBIGUOUS_TERMS: Dict[str, AmbiguousTermConfig] = {
    # ---- Health / Medical ----
    'who': {
        'entity_meaning': 'World Health Organization',
        'common_meanings': ['relative pronoun', 'interrogative pronoun'],
        'case_forms': ['WHO'],           # Must appear as uppercase "WHO" in original text
        'collocates': {
            'health', 'medical', 'vaccine', 'pandemic', 'disease', 'virus', 'epidemic',
            'outbreak', 'advisory', 'guidelines', 'clinical', 'infection', 'hospital',
            'patients', 'mortality', 'treatment', 'diagnosis', 'organization', 'organisation',
            'global health', 'public health', 'tedros', 'geneva'
        },
        'min_collocates': 1,             # Need at least 1 collocate if case doesn't match
        'exclusion_words': {
            'login', 'log in', 'logs in', 'crypto', 'token', 'tokens', 'giveaway',
            'bitcoin', 'blockchain', 'app', 'download', 'subscribe', 'registered',
            'signed up', 'lottery', 'referral', 'coupon', 'promo', 'discount'
        },
    },
    'cell': {
        'entity_meaning': 'biological cell / Cell journal',
        'common_meanings': ['prison cell', 'phone cell', 'battery cell'],
        'case_forms': None,              # Case-insensitive ("cell" is valid in biology)
        'collocates': {
            'biology', 'protein', 'gene', 'dna', 'rna', 'membrane', 'stem',
            'cancer', 'tissue', 'research', 'study', 'organism', 'mutation'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'phone', 'prison', 'jail', 'tower', 'signal', 'battery', 'inmate',
            'cell tower', 'cellular network'
        },
    },

    # ---- Space / Astronomy ----
    'mars': {
        'entity_meaning': 'Planet Mars',
        'common_meanings': ['Mars chocolate brand', 'verb "to mar" (mars)'],
        'case_forms': ['Mars', 'MARS'],
        'collocates': {
            'planet', 'space', 'nasa', 'isro', 'rover', 'orbit', 'landing', 'mission',
            'spacecraft', 'atmosphere', 'solar', 'red planet', 'perseverance',
            'curiosity', 'ingenuity', 'crater', 'martian'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'chocolate', 'candy', 'bar', 'snickers', 'wrapper', 'confectionery'
        },
    },

    # ---- Governance ----
    'bill': {
        'entity_meaning': 'legislative bill',
        'common_meanings': ['person name Bill', 'restaurant bill', 'dollar bill'],
        'case_forms': None,              # Lowercase "bill" is valid in legislation
        'collocates': {
            'passed', 'parliament', 'assembly', 'senate', 'congress', 'legislation',
            'amendment', 'act', 'house', 'committee', 'vote', 'law', 'ordinance',
            'rajya sabha', 'lok sabha', 'cabinet', 'gazette'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'restaurant', 'invoice', 'payment', 'waiter', 'tip', 'receipt'
        },
    },

    # ---- Corporate / Finance ----
    'apple': {
        'entity_meaning': 'Apple Inc.',
        'common_meanings': ['fruit apple'],
        'case_forms': ['Apple', 'APPLE'],
        'collocates': {
            'iphone', 'macbook', 'ios', 'ceo', 'stock', 'revenue', 'tech', 'silicon',
            'app store', 'tim cook', 'wwdc', 'ipad', 'safari', 'mac', 'airpods'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'fruit', 'orchard', 'tree', 'juice', 'pie', 'recipe', 'cider', 'harvest'
        },
    },
    'exchange': {
        'entity_meaning': 'stock/crypto exchange',
        'common_meanings': ['verbal exchange', 'exchange of gifts', 'heat exchange'],
        'case_forms': None,
        'collocates': {
            'stock', 'crypto', 'bitcoin', 'trading', 'shares', 'listed', 'market',
            'nse', 'bse', 'binance', 'coinbase', 'demat', 'securities'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'student exchange', 'cultural exchange', 'heat exchange', 'conversation',
            'foreign exchange student'
        },
    },
    'mining': {
        'entity_meaning': 'crypto mining',
        'common_meanings': ['coal/mineral mining', 'data mining'],
        'case_forms': None,
        'collocates': {
            'crypto', 'bitcoin', 'ethereum', 'blockchain', 'hash', 'gpu', 'rig',
            'proof of work', 'token', 'node'
        },
        'min_collocates': 1,
        'exclusion_words': {
            'coal', 'iron ore', 'gold mine', 'quarry', 'mineral', 'underground'
        },
    },

    # ---- Sports vs. Global Affairs ----
    'icc': {
        'entity_meaning': 'International Cricket Council OR International Criminal Court',
        'common_meanings': ['ambiguous abbreviation'],
        'case_forms': ['ICC'],
        'collocates': {
            'cricket', 'match', 'world cup', 'odi', 't20', 'test match', 'bcci',
            'batting', 'bowling', 'squad',
            'court', 'tribunal', 'hague', 'war crimes', 'genocide', 'prosecution'
        },
        'min_collocates': 0,             # ICC as abbreviation is generally valid with case
        'exclusion_words': set(),
    },
}


def _disambiguate_keyword(keyword: str, original_text: str) -> bool:
    """
    Contextual disambiguation for polysemous keywords.

    Determines whether a keyword match is a genuine domain-specific reference
    or a false positive caused by a common English homonym/homophone.

    Uses three orthogonal signals:
      1. Orthographic: Does the original text contain the expected case form? (WHO vs who)
      2. Collocates: Are domain-specific co-occurring words present nearby?
      3. Exclusion: Are words present that indicate the other (non-entity) meaning?

    Args:
        keyword: The keyword that was matched (always lowercase).
        original_text: The ORIGINAL text with preserved case and context.

    Returns:
        True if the keyword is a genuine domain-specific match, False if it's a
        common English word used in its everyday sense.
    """
    if keyword not in AMBIGUOUS_TERMS:
        return True  # Not ambiguous → always genuine

    config = AMBIGUOUS_TERMS[keyword]
    safe_text = original_text or ""
    lower_text = safe_text.lower()

    # Pre-compute: check which case forms (if any) appear in the original text
    case_forms: Optional[List[str]] = config.get('case_forms')
    matched_case_form: Optional[str] = None
    if case_forms:
        for cf in case_forms:
            if re.search(r'\b' + re.escape(cf) + r'\b', safe_text):
                matched_case_form = cf
                break

    # Pre-compute: count domain collocates present
    collocates: Set[str] = config.get('collocates', set())
    min_collocates: int = config.get('min_collocates', 1)
    collocate_count = sum(1 for c in collocates if c in lower_text) if collocates else 0
    has_collocates = collocate_count >= min_collocates if min_collocates > 0 else True

    # ── Signal 1: Exclusion words (strongest negative signal) ──
    # If exclusion words are present, the keyword is very likely used in its common meaning
    exclusion_words: Set[str] = config.get('exclusion_words', set())
    exclusion_hit = False
    if exclusion_words:
        for exc in exclusion_words:
            if exc in lower_text:
                exclusion_hit = True
                break

    if exclusion_hit:
        # Exclusion word found → strong signal for common meaning
        # ALL-UPPERCASE forms (e.g., "WHO") are strong abbreviation signals that
        # can override exclusion, but ONLY if collocates also confirm the domain
        if matched_case_form and matched_case_form.isupper() and has_collocates:
            return True   # "WHO warns about crypto health scams" → accept (WHO + health collocate)
        return False      # "Mars chocolate" or "User Who Logs In crypto" → reject

    # ── Signal 2: Orthographic case analysis (no exclusion present) ──
    if case_forms:
        if matched_case_form:
            return True   # Exact case form found without exclusion → genuine
        # Case form NOT found (e.g., only "who" or "Who" exists)
        # Fall through to collocate check as a safety net
    else:
        # No case requirement (e.g., "bill" is valid lowercase in legislation context)
        # Rely on collocates to confirm
        pass

    # ── Signal 3: Domain collocate validation ──
    if collocates and min_collocates > 0:
        return has_collocates

    # Default: accept if no disambiguation rules triggered a rejection
    return True


# Pre-compiled Indian District Names for automatic *.nic.in resolution
INDIAN_DISTRICTS = [
    'ariyalur', 'chengalpattu', 'chennai', 'coimbatore', 'cuddalore', 'dharmapuri',
    'dindigul', 'erode', 'kallakurichi', 'kanchipuram', 'kanyakumari', 'karur',
    'krishnagiri', 'madurai', 'mayiladuthurai', 'nagapattinam', 'namakkal', 'nilgiris',
    'perambalur', 'pudukkottai', 'ramanathapuram', 'ranipet', 'salem', 'sivaganga',
    'tenkasi', 'thanjavur', 'theni', 'thoothukudi', 'tiruchirappalli', 'trichy',
    'tirunelveli', 'tirupathur', 'tiruppur', 'tiruvallur', 'tiruvannamalai',
    'tiruvarur', 'vellore', 'viluppuram', 'virudhunagar',
    # Key National Districts
    'bengaluru', 'bangalore', 'mumbai', 'delhi', 'hyderabad', 'kolkata', 'ahmedabad',
    'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'patna', 'indore', 'bhopal',
    'visakhapatnam', 'kochi', 'thiruvananthapuram', 'kozhikode', 'guwahati'
]

# -------------------------------------------------------------------------
# Backward-Compatible Authority Registry (Sector-specific authority routing)
# -------------------------------------------------------------------------
class AuthorityRegistryEntry(TypedDict):
    name: str
    role: str
    category: str
    keywords: List[str]
    primary_domains: List[str]
    official_domains: List[str]
    official_handles: List[str]
    trusted_media: List[str]
    factcheck_domains: List[str]


class SectorEntityEntry(TypedDict):
    name: str
    role: str
    category: str
    official_handles: List[str]
    official_domains: List[str]
    associated_figures: List[str]


class CinemaFigureEntry(TypedDict):
    name: str
    handle: str
    role: str


class MovieOriginEntry(TypedDict):
    production_house: str
    producer: str
    lead_actor: str
    associated_handles: List[str]


AUTHORITY_REGISTRY: Dict[str, AuthorityRegistryEntry] = {
    'finance_banking': {
        'name': 'Reserve Bank of India & Financial Regulators',
        'role': 'Central Monetary Authority / Financial Regulator',
        'category': 'finance_banking',
        'keywords': [
            'rbi', 'reserve bank', 'central bank', 'upi', 'npci', 'gst', 'gst council',
            'repo rate', '500 rupee note', '500 note', 'bank locker', 'atm', 'sebi',
            'income tax', 'cbic', 'finmin'
        ],
        'primary_domains': ['rbi.org.in', 'npci.org.in', 'cbic.gov.in', 'finmin.nic.in'],
        'official_domains': ['rbi.org.in', 'npci.org.in', 'cbic.gov.in', 'finmin.nic.in'],
        'official_handles': ['@RBI', '@RBIsays'],
        'trusted_media': ['economictimes.indiatimes.com', 'livemint.com', 'thehindu.com'],
        'factcheck_domains': ['pib.gov.in', 'boomlive.in', 'youturn.in', 'altnews.in']
    },
    'tamil_nadu_govt': {
        'name': 'Government of Tamil Nadu',
        'role': 'State Government & Administration',
        'category': 'tamil_nadu_govt',
        'keywords': [
            'tamil nadu', 'tamilnadu', 'tn govt', 'tn government', 'mk stalin',
            'kalaignar magalir urimai', 'magalir urimai', 'pudhumai penn', 'pongal gift',
            'tnpsc', 'tasmac', 'dipr'
        ],
        'primary_domains': ['tn.gov.in', 'dipr.tn.gov.in', 'tnsocialwelfare.tn.gov.in'],
        'official_domains': ['tn.gov.in', 'dipr.tn.gov.in', 'tnsocialwelfare.tn.gov.in'],
        'official_handles': ['@TNDIPRNEWS', '@CMOTamilnadu'],
        'trusted_media': ['puthiyathalaimurai.com', 'thanthitv.com', 'thehindu.com'],
        'factcheck_domains': ['youturn.in', 'boomlive.in', 'altnews.in', 'factcrescendo.com']
    },
    'tvk_party': {
        'name': 'Tamilaga Vettri Kazhagam (TVK)',
        'role': 'Political Party & Official Headquarters',
        'category': 'tvk_party',
        'keywords': [
            'tvk', 'tamilaga vettri kazhagam', 'thamizhaga vetri kazhagam', 'tvk party',
            'thalapathy vijay', 'tvk flag', 'vikravandi', 'tvkvijayhq'
        ],
        'primary_domains': ['x.com/tvkvijayhq', 'tvk.family'],
        'official_domains': ['tvk.family'],
        'official_handles': ['@tvkvijayhq', '@actorvijay'],
        'trusted_media': ['puthiyathalaimurai.com', 'thanthitv.com', 'thehindu.com'],
        'factcheck_domains': ['youturn.in', 'boomlive.in', 'altnews.in']
    }
}

# Known Multi-Sector Primary Authorities & Official Channels Registry
KNOWN_SECTOR_ENTITIES: Dict[str, SectorEntityEntry] = {
    # Cinema & Entertainment: Production Houses & Creators
    'dream warrior pictures': {
        'name': 'Dream Warrior Pictures',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@DreamWarriorpic', '@prabhu_sr'],
        'official_domains': ['dreamwarriorpictures.com'],
        'associated_figures': ['SR Prabhu', 'SR Prakashbabu']
    },
    'sun pictures': {
        'name': 'Sun Pictures',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@sunpictures'],
        'official_domains': ['sunnetwork.in'],
        'associated_figures': ['Kalanithi Maran']
    },
    'lyca productions': {
        'name': 'Lyca Productions',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@LycaProductions'],
        'official_domains': ['lycaproductions.com'],
        'associated_figures': ['Subaskaran']
    },
    '2d entertainment': {
        'name': '2D Entertainment',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@2D_ENTPVTLTD'],
        'official_domains': ['2dentertainment.in'],
        'associated_figures': ['Suriya', 'Jyotika', 'Rajsekar Pandian']
    },
    'red giant movies': {
        'name': 'Red Giant Movies',
        'role': 'Production House / Distributor',
        'category': 'cinema_entertainment',
        'official_handles': ['@RedGiantMovies_'],
        'official_domains': ['redgiantmovies.in'],
        'associated_figures': ['Udhayanidhi Stalin']
    },
    'studio green': {
        'name': 'Studio Green',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@StudioGreen2'],
        'official_domains': ['studiogreen.in'],
        'associated_figures': ['K.E. Gnanavel Raja']
    },
    'mythri movie makers': {
        'name': 'Mythri Movie Makers',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@MythriOfficial'],
        'official_domains': ['mythriofficial.com'],
        'associated_figures': ['Naveen Yerneni', 'Y. Ravi Shankar']
    },
    'hombale films': {
        'name': 'Hombale Films',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@hombalefilms'],
        'official_domains': ['hombalefilms.com'],
        'associated_figures': ['Vijay Kiragandur']
    },
    'dharma productions': {
        'name': 'Dharma Productions',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@DharmaMovies'],
        'official_domains': ['dharma-production.com'],
        'associated_figures': ['Karan Johar']
    },
    'yash raj films': {
        'name': 'Yash Raj Films',
        'role': 'Production House',
        'category': 'cinema_entertainment',
        'official_handles': ['@yrf'],
        'official_domains': ['yashrajfilms.com'],
        'associated_figures': ['Aditya Chopra']
    },

    # Governance & Regulatory Bodies
    'reserve bank of india': {
        'name': 'Reserve Bank of India (RBI)',
        'role': 'Central Monetary Authority',
        'category': 'finance_banking',
        'official_handles': ['@RBI', '@RBIsays'],
        'official_domains': ['rbi.org.in', 'npci.org.in', 'cbic.gov.in'],
        'associated_figures': ['Governor, RBI']
    },
    'rbi': {
        'name': 'Reserve Bank of India (RBI)',
        'role': 'Central Monetary Authority',
        'category': 'finance_banking',
        'official_handles': ['@RBI', '@RBIsays'],
        'official_domains': ['rbi.org.in', 'npci.org.in'],
        'associated_figures': ['Governor, RBI']
    },
    'pmo india': {
        'name': 'Prime Minister\'s Office (PMO)',
        'role': 'Head of Government',
        'category': 'governance_state_national',
        'official_handles': ['@PMOIndia', '@narendramodi'],
        'official_domains': ['pmindia.gov.in', 'pib.gov.in'],
        'associated_figures': ['Prime Minister']
    },
    'narendra modi': {
        'name': 'Prime Minister\'s Office (PMO) / Narendra Modi',
        'role': 'Head of Government / Prime Minister',
        'category': 'governance_state_national',
        'official_handles': ['@narendramodi', '@PMOIndia'],
        'official_domains': ['pmindia.gov.in', 'pib.gov.in'],
        'associated_figures': ['Prime Minister Narendra Modi']
    },
    'pm modi': {
        'name': 'Prime Minister\'s Office (PMO) / Narendra Modi',
        'role': 'Head of Government / Prime Minister',
        'category': 'governance_state_national',
        'official_handles': ['@narendramodi', '@PMOIndia'],
        'official_domains': ['pmindia.gov.in', 'pib.gov.in'],
        'associated_figures': ['Prime Minister Narendra Modi']
    },
    'chennai metro': {
        'name': 'Chennai Metro Rail Limited (CMRL)',
        'role': 'Metro Rail Transit Authority',
        'category': 'tamil_nadu_govt',
        'official_handles': ['@cmrlofficial'],
        'official_domains': ['chennaimetrorail.org', 'tn.gov.in'],
        'associated_figures': ['Managing Director, CMRL']
    },
    'cmrl': {
        'name': 'Chennai Metro Rail Limited (CMRL)',
        'role': 'Metro Rail Transit Authority',
        'category': 'tamil_nadu_govt',
        'official_handles': ['@cmrlofficial'],
        'official_domains': ['chennaimetrorail.org', 'tn.gov.in'],
        'associated_figures': ['Managing Director, CMRL']
    },
    'government of tamil nadu': {
        'name': 'Government of Tamil Nadu',
        'role': 'State Government & Administration',
        'category': 'tamil_nadu_govt',
        'official_handles': ['@TNDIPRNEWS', '@CMOTamilnadu'],
        'official_domains': ['tn.gov.in', 'dipr.tn.gov.in', 'tnsocialwelfare.tn.gov.in'],
        'associated_figures': ['Chief Minister MK Stalin']
    },
    'cm vijay': {
        'name': 'Tamilaga Vettri Kazhagam (TVK) / Thalapathy Vijay',
        'role': 'Political Party & Public Leader',
        'category': 'tvk_party',
        'official_handles': ['@tvkvijayhq', '@actorvijay'],
        'official_domains': ['tvk.family'],
        'associated_figures': ['Thalapathy Vijay']
    },
    'tamilaga vettri kazhagam': {
        'name': 'Tamilaga Vettri Kazhagam (TVK)',
        'role': 'Political Party & Official Headquarters',
        'category': 'tvk_party',
        'official_handles': ['@tvkvijayhq', '@actorvijay'],
        'official_domains': ['tvk.family'],
        'associated_figures': ['Thalapathy Vijay']
    },
    'tvk': {
        'name': 'Tamilaga Vettri Kazhagam (TVK)',
        'role': 'Political Party & Official Headquarters',
        'category': 'tvk_party',
        'official_handles': ['@tvkvijayhq', '@actorvijay'],
        'official_domains': ['tvk.family'],
        'associated_figures': ['Thalapathy Vijay']
    },

    # Science & Space
    'isro': {
        'name': 'Indian Space Research Organisation (ISRO)',
        'role': 'National Space Agency',
        'category': 'space_astronomy',
        'official_handles': ['@isro'],
        'official_domains': ['isro.gov.in'],
        'associated_figures': ['Chairman, ISRO']
    },
    'nasa': {
        'name': 'National Aeronautics and Space Administration (NASA)',
        'role': 'Civilian Space Agency',
        'category': 'space_astronomy',
        'official_handles': ['@NASA'],
        'official_domains': ['nasa.gov'],
        'associated_figures': ['NASA Administrator']
    },

    # Health & Medicine
    'world health organization': {
        'name': 'World Health Organization (WHO)',
        'role': 'International Public Health Agency',
        'category': 'health_medical',
        'official_handles': ['@WHO', '@DrTedros'],
        'official_domains': ['who.int'],
        'associated_figures': ['Director-General WHO']
    },
    'who': {
        'name': 'World Health Organization (WHO)',
        'role': 'International Public Health Agency',
        'category': 'health_medical',
        'official_handles': ['@WHO'],
        'official_domains': ['who.int'],
        'associated_figures': ['Director-General WHO']
    },

    'icmr': {
        'name': 'Indian Council of Medical Research (ICMR)',
        'role': 'Apex Medical Research Council',
        'category': 'health_medical',
        'official_handles': ['@ICMRDELHI'],
        'official_domains': ['icmr.gov.in', 'mohfw.gov.in'],
        'associated_figures': ['Director General, ICMR']
    },

    # Sports
    'bcci': {
        'name': 'Board of Control for Cricket in India (BCCI)',
        'role': 'National Cricket Governing Body',
        'category': 'sports',
        'official_handles': ['@BCCI'],
        'official_domains': ['bcci.tv'],
        'associated_figures': ['BCCI President', 'BCCI Secretary']
    },
    'icc': {
        'name': 'International Cricket Council (ICC)',
        'role': 'Global Cricket Governing Body',
        'category': 'sports',
        'official_handles': ['@ICC'],
        'official_domains': ['icc-cricket.com'],
        'associated_figures': ['ICC Chair']
    },

    # Corporate & Tech
    'openai': {
        'name': 'OpenAI',
        'role': 'AI Research & Deployment Corporation',
        'category': 'corporate_business',
        'official_handles': ['@OpenAI', '@sama'],
        'official_domains': ['openai.com'],
        'associated_figures': ['Sam Altman']
    },
    'tata': {
        'name': 'Tata Group',
        'role': 'Multinational Conglomerate',
        'category': 'corporate_business',
        'official_handles': ['@TataCompanies'],
        'official_domains': ['tata.com'],
        'associated_figures': ['Tata Sons Chairman']
    },
    'reliance': {
        'name': 'Reliance Industries',
        'role': 'Energy & Telecommunications Conglomerate',
        'category': 'corporate_business',
        'official_handles': ['@RIL_Updates'],
        'official_domains': ['ril.com'],
        'associated_figures': ['Mukesh Ambani']
    }
}

# Prominent Cinema Public Figures & Handles Mapping
CINEMA_FIGURES_HANDLES: Dict[str, CinemaFigureEntry] = {
    'suriya': {'name': 'Suriya Sivakumar', 'handle': '@Suriya_offl', 'role': 'Lead Actor & Producer'},
    'actor suriya': {'name': 'Suriya Sivakumar', 'handle': '@Suriya_offl', 'role': 'Lead Actor & Producer'},
    'vijay': {'name': 'Thalapathy Vijay', 'handle': '@actorvijay', 'role': 'Lead Actor & Leader'},
    'rajinikanth': {'name': 'Superstar Rajinikanth', 'handle': '@rajinikanth', 'role': 'Lead Actor'},
    'kamal haasan': {'name': 'Kamal Haasan', 'handle': '@ikamalhaasan', 'role': 'Lead Actor & Producer'},
    'ajith': {'name': 'Ajith Kumar', 'handle': '@SureshChandraa', 'role': 'Lead Actor'},
    'ajith kumar': {'name': 'Ajith Kumar', 'handle': '@SureshChandraa', 'role': 'Lead Actor'},
    'dhanush': {'name': 'Dhanush', 'handle': '@dhanushkraja', 'role': 'Lead Actor & Director'},
    'sivakarthikeyan': {'name': 'Sivakarthikeyan', 'handle': '@Siva_Kartikeyan', 'role': 'Lead Actor & Producer'},
    'karthi': {'name': 'Karthi', 'handle': '@Karthi_Offl', 'role': 'Lead Actor'},
    'lokesh kanagaraj': {'name': 'Lokesh Kanagaraj', 'handle': '@Dir_Lokesh', 'role': 'Director & Screenwriter'},
    'nelson dilipkumar': {'name': 'Nelson Dilipkumar', 'handle': '@Nelsondilpkumar', 'role': 'Director'},
    'atlee': {'name': 'Atlee Kumar', 'handle': '@Atlee_dir', 'role': 'Director & Producer'},
    'shankar': {'name': 'S. Shankar', 'handle': '@shankarshanmugh', 'role': 'Director & Producer'},
    'anirudh': {'name': 'Anirudh Ravichander', 'handle': '@anirudhofficial', 'role': 'Music Composer'},
    'sr prabhu': {'name': 'S.R. Prabhu', 'handle': '@prabhu_sr', 'role': 'Film Producer (Dream Warrior Pictures)'},
    'sr prabu': {'name': 'S.R. Prabhu', 'handle': '@prabhu_sr', 'role': 'Film Producer (Dream Warrior Pictures)'}
}

# Movie Title Knowledge Mapping (For automatic origin linkage)
MOVIE_ORIGIN_MAP: Dict[str, MovieOriginEntry] = {
    'karuppu': {
        'production_house': 'Dream Warrior Pictures',
        'producer': 'SR Prabhu',
        'lead_actor': 'Suriya',
        'associated_handles': ['@DreamWarriorpic', '@prabhu_sr', '@Suriya_offl']
    },
    'kanguva': {
        'production_house': 'Studio Green',
        'producer': 'K.E. Gnanavel Raja',
        'lead_actor': 'Suriya',
        'associated_handles': ['@StudioGreen2', '@Suriya_offl']
    },
    'leo': {
        'production_house': 'Seven Screen Studio',
        'producer': 'SS Lalit Kumar',
        'lead_actor': 'Thalapathy Vijay',
        'associated_handles': ['@7screenstudio', '@actorvijay']
    },
    'jailer': {
        'production_house': 'Sun Pictures',
        'producer': 'Kalanithi Maran',
        'lead_actor': 'Rajinikanth',
        'associated_handles': ['@sunpictures', '@rajinikanth']
    },
    'coolie': {
        'production_house': 'Sun Pictures',
        'producer': 'Kalanithi Maran',
        'lead_actor': 'Rajinikanth',
        'associated_handles': ['@sunpictures', '@rajinikanth', '@Dir_Lokesh']
    },
    'goat': {
        'production_house': 'AGS Entertainment',
        'producer': 'Archana Kalpathi',
        'lead_actor': 'Thalapathy Vijay',
        'associated_handles': ['@Ags_production', '@actorvijay']
    }
}


# -------------------------------------------------------------------------
# Universal Primary Origin Resolver
# -------------------------------------------------------------------------

class UniversalOriginResolver:
    """
    Universal Dynamic Origin Resolver for all news sectors.
    Extracts the claimant/originating entity and resolves primary official sources,
    district administration portals, journal repositories, and news wires dynamically.
    """

    @classmethod
    def infer_category(cls, text: str, entities: Optional[List[str]] = None) -> str:
        """
        Infer the domain category of any news statement across all sectors.
        """
        safe_text = text or ""
        lower_t = safe_text.lower()
        if entities:
            lower_t += " " + " ".join([e.lower() for e in entities if e])

        category_scores: Dict[str, int] = {}
        for cat, keywords in DOMAIN_CATEGORY_KEYWORDS.items():
            score = sum(
                1 for kw in keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_t)
                and _disambiguate_keyword(kw, safe_text)
            )
            if score > 0:
                category_scores[cat] = score

        if category_scores:
            return max(category_scores, key=lambda k: category_scores[k])

        return 'general'

    @classmethod
    def extract_claimant_and_district(
        cls,
        text: str,
        entities: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the primary claimant organization/person and any referenced district.
        """
        safe_text = text or ""
        lower_t = safe_text.lower()

        # 1. Check for Indian District Name ONLY if text context relates to district administration
        # (e.g. collector, rain holiday, school holiday, section 144, curfew, flood alert)
        district_admin_cues = (
            'collector', 'district magistrate', 'dm', 'collectorate', 'taluk', 'tahsildar',
            'tehsildar', 'rdo', 'school holiday', 'college holiday', 'rain holiday',
            'flood alert', 'red alert', 'curfew', 'section 144', 'district administration',
            'local body', 'zila parishad', 'district collector'
        )
        is_district_admin_matter = any(re.search(r'\b' + re.escape(c) + r'\b', lower_t) for c in district_admin_cues)

        detected_district: Optional[str] = None
        if is_district_admin_matter:
            for dist in INDIAN_DISTRICTS:
                if re.search(r'\b' + re.escape(dist) + r'\b', lower_t):
                    detected_district = dist
                    break

        # 2. Extract Claimant (Who announced, declared, ordered, issued, or inaugurated this?)
        detected_claimant: Optional[str] = None

        speech_pattern = (
            r'([A-Z][A-Za-z0-9\s\.\-]{2,40}?)\s+'
            r'(?:announced|declares?|declared|issued|published|ordered|approved|'
            r'launched|warned|confirmed|stated|reported|shared|'
            r'inaugurated?|inaugurates?|opened?|unveiled?|flags?\s+off|flagged\s+off|dedicated|commissioned)'
        )
        speech_match = re.search(speech_pattern, safe_text)
        if speech_match:
            candidate = speech_match.group(1).strip()
            if candidate.lower() not in ('today', 'yesterday', 'breaking news', 'reports', 'sources', 'news'):
                detected_claimant = candidate

        # Fallback to recognized institutional or authority names in entities
        institutional_markers = (
            'ministry', 'department', 'collector', 'commission', 'council',
            'bank', 'board', 'agency', 'organization', 'organisation',
            'isro', 'nasa', 'icmr', 'un', 'court', 'police',
            'government', 'govt', 'tvk', 'pib', 'party', 'productions',
            'studios', 'pictures', 'hospital', 'university', 'institute', 'bcci', 'icc'
        )
        if not detected_claimant and entities:
            for ent in entities:
                if not ent:
                    continue
                ent_clean = ent.strip()
                ent_lower = ent_clean.lower()
                if any(marker in ent_lower for marker in institutional_markers):
                    detected_claimant = ent_clean
                    break

        return detected_claimant, detected_district

    @classmethod
    def resolve_primary_origins(
        cls,
        text: Optional[str],
        entities: Optional[List[str]] = None,
        ocr_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-Source Primary Entity Detection across ALL sectors.
        Identifies all potential original sources (e.g. production houses, producers,
        lead actors, district authorities, ministries, sports boards, corporations)
        and constructs their official handles, domains, and targeted search queries.

        Args:
            text: Claim text, headline, or caption.
            entities: Extracted entity tokens.
            ocr_data: Optional OCR extraction data from uploaded image/poster.

        Returns:
            List of PrimaryOriginEntity dictionaries.
        """
        if not text and not (ocr_data and ocr_data.get('extracted_text')):
            return []

        combined_text = (text or "") + " "
        if ocr_data and ocr_data.get('extracted_text'):
            combined_text += ocr_data['extracted_text'] + " "
        if entities:
            combined_text += " ".join(entities)

        lower_t = combined_text.lower()
        category = cls.infer_category(combined_text, entities)
        primary_origins: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()

        # ---------------------------------------------------------------------
        # 1. Check Known Sector Entities Registry (All Sectors)
        # ---------------------------------------------------------------------
        for key, info in KNOWN_SECTOR_ENTITIES.items():
            if re.search(r'\b' + re.escape(key) + r'\b', lower_t):
                # Contextual disambiguation: reject false matches for homonymous keys
                if not _disambiguate_keyword(key, text or combined_text):
                    continue
                norm_name = info['name'].lower()
                if norm_name not in seen_names and key not in seen_names:
                    seen_names.add(norm_name)
                    seen_names.add(key)
                    # Also register any parenthetical abbreviation e.g. (RBI), (WHO), (BCCI)
                    abbr_match = re.search(r'\(([A-Za-z0-9]+)\)', info['name'])
                    if abbr_match:
                        seen_names.add(abbr_match.group(1).lower())
                    handles = list(info.get('official_handles', []))
                    domains = list(info.get('official_domains', []))
                    queries = cls._build_entity_queries(
                        info['name'],
                        handles,
                        combined_text,
                        domains=domains
                    )
                    primary_origins.append({
                        'key': key,
                        'name': info['name'],
                        'role': info.get('role', 'Primary Authority'),
                        'category': info.get('category', category),
                        'official_handles': handles,
                        'official_domains': domains,
                        'associated_figures': list(info.get('associated_figures', [])),
                        'search_queries': queries
                    })

        # ---------------------------------------------------------------------
        # 2. Cinema & Entertainment Context-Aware Multi-Source Resolution
        # ---------------------------------------------------------------------
        # Check for movie title mappings (e.g. Karuppu -> Dream Warrior Pictures, SR Prabhu, Suriya)
        for m_title, m_info in MOVIE_ORIGIN_MAP.items():
            if re.search(r'\b' + re.escape(m_title) + r'\b', lower_t):
                prod_house = m_info.get('production_house')
                if prod_house and prod_house not in seen_names:
                    seen_names.add(prod_house)
                    p_key = prod_house.lower()
                    p_reg = KNOWN_SECTOR_ENTITIES.get(p_key)
                    handles = (
                        list(p_reg['official_handles'])
                        if p_reg
                        else list(m_info.get('associated_handles', []))
                    )
                    domains = list(p_reg['official_domains']) if p_reg else []
                    queries = cls._build_entity_queries(prod_house, handles, combined_text, movie_title=m_title, domains=domains)
                    associated_figures = [
                        fig for fig in (m_info.get('producer'), m_info.get('lead_actor')) if fig
                    ]
                    primary_origins.append({
                        'key': p_key,
                        'name': prod_house,
                        'role': 'Production House (Original Publisher)',
                        'category': 'cinema_entertainment',
                        'official_handles': handles,
                        'official_domains': domains,
                        'associated_figures': associated_figures,
                        'search_queries': queries
                    })

                producer = m_info.get('producer')
                if producer and producer not in seen_names:
                    seen_names.add(producer)
                    pr_handle = [
                        h for h in m_info.get('associated_handles', [])
                        if 'prabhu' in h.lower() or 'lalit' in h.lower()
                    ]
                    queries = cls._build_entity_queries(producer, pr_handle, combined_text, movie_title=m_title)
                    primary_origins.append({
                        'key': producer.lower(),
                        'name': producer,
                        'role': 'Film Producer (Primary Official)',
                        'category': 'cinema_entertainment',
                        'official_handles': pr_handle,
                        'official_domains': [],
                        'associated_figures': [],
                        'search_queries': queries
                    })

        # Check for prominent cinema figures (Actors, Directors)
        for fig_key, fig_info in CINEMA_FIGURES_HANDLES.items():
            if re.search(r'\b' + re.escape(fig_key) + r'\b', lower_t):
                if fig_info['name'] not in seen_names:
                    seen_names.add(fig_info['name'])
                    queries = cls._build_entity_queries(fig_info['name'], [fig_info['handle']], combined_text)
                    primary_origins.append({
                        'key': fig_key,
                        'name': fig_info['name'],
                        'role': fig_info['role'],
                        'category': 'cinema_entertainment',
                        'official_handles': [fig_info['handle']],
                        'official_domains': [],
                        'associated_figures': [],
                        'search_queries': queries
                    })

        # ---------------------------------------------------------------------
        # 3. District Administration Resolution (*.nic.in)
        # ---------------------------------------------------------------------
        detected_claimant, detected_district = cls.extract_claimant_and_district(combined_text, entities)
        if detected_district:
            dist_name = f"{detected_district.title()} District Administration"
            if dist_name not in seen_names:
                seen_names.add(dist_name)
                dist_domain = f"{detected_district}.nic.in"
                dist_queries = [
                    f"site:{dist_domain} official circular OR holiday OR order",
                    f"site:x.com/Collector{detected_district.title()} official",
                    f"{detected_district} district collector official notification"
                ]
                primary_origins.append({
                    'key': f"district_{detected_district}",
                    'name': dist_name,
                    'role': 'Local District Administration Authority',
                    'category': 'district_local_gov',
                    'official_handles': [f"@Collector{detected_district.title()}"],
                    'official_domains': [dist_domain],
                    'associated_figures': [f"Collector of {detected_district.title()}"],
                    'search_queries': dist_queries
                })

        # ---------------------------------------------------------------------
        # 4. Dynamic Claimant / Issuing Organization Resolution
        # ---------------------------------------------------------------------
        if detected_claimant:
            c_low = detected_claimant.strip().lower()
            if c_low not in seen_names and not any(c_low in s or s in c_low for s in seen_names):
                clean_claimant = detected_claimant.strip()
                if len(clean_claimant) > 2 and c_low not in ('reports', 'breaking'):
                    seen_names.add(c_low)
                    topic_tail = cls._extract_topic_terms(combined_text, exclude_text=clean_claimant, limit=4)
                    if topic_tail:
                        queries = [
                            f'"{clean_claimant}" {topic_tail}',
                            f'{clean_claimant} {topic_tail} fact check',
                        ]
                    else:
                        queries = [
                            f'"{clean_claimant}" official statement OR announcement OR circular',
                        ]
                    primary_origins.append({
                        'key': clean_claimant.lower(),
                        'name': clean_claimant,
                        'role': 'Issuing Entity / Claimant',
                        'category': category,
                        'official_handles': [],
                        'official_domains': [],
                        'associated_figures': [],
                        'search_queries': queries
                    })

        # ---------------------------------------------------------------------
        # 5. Sports Sector Resolution (BCCI, ICC, Teams)
        # ---------------------------------------------------------------------
        if category == 'sports' and not primary_origins:
            primary_origins.append({
                'key': 'sports_governing_body',
                'name': 'BCCI / Official Sports Federation',
                'role': 'Official Sports Regulatory Board',
                'category': 'sports',
                'official_handles': ['@BCCI', '@ICC'],
                'official_domains': ['bcci.tv', 'icc-cricket.com'],
                'associated_figures': [],
                'search_queries': ['site:bcci.tv press release', 'site:x.com/BCCI official']
            })

        return primary_origins

    @classmethod
    def _extract_topic_terms(
        cls,
        context_text: str,
        exclude_text: str = "",
        limit: int = 5
    ) -> str:
        """
        Extract discriminative topic keywords in original sentence order without
        truncating late identifiers (e.g., K2-18b, 1969, 500, curvature).
        """
        if not context_text or not context_text.strip():
            return ""

        stopwords = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'a', 'an', 'and', 'or',
            'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as', 'that', 'this',
            'it', 'its', 'they', 'them', 'their', 'his', 'her', 'which', 'who', 'whom', 'whose',
            'what', 'when', 'where', 'why', 'how', 'not', 'no', 'all', 'every', 'about', 'into',
            'over', 'after', 'before', 'under', 'between', 'also', 'very', 'just', 'only',
            'more', 'most', 'some', 'such', 'announced', 'declared', 'issued', 'confirmed',
            'detected', 'spotted', 'reported', 'stated', 'said', 'claimed', 'claims', 'claim',
            'shows', 'show', 'truth', 'recent', 'recently', 'covering', 'starting',
            'today', 'yesterday', 'tomorrow', 'official', 'has', 'have', 'had', 'will',
            'would', 'can', 'could', 'may', 'might', 'should', 'up', 'out', 'new'
        }

        # Exclude short agency acronyms or short entity names from duplicate prefixing
        exclude_tokens: Set[str] = set()
        if exclude_text:
            abbr = re.search(r'\(([A-Za-z0-9]{2,12})\)', exclude_text)
            if abbr:
                exclude_tokens.add(abbr.group(1).lower())
            base_excl = re.sub(r'\([^)]*\)', '', exclude_text).strip()
            base_words = [w.lower() for w in re.findall(r"[A-Za-z0-9\u0B80-\u0BFF]+", base_excl)]
            if len(base_words) <= 2:
                exclude_tokens.update(base_words)
            else:
                # For long institution names (e.g. National Aeronautics and Space Administration),
                # avoid stripping common science/topic words like 'space' or 'india'
                exclude_tokens.update(w for w in base_words if len(w) <= 5 and w not in ('space', 'india', 'earth', 'health', 'bank'))

        tokens = re.findall(r"[A-Za-z0-9\u0B80-\u0BFF]+(?:-[A-Za-z0-9]+)*", context_text)
        candidates: List[Tuple[int, int, str]] = []
        seen_low: Set[str] = set()
        for idx, tok in enumerate(tokens):
            low = tok.lower()
            if len(low) <= 1 and not tok.isdigit():
                continue
            if low in stopwords or low in exclude_tokens or low in seen_low:
                continue
            seen_low.add(low)
            has_alnum_code = bool(re.search(r'[A-Za-z].*\d|\d.*[A-Za-z]', tok))
            has_long_num = tok.isdigit() and len(tok) >= 3
            is_proper = tok[0].isupper()
            # tier: 0 = mixed alphanumeric code (K2-18b, COVID-19), 1 = proper noun / large number, 2 = content word
            tier = 0 if has_alnum_code else (1 if (is_proper or has_long_num) else 2)
            candidates.append((tier, idx, tok))

        if not candidates:
            return ""

        # Reserve slots for identifiers and key proper nouns so end-of-sentence codes (K2-18b) stay in
        selected: List[Tuple[int, int, str]] = []
        id_cands = [c for c in candidates if c[0] == 0]
        prop_cands = [c for c in candidates if c[0] == 1]
        other_cands = [c for c in candidates if c[0] == 2]

        for c in id_cands[:min(2, limit)]:
            selected.append(c)
        prop_cap = max(1, min(2, limit - len(selected)))
        for c in prop_cands:
            if len(selected) >= limit:
                break
            if len([s for s in selected if s[0] == 1]) < prop_cap:
                selected.append(c)
        for c in other_cands + [c for c in prop_cands if c not in selected] + [c for c in id_cands if c not in selected]:
            if len(selected) >= limit:
                break
            if c not in selected:
                selected.append(c)

        selected.sort(key=lambda c: c[1])
        return " ".join(c[2] for c in selected)

    @classmethod
    def _build_entity_queries(
        cls,
        entity_name: str,
        handles: List[str],
        context_text: str,
        movie_title: Optional[str] = None,
        domains: Optional[List[str]] = None
    ) -> List[str]:
        """Generate high-priority topic-anchored queries targeted at official entity domains or channels."""
        queries: List[str] = []

        # Check if context concerns a public speech, press meet, rally, interview, or inauguration/event
        is_event = bool(re.search(
            r'\b(speech|spoke|speaks|speaking|said|stated|press\s*meet|press\s*conference|interview|'
            r'rally|addressed|addressing|parliamentary|parliament|assembly|bhashan|'
            r'inaugurat\w*|launch\w*|opened|opens|unveil\w*|flag\w*\s+off|dedicat\w*|commission\w*|'
            r'stretch|metro|phase\s+[ivx0-9]+|infrastructure|'
            r'பேச்சு|பேசியது|செய்தியாளர்\s*சந்திப்பு|பேட்டி)\b',
            context_text,
            re.IGNORECASE
        ))

        if is_event:
            # Events, public speeches & press meets are documented live by TV news channels and accredited newspapers
            queries.append(f'"{entity_name}" news')
            queries.append(f'"{entity_name}" live news updates')
            queries.append(
                f'"{entity_name}" (site:thehindu.com OR site:puthiyathalaimurai.com OR '
                f'site:thanthitv.com OR site:polimernews.com OR site:dinamalar.com OR '
                f'site:indianexpress.com OR site:ndtv.com OR site:etvbharat.com)'
            )
            queries.append(f'"{entity_name}" fact check')
            return list(dict.fromkeys(queries))[:5]

        # Extract salient keywords from context (e.g. "340", "crores", "holiday", "teaser")
        num_match = re.findall(r'\b\d+\+?\s*(?:crores?|cr|lakhs?|billion|million|holiday|launch|release)\b', context_text, re.IGNORECASE)
        keyword_token = num_match[0] if num_match else ""

        abbr_m = re.search(r'\(([A-Za-z0-9]{2,12})\)', entity_name)
        short_name = abbr_m.group(1) if abbr_m else entity_name
        topic_str = cls._extract_topic_terms(context_text, exclude_text=entity_name, limit=4)
        # Keep site: queries ultra-focused (2 terms; Prefer single code identifier if present e.g. K2-18b)
        code_tokens = re.findall(r'\b[A-Za-z]+[0-9]+-[A-Za-z0-9]+\b|\b[A-Za-z]+\d+-[A-Za-z0-9]+\b|\bK\d+-\d+[A-Za-z]*\b', context_text)
        if code_tokens:
            site_topic = code_tokens[0]
        else:
            site_topic = cls._extract_topic_terms(context_text, exclude_text=entity_name, limit=2)

        title_str = f'"{movie_title}"' if movie_title else ""

        # 1. Domain-first queries for institutions with official web portals (gov, nasa.gov, rbi.org.in, who.int)
        if domains and not movie_title:
            primary_dom = domains[0].strip()
            dom_kw = site_topic or keyword_token or topic_str
            if primary_dom and dom_kw:
                queries.append(f'site:{primary_dom} {dom_kw}')
            if topic_str:
                queries.append(f'{short_name} {topic_str}'.strip())

        # 2. Queries directed at official social handles (only topic-anchored; avoid blind "official announcement")
        if handles and (movie_title or not domains or any(d.lower().endswith('tvk.family') for d in (domains or []))):
            for h in handles[:1]:
                h_clean = h.lstrip('@')
                if keyword_token and title_str:
                    queries.append(f'site:x.com/{h_clean} {title_str} {keyword_token}')
                elif title_str:
                    queries.append(f'site:x.com/{h_clean} {title_str} official')
                elif keyword_token:
                    queries.append(f'site:x.com/{h_clean} {keyword_token}')
                elif site_topic:
                    queries.append(f'site:x.com/{h_clean} {site_topic}')

        # 3. Query for official statements / posters / press releases
        if title_str and keyword_token:
            queries.append(f'"{entity_name}" {title_str} {keyword_token} (official OR statement OR poster)')
        elif title_str:
            queries.append(f'"{entity_name}" {title_str} (official OR statement OR poster OR "box office")')
        elif topic_str:
            queries.append(f'{short_name} {topic_str}'.strip())
        else:
            queries.append(f'"{entity_name}" official (circular OR notification OR statement OR "press release")')

        # 4. Discovery query for the actual official milestone/number
        lower_ctx = context_text.lower()
        if title_str and ('crore' in lower_ctx or 'box office' in lower_ctx or 'collection' in lower_ctx):
            queries.append(f'"{entity_name}" {title_str} (collection OR "box office" OR "official poster")')

        return list(dict.fromkeys(queries))[:4]

    @classmethod
    def generate_universal_queries(
        cls,
        claim_text: str,
        entities: Optional[List[str]] = None,
        max_queries: int = 4
    ) -> List[str]:
        """
        Generate targeted search queries specifically directed at primary sources,
        official government gazettes/circulars, research journals, or news wires.
        """
        if not claim_text or not claim_text.strip():
            return []

        category = cls.infer_category(claim_text, entities)
        claimant, district = cls.extract_claimant_and_district(claim_text, entities)

        clean_claim = re.sub(r'[^\w\s\-\u0B80-\u0BFF]', ' ', claim_text).strip()
        words = clean_claim.split()
        core_snippet = cls._extract_topic_terms(claim_text, limit=6) or " ".join(words[:10])

        queries: List[str] = []

        # 1. Check Primary Origins first
        primary_origins = cls.resolve_primary_origins(claim_text, entities=entities)
        for po in primary_origins:
            for q in po.get('search_queries', []):
                if q not in queries:
                    queries.append(q)
                if len(queries) >= 3:
                    break
            if len(queries) >= 3:
                break

        # 2. District / Local Administration
        if district:
            dist_domain = f"site:{district}.nic.in"
            queries.append(f"{core_snippet} ({dist_domain} OR site:gov.in)")

        # 3. Category-Specific Primary Origins
        if category == 'space_astronomy':
            queries.append(f"{core_snippet} (site:isro.gov.in OR site:nasa.gov OR site:esa.int)")
            queries.append(f"{core_snippet} (site:space.com OR site:nature.com)")

        elif category == 'health_medical':
            queries.append(f"{core_snippet} (site:who.int OR site:icmr.gov.in OR site:mohfw.gov.in OR site:nih.gov)")
            queries.append(f"{core_snippet} (site:thelancet.com OR site:nature.com OR site:pubmed.ncbi.nlm.nih.gov)")

        elif category == 'agriculture_farming':
            queries.append(f"{core_snippet} (site:agricoop.nic.in OR site:icar.org.in OR site:pib.gov.in)")

        elif category == 'cinema_entertainment':
            if claimant:
                queries.append(f'{claimant} "{core_snippet}" (official statement OR press release OR "x.com")')
            queries.append(f"{core_snippet} (site:thehindu.com OR site:puthiyathalaimurai.com OR site:variety.com)")

        elif category == 'tamil_nadu_govt':
            queries.append(f"{core_snippet} (site:tn.gov.in OR site:dipr.tn.gov.in OR site:tnsocialwelfare.tn.gov.in)")

        elif category == 'tvk_party':
            queries.append(f'{core_snippet} (site:x.com/tvkvijayhq OR "tvkvijayhq")')

        elif category == 'finance_banking':
            queries.append(f"{core_snippet} (site:rbi.org.in OR site:npci.org.in OR site:cbic.gov.in OR site:finmin.nic.in)")

        elif category == 'corporate_business':
            queries.append(f"{core_snippet} (site:reuters.com OR site:bloomberg.com OR site:economictimes.indiatimes.com)")

        elif category == 'sports':
            queries.append(f"{core_snippet} (site:bcci.tv OR site:icc-cricket.com OR site:espncricinfo.com)")

        # 4. Accredited Fact-Checkers & News Wire Query (Worldwide coverage)
        wire_fc_filter = (
            "site:pib.gov.in OR site:youturn.in OR site:boomlive.in OR "
            "site:altnews.in OR site:ptinews.com OR site:aninews.in OR site:reuters.com"
        )
        queries.append(f"{core_snippet} ({wire_fc_filter})")

        return list(dict.fromkeys(queries))[:max_queries]


# -------------------------------------------------------------------------
# Algorithmic Universal Domain Authority Scorer with Primary Origin Priority
# -------------------------------------------------------------------------

def _host_matches_domain(hostname: str, domain: str) -> bool:
    """Return True if hostname equals domain or is a strict subdomain of it."""
    h = (hostname or '').lower().strip('.')
    d = (domain or '').lower().strip('.')
    return bool(h and d and (h == d or h.endswith('.' + d)))


def get_domain_authority_score(
    url: Optional[str],
    source_type: str = "web_search",
    primary_entities: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Algorithmic Domain Credibility Scorer:
    Evaluates any URL worldwide across 6 credibility tiers with strict primary entity weighting.

    Tier 1 (0.99): Verified Primary Origin (Official Handle or Official Portal of Identified Primary Entities)
                   OR Official Government / Central Bank / Intergovernmental / Peer-Reviewed Journals
    Tier 2 (0.95): Accredited Fact-Checkers (IFCN Certified Worldwide)
    Tier 3 (0.92): Primary News Wire Services (PTI, ANI, Reuters, AP, AFP, Bloomberg)
    Tier 4 (0.90): Accredited Regional & National Mainstream Media (The Hindu, Puthiya Thalaimurai, etc.)
    Tier 5 (0.85): Peer-Reviewed Curated Reference (Britannica)
    Tier 6 (0.45): Unverified Social Media (Generic posts on X, Reddit, Facebook, Instagram) or personal blogs
    Disallowed (0.0): Open-edit wikis (Wikipedia, Wikimedia)
    """
    if not url:
        return 0.50

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()
        path = (parsed.path or '').lower()
    except Exception:
        hostname = ''
        path = ''

    if not hostname:
        return 0.50

    social_hosts = {
        'x.com', 'twitter.com', 'mobile.twitter.com', 'mobile.x.com',
        'instagram.com', 'facebook.com'
    }
    is_social_host = any(_host_matches_domain(hostname, sh) for sh in social_hosts)
    path_segments = [seg for seg in path.split('/') if seg]
    first_seg = path_segments[0].lower() if path_segments else ''

    # ---------------------------------------------------------------------
    # Priority Check: Is this URL the Official Handle or Domain of a Primary Entity?
    # ---------------------------------------------------------------------
    if primary_entities:
        for pe in primary_entities:
            # Check official domains (exact host/subdomain, or host+path for social channels)
            for d in (pe.get('official_domains') or []):
                if not d:
                    continue
                d_low = d.lower().strip()
                if '/' in d_low:
                    d_host, d_path = d_low.split('/', 1)
                    d_path_clean = d_path.strip('/')
                    norm_path = path.strip('/')
                    if _host_matches_domain(hostname, d_host) and (
                        norm_path == d_path_clean or norm_path.startswith(d_path_clean + '/')
                    ):
                        return 0.99
                else:
                    if _host_matches_domain(hostname, d_low):
                        return 0.99

            # Check official social media handles ONLY on verified social hosts (exact 1st path segment)
            if is_social_host and first_seg:
                for h in (pe.get('official_handles') or []):
                    h_clean = h.lstrip('@').lower().strip()
                    if h_clean and first_seg == h_clean:
                        return 0.99

    # ---------------------------------------------------------------------
    # Tier 1: Primary Official Authority (Government, Central Banks, UN, Academic Journals)
    # ---------------------------------------------------------------------
    is_core_gov = (
        hostname.endswith('.gov') or
        hostname.endswith('.gov.in') or
        hostname.endswith('.mil') or
        hostname.endswith('.europa.eu') or
        hostname.endswith('.int') or             # Intergovernmental (who.int, wmo.int)
        any(_host_matches_domain(hostname, d) for d in (
            'rbi.org.in', 'npci.org.in', 'un.org', 'isro.gov.in', 'nasa.gov',
            'gstcouncil.gov.in', 'cbic.gov.in', 'bcci.tv'
        ))
    )
    if is_core_gov:
        return 0.99

    # National Informatics Centre (.nic.in):
    # Only assign 0.99 Tier 1 if the claim relates to government, district administration, or education.
    # For cinema/entertainment or celebrity/party speech claims, unrelated exam/agency portals
    # (e.g. cuet.nta.nic.in) are general web results (0.60), NOT primary origins.
    if hostname.endswith('.nic.in'):
        if primary_entities and any(pe.get('category') in ('cinema_entertainment', 'tvk_party') for pe in primary_entities):
            return 0.60
        return 0.99

    # Peer-reviewed scientific & medical research repositories
    if any(_host_matches_domain(hostname, j) for j in ACADEMIC_JOURNAL_DOMAINS) or _host_matches_domain(hostname, 'doi.org'):
        return 0.99

    # Known high-priority verified handles (strictly on X/Twitter host + first segment)
    if _host_matches_domain(hostname, 'tvk.family') or (
        any(_host_matches_domain(hostname, sh) for sh in ('x.com', 'twitter.com'))
        and first_seg in ('tvkvijayhq', 'dreamwarriorpic', 'prabhu_sr', 'suriya_offl')
    ):
        return 0.99

    # ---------------------------------------------------------------------
    # Tier 2: Certified Fact-Checkers (IFCN Network Worldwide)
    # Only dedicated fact-checking domains or dedicated factcheck subdomains qualify;
    # arbitrary websites with '/fact-check' in URL path stay mainstream/general tier.
    # ---------------------------------------------------------------------
    dedicated_fc = [
        'pib.gov.in', 'youturn.in', 'boomlive.in', 'altnews.in', 'factly.in',
        'vishvasnews.com', 'factcrescendo.com', 'thequint.com',
        'snopes.com', 'politifact.com', 'factcheck.org', 'fullfact.org',
        'checkyourfact.com', 'leadstories.com', 'newsmobile.in',
        'reuters.com', 'apnews.com'
    ]
    if any(_host_matches_domain(hostname, fc) for fc in dedicated_fc) or 'factcheck' in hostname:
        return 0.95

    # ---------------------------------------------------------------------
    # Tier 3: Primary News Wire Agencies
    # ---------------------------------------------------------------------
    if any(_host_matches_domain(hostname, wire) for wire in NEWS_WIRE_DOMAINS):
        return 0.92

    # ---------------------------------------------------------------------
    # Tier 4: Accredited Regional & National Mainstream Press
    # ---------------------------------------------------------------------
    accredited_press = [
        'thehindu.com', 'hindutamil.in', 'puthiyathalaimurai.com', 'thanthitv.com',
        'polimernews.com', 'dinamalar.com', 'dinakaran.com', 'vikatan.com',
        'dtnext.in', 'economictimes.indiatimes.com', 'livemint.com',
        'moneycontrol.com', 'business-standard.com', 'indianexpress.com',
        'ndtv.com', 'bbc.com', 'bbc.co.uk', 'nytimes.com', 'washingtonpost.com',
        'theguardian.com', 'space.com', 'scmp.com', 'variety.com', 'hollywoodreporter.com',
        'espncricinfo.com', 'cricbuzz.com', 'usatoday.com', 'cnn.com', 'npr.org',
        'cbsnews.com', 'nbcnews.com', 'abcnews.go.com', 'forbes.com',
        'timesofindia.indiatimes.com', 'hindustantimes.com', 'indiatoday.in',
        'scroll.in', 'thewire.in', 'aljazeera.com', 'dw.com'
    ]
    if any(_host_matches_domain(hostname, media) for media in accredited_press):
        return 0.90

    # ---------------------------------------------------------------------
    # Disallowed: Open-edit / Crowd-sourced Wikis (Wikipedia, Wikimedia, Fandom)
    # ---------------------------------------------------------------------
    if any(_host_matches_domain(hostname, wiki) for wiki in ['wikipedia.org', 'wikimedia.org', 'fandom.com', 'wikia.org', 'wikibooks.org']):
        return 0.0

    # ---------------------------------------------------------------------
    # Tier 5: Peer-Reviewed Editorial Knowledge Bases
    # ---------------------------------------------------------------------
    if _host_matches_domain(hostname, 'britannica.com'):
        return 0.85

    # ---------------------------------------------------------------------
    # Tier 6: Unverified Social Media & Forums (Strict Low Credibility)
    # Generic posts on X, Reddit, Facebook, or Instagram that are NOT from verified primary accounts
    # ---------------------------------------------------------------------
    social_media_domains = [
        'x.com', 'twitter.com', 'instagram.com', 'reddit.com', 'facebook.com',
        'threads.net', 'tiktok.com', 'youtube.com', 't.me', 'telegram.org'
    ]
    if any(_host_matches_domain(hostname, smd) for smd in social_media_domains):
        return 0.45

    # Standard general web search result / blog
    return 0.60


def detect_authorities(
    text: Optional[str],
    entities: Optional[List[str]] = None,
    ocr_data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Dynamic authority and multi-source primary origin detection for any claim across ALL sectors.
    Returns matched primary origin entities, categories, roles, and official channels.
    """
    if not text and not (ocr_data and ocr_data.get('extracted_text')):
        return []

    combined_text = (text or "") + " "
    if ocr_data and ocr_data.get('extracted_text'):
        combined_text += ocr_data['extracted_text'] + " "
    if entities:
        combined_text += " ".join(entities)
    lower_t = combined_text.lower()
    query_hint = (text or combined_text).strip()[:35]

    matched: List[Dict[str, Any]] = []
    matched_keys: Set[str] = set()

    # 1. Authority Registry checks (Maintains backward compatibility for sector test keys)
    for reg_key, reg_info in AUTHORITY_REGISTRY.items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', lower_t) for kw in reg_info.get('keywords', [])):
            primary_doms = list(reg_info.get('primary_domains', []))
            off_doms = list(reg_info.get('official_domains', primary_doms))
            reg_queries = [f'"{reg_info["name"]}" official statement OR circular']
            if primary_doms:
                reg_queries.append(f'site:{primary_doms[0]} {query_hint}')
            matched.append({
                'key': reg_key,
                'name': reg_info['name'],
                'role': reg_info.get('role', 'Primary Authority'),
                'category': reg_info.get('category', reg_key),
                'primary_domains': primary_doms,
                'official_domains': off_doms,
                'official_handles': list(reg_info.get('official_handles', [])),
                'trusted_media': list(reg_info.get('trusted_media', [])),
                'factcheck_domains': list(reg_info.get('factcheck_domains', [])),
                'associated_figures': [],
                'search_queries': reg_queries
            })
            matched_keys.add(reg_key)

    # 2. Universal Dynamic Origin Resolver (Multi-sector primary origin entities)
    universal_origins = UniversalOriginResolver.resolve_primary_origins(text, entities=entities, ocr_data=ocr_data)
    for uo in universal_origins:
        # Guarantee primary_domains key is present
        if 'primary_domains' not in uo:
            p_doms: List[str] = list(uo.get('official_domains') or [])
            for h in (uo.get('official_handles') or []):
                clean_h = h.lstrip('@')
                p_doms.append(f"x.com/{clean_h}")
            uo['primary_domains'] = p_doms

        # If not already matched by authority registry, append
        uo_key = uo.get('key')
        uo_name = str(uo.get('name', '')).lower()
        if uo_key not in matched_keys and not any(str(m.get('name', '')).lower() == uo_name for m in matched):
            matched.append(uo)
            if uo_key:
                matched_keys.add(str(uo_key))

    return matched


def generate_authority_queries(
    claim_text: str,
    matched_authorities: Optional[List[Dict[str, Any]]],
    max_queries: int = 4
) -> List[str]:
    """Universal query generator utilizing matched authorities and the UniversalOriginResolver."""
    if not claim_text or not claim_text.strip():
        return []

    queries: List[str] = []
    clean_claim = re.sub(r'[^\w\s\-\u0B80-\u0BFF]', ' ', claim_text).strip()
    words = clean_claim.split()
    core_snippet = UniversalOriginResolver._extract_topic_terms(claim_text, limit=6) or " ".join(words[:8])

    if matched_authorities:
        for auth in matched_authorities:
            # Query official/primary domains
            domains = auth['primary_domains'] if 'primary_domains' in auth else auth.get('official_domains', [])
            for d in (domains or []):
                if not d:
                    continue
                if 'x.com/' in d or 'twitter.com/' in d:
                    queries.append(f'{core_snippet} site:{d}')
                elif not d.startswith('@'):
                    queries.append(f'{core_snippet} site:{d}')
            # Query official handles
            for h in (auth.get('official_handles') or []):
                if not h:
                    continue
                clean_h = h.lstrip('@')
                queries.append(f'site:x.com/{clean_h} {core_snippet}')
            # Query trusted media if available
            for tm in (auth.get('trusted_media') or [])[:1]:
                if tm:
                    queries.append(f'{core_snippet} site:{tm}')

    # Also pull from universal resolver
    u_queries = UniversalOriginResolver.generate_universal_queries(claim_text, max_queries=max_queries)
    for uq in u_queries:
        if uq not in queries:
            queries.append(uq)

    return list(dict.fromkeys(queries))[:max(max_queries, 4)]
