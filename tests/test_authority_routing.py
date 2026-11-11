# tests/test_authority_routing.py
# Unit tests for official authority detection, domain authority scoring,
# and targeted retrieval query generation.

import pytest
from src.authority_sources import (
    detect_authorities,
    generate_authority_queries,
    get_domain_authority_score,
    AUTHORITY_REGISTRY
)


class TestAuthorityDetection:
    """Tests for detecting official authorities (RBI, TN Government, TVK, etc.)."""

    def test_detect_rbi_financial_authority(self):
        claim = "RBI increases UPI transaction limit to 10 lakhs from tomorrow"
        matched = detect_authorities(claim)
        assert len(matched) > 0
        keys = [m['key'] for m in matched]
        assert 'finance_banking' in keys
        assert any('rbi.org.in' in doms for doms in [m['primary_domains'] for m in matched])

    def test_detect_gst_financial_authority(self):
        claim = "GST Council announced 0% tax on health insurance"
        matched = detect_authorities(claim)
        assert any(m['key'] == 'finance_banking' for m in matched)

    def test_detect_tamil_nadu_government_scheme(self):
        claim = "Tamil Nadu government announced Kalaignar Magalir Urimai scheme ₹2000"
        matched = detect_authorities(claim)
        assert len(matched) > 0
        assert any(m['key'] == 'tamil_nadu_govt' for m in matched)
        tn_auth = next(m for m in matched if m['key'] == 'tamil_nadu_govt')
        assert 'tn.gov.in' in tn_auth['primary_domains']
        assert 'dipr.tn.gov.in' in tn_auth['primary_domains']

    def test_detect_tvk_party(self):
        claim = "TVK Thalapathy Vijay announced party flag and Vikravandi conference"
        matched = detect_authorities(claim)
        assert len(matched) > 0
        assert any(m['key'] == 'tvk_party' for m in matched)
        tvk_auth = next(m for m in matched if m['key'] == 'tvk_party')
        assert any('tvkvijayhq' in d for d in tvk_auth['primary_domains'])

    def test_unrelated_claim_no_false_authority(self):
        claim = "A giant meteor was spotted over the Atlantic Ocean by astronomers"
        matched = detect_authorities(claim)
        assert len(matched) == 0


class TestAuthorityQueries:
    """Tests for generating targeted search queries."""

    def test_generate_rbi_queries(self):
        claim = "RBI ban 500 rupee note"
        matched = detect_authorities(claim)
        queries = generate_authority_queries(claim, matched)
        assert len(queries) > 0
        combined = " ".join(queries)
        assert "site:rbi.org.in" in combined or "rbi.org.in" in combined

    def test_generate_tn_govt_queries(self):
        claim = "Tamil Nadu government free bus travel scheme for students"
        matched = detect_authorities(claim)
        queries = generate_authority_queries(claim, matched)
        assert len(queries) > 0
        combined = " ".join(queries)
        assert "site:tn.gov.in" in combined

    def test_generate_tvk_queries(self):
        claim = "TVK party alliance with major party"
        matched = detect_authorities(claim)
        queries = generate_authority_queries(claim, matched)
        assert len(queries) > 0
        combined = " ".join(queries)
        assert "tvkvijayhq" in combined or "site:puthiyathalaimurai.com" in combined


class TestDomainAuthorityScoring:
    """Tests for multi-tier source credibility weights."""

    def test_tier1_primary_official_government(self):
        # RBI, Central Govt, State Govt (.gov.in, .nic.in, rbi.org.in)
        assert get_domain_authority_score("https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx") == 0.99
        assert get_domain_authority_score("https://www.tn.gov.in/scheme/data_view/68532") == 0.99
        assert get_domain_authority_score("https://pib.gov.in/PressReleasePage.aspx?PRID=198372") == 0.99
        assert get_domain_authority_score("https://www.npci.org.in/what-we-do/upi/product-overview") == 0.99
        assert get_domain_authority_score("https://x.com/tvkvijayhq/status/184857282") == 0.99

    def test_tier2_certified_factcheckers(self):
        assert get_domain_authority_score("https://youturn.in/factcheck/rbi-upi-claim") == 0.95
        assert get_domain_authority_score("https://www.boomlive.in/fact-check/tamil-nadu-scheme") == 0.95
        assert get_domain_authority_score("https://www.altnews.in/viral-tvk-claim/") == 0.95
        assert get_domain_authority_score("https://www.snopes.com/fact-check/some-claim") == 0.95

    def test_tier3_reputable_regional_and_national_media(self):
        assert get_domain_authority_score("https://www.thehindu.com/news/national/tamil-nadu/") == 0.90
        assert get_domain_authority_score("https://puthiyathalaimurai.com/tamilnadu-news") == 0.90
        assert get_domain_authority_score("https://www.thanthitv.com/latest-news") == 0.90
        assert get_domain_authority_score("https://economictimes.indiatimes.com/wealth") == 0.90

    def test_tier4_open_wiki_disallowed_and_editorial_ref(self):
        assert get_domain_authority_score("https://en.wikipedia.org/wiki/Reserve_Bank_of_India") == 0.0
        assert get_domain_authority_score("https://www.britannica.com/topic/Reserve-Bank-of-India") == 0.85

    def test_tier5_generic_unverified_web(self):
        assert get_domain_authority_score("https://random-rumors-blog.wordpress.com/post123") == 0.60
