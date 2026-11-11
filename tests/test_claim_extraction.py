# tests/test_claim_extraction.py
# Unit tests for the Claim Extraction module.

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.claim_extraction import ClaimExtractor, Claim, generate_queries


class TestClaimDataclass:
    """Tests for the Claim dataclass."""

    def test_claim_defaults(self):
        """Claim should initialize with sane defaults."""
        claim = Claim()
        assert claim.entities == []
        assert claim.event == ""
        assert claim.temporal == {}
        assert claim.domain == "general"
        assert claim.original_text == ""

    def test_claim_with_values(self):
        """Claim should store provided values."""
        claim = Claim(
            entities=["Bill Gates", "Microsoft"],
            event="founded",
            temporal={"mentioned_dates": [datetime(1975, 4, 4)]},
            domain="business",
            original_text="Bill Gates founded Microsoft in 1975."
        )
        assert "Bill Gates" in claim.entities
        assert claim.domain == "business"


class TestClaimExtractor:
    """Tests for the ClaimExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a ClaimExtractor with the lightweight spaCy model."""
        return ClaimExtractor(ner_model="en_core_web_sm")

    def test_extract_entities(self, extractor):
        """Should extract named entities from text."""
        text = "Bill Gates founded Microsoft in Seattle."
        entities = extractor.extract_entities(text)

        assert isinstance(entities, list)
        # Should find at least some entities
        entity_texts = [e['text'] for e in entities]
        # Bill Gates and Microsoft should be recognized
        assert any('Gates' in t or 'Bill' in t for t in entity_texts)

    def test_extract_entities_empty(self, extractor):
        """Empty text should return empty entities list."""
        entities = extractor.extract_entities("")
        assert entities == []

    def test_extract_temporal_info_iso_date(self, extractor):
        """Should extract ISO format dates."""
        text = "The event happened on 2024-01-15."
        temporal = extractor.extract_temporal_info(text)

        assert 'mentioned_dates' in temporal
        assert 'relative_times' in temporal
        assert len(temporal['mentioned_dates']) > 0

    def test_extract_temporal_info_written_date(self, extractor):
        """Should extract written-out dates."""
        text = "He announced it on January 15, 2024."
        temporal = extractor.extract_temporal_info(text)
        assert len(temporal['mentioned_dates']) > 0

    def test_extract_temporal_info_no_dates(self, extractor):
        """Text without dates should return empty lists."""
        text = "The sky is blue."
        temporal = extractor.extract_temporal_info(text)
        assert temporal['mentioned_dates'] == []
        assert temporal['relative_times'] == []

    def test_extract_main_event(self, extractor):
        """Should extract the main verb/event."""
        text = "Bill Gates announced a new vaccine."
        event = extractor.extract_main_event(text)

        assert isinstance(event, str)
        assert len(event) > 0

    def test_extract_main_event_fallback(self, extractor):
        """Very short/edge-case text should return fallback."""
        event = extractor.extract_main_event("Hello")
        assert isinstance(event, str)

    def test_detect_domain_politics(self, extractor):
        """Should detect political domain."""
        text = "The president signed a new election bill in congress."
        domain = extractor.detect_domain(text)
        assert domain == "politics"

    def test_detect_domain_health(self, extractor):
        """Should detect health domain."""
        text = "A new vaccine for the disease was approved by doctors."
        domain = extractor.detect_domain(text)
        assert domain == "health"

    def test_detect_domain_science(self, extractor):
        """Should detect science domain."""
        text = "A new research study by scientists found evidence."
        domain = extractor.detect_domain(text)
        assert domain == "science"

    def test_detect_domain_business(self, extractor):
        """Should detect business domain."""
        text = "The company reported record profit in the stock market."
        domain = extractor.detect_domain(text)
        assert domain == "business"

    def test_detect_domain_general(self, extractor):
        """Unmatched text should return 'general'."""
        text = "The cat sat on the mat."
        domain = extractor.detect_domain(text)
        assert domain == "general"

    def test_detect_domain_crypto_finance(self, extractor):
        """Crypto/finance claims should be classified as business, not health."""
        text = "TechGiantX to Give Free 1 Lakh Crypto to Every User Who Logs In Tomorrow"
        domain = extractor.detect_domain(text)
        assert domain == "business", f"Expected 'business' but got '{domain}'"
        assert domain != "health", "Crypto claim must not be classified as health"

    def test_detect_domain_pronoun_who_not_health(self, extractor):
        """Pronoun 'who' in non-health text should not trigger health domain."""
        text = "The person who won the lottery collected 50 lakh tokens"
        domain = extractor.detect_domain(text)
        assert domain != "health", f"Pronoun 'who' incorrectly triggered health domain"

    def test_extract_claim_full_pipeline(self, extractor):
        """Full extraction should produce a valid Claim."""
        text = "Bill Gates announced a new vaccine on January 15, 2024."
        claim = extractor.extract_claim(text)

        assert isinstance(claim, Claim)
        assert claim.original_text == text
        assert isinstance(claim.entities, list)
        assert isinstance(claim.event, str)
        assert isinstance(claim.temporal, dict)
        assert claim.domain in ['health', 'science', 'business', 'politics', 'general']


class TestGenerateQueries:
    """Tests for the generate_queries function."""

    def test_basic_query_generation(self):
        """Should generate queries from a structured claim."""
        claim = Claim(
            entities=["Bill Gates", "WHO"],
            event="announced vaccine",
            temporal={"mentioned_dates": [datetime(2024, 1, 15)]},
            domain="health",
            original_text="Bill Gates announced a new vaccine with WHO."
        )

        queries = generate_queries(claim)

        assert isinstance(queries, list)
        assert len(queries) > 0
        # Should include entity-event combinations
        assert any("Bill Gates" in q for q in queries)
        # Should include temporal query
        assert any("2024-01-15" in q for q in queries)

    def test_no_duplicates(self):
        """Should remove duplicate queries."""
        claim = Claim(
            entities=["Test"],
            event="Test",
            temporal={"mentioned_dates": []},
            domain="general",
            original_text="Test"
        )

        queries = generate_queries(claim)
        assert len(queries) == len(set(queries))

    def test_empty_claim(self):
        """Should handle empty claim gracefully."""
        claim = Claim()
        queries = generate_queries(claim)

        assert isinstance(queries, list)
        # Should at least have the domain and fallback queries
        assert len(queries) >= 1
