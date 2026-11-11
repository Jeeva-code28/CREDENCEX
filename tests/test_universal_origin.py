# tests/test_universal_origin.py
# Comprehensive tests for the Universal Dynamic Origin Resolver
# Verifies dynamic source attribution across Space, Medical, Agriculture,
# District Administration, Cinema, Global Affairs, and Algorithmic Credibility.

import pytest
from src.authority_sources import (
    UniversalOriginResolver,
    detect_authorities,
    get_domain_authority_score
)


class TestUniversalCategoryInference:
    """Tests for dynamic category inference across diverse domains."""

    def test_space_category(self):
        claim = "NASA James Webb telescope spotted potential signs of water on exoplanet"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'space_astronomy'

    def test_medical_category(self):
        claim = "WHO issued new global health advisory regarding antibiotic resistant bacteria"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'health_medical'

    def test_agriculture_category(self):
        claim = "Ministry of Agriculture announced 15 percent increase in MSP for paddy and wheat"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'agriculture_farming'

    def test_district_category(self):
        claim = "District Collector declared holiday for schools due to heavy rain in Coimbatore"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'district_local_gov'

    def test_cinema_category(self):
        claim = "Director Shankar announced official teaser release date for new big budget movie"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'cinema_entertainment'

    def test_global_affairs_category(self):
        claim = "United Nations Security Council passed unanimous resolution on humanitarian ceasefire"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'global_affairs'


class TestDynamicOriginDiscovery:
    """Tests for claimant, district, and primary domain query discovery."""

    def test_discover_district_collector_nic_in(self):
        claim = "Madurai District Collector announced school holiday due to heavy rain"
        claimant, district = UniversalOriginResolver.extract_claimant_and_district(claim)
        assert district == 'madurai'
        queries = UniversalOriginResolver.generate_universal_queries(claim)
        assert any("site:madurai.nic.in" in q for q in queries)

    def test_discover_space_primary_origin(self):
        claim = "ISRO announced launch date for Chandrayaan next phase"
        queries = UniversalOriginResolver.generate_universal_queries(claim)
        combined = " ".join(queries)
        assert "site:isro.gov.in" in combined or "site:nasa.gov" in combined

    def test_discover_medical_primary_origin(self):
        claim = "WHO confirmed clinical trial results published in Lancet"
        queries = UniversalOriginResolver.generate_universal_queries(claim)
        combined = " ".join(queries)
        assert "site:who.int" in combined or "site:thelancet.com" in combined

    def test_discover_agriculture_primary_origin(self):
        claim = "ICAR announced new high yield drought resistant rice seed"
        queries = UniversalOriginResolver.generate_universal_queries(claim)
        combined = " ".join(queries)
        assert "site:icar.org.in" in combined or "site:agricoop.nic.in" in combined

    def test_discover_cinema_origin(self):
        claim = "Lyca Productions announced trailer launch date for upcoming action movie"
        claimant, _ = UniversalOriginResolver.extract_claimant_and_district(claim)
        queries = UniversalOriginResolver.generate_universal_queries(claim)
        assert len(queries) > 0


class TestContextualDisambiguation:
    """
    Comprehensive tests for the contextual disambiguation engine.
    Verifies that homonymous/polysemous keywords are correctly handled
    using case analysis, domain collocates, and exclusion context.
    """

    # ---- WHO disambiguation (pronoun vs. organization) ----

    def test_pronoun_who_lowercase_rejected(self):
        """Lowercase 'who' in non-health context must be rejected as a false match."""
        from src.authority_sources import _disambiguate_keyword
        # "who" as relative pronoun — no uppercase WHO, no health collocates
        assert _disambiguate_keyword('who', 'Anyone who logs in tomorrow gets free tokens') is False

    def test_pronoun_who_titlecase_rejected(self):
        """Title-case 'Who' (headline style) must also be rejected without health context."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('who', 'The Person Who Won the Lottery Gets 50 Lakh') is False

    def test_uppercase_WHO_accepted(self):
        """Uppercase 'WHO' should be accepted as World Health Organization."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('who', 'WHO issued new pandemic advisory') is True

    def test_lowercase_who_with_health_collocates_accepted(self):
        """Even lowercase 'who' is accepted if health domain collocates are present."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('who', 'Doctors say who issued vaccine guidelines for pandemic') is True

    def test_who_with_exclusion_words_rejected(self):
        """'who' with crypto/login exclusion words must be rejected even with ambiguous context."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('who', 'Give Free Crypto to Every User Who Logs In Tomorrow') is False

    def test_uppercase_WHO_overrides_exclusion(self):
        """Uppercase 'WHO' overrides exclusion words (could be WHO commenting on crypto scams)."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('who', 'WHO warns about crypto token health scams') is True

    # ---- Full pipeline integration for WHO ----

    def test_crypto_headline_not_health_category(self):
        """Crypto headline with pronoun 'who' must NOT be classified as health_medical."""
        claim = "TechGiantX to Give Free 1 Lakh Crypto to Every User Who Logs In Tomorrow"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat != 'health_medical', f"Pronoun 'who' incorrectly triggered health_medical (got '{cat}')"

    def test_crypto_headline_classified_as_finance(self):
        """Crypto-related claim should be classified as finance_banking."""
        claim = "New crypto exchange platform offers blockchain wallet with free Bitcoin airdrop"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'finance_banking', f"Expected 'finance_banking' but got '{cat}'"

    def test_pronoun_who_not_resolved_as_WHO_entity(self):
        """Pronoun 'who' must NOT resolve to World Health Organization as a primary origin."""
        claim = "Anyone who logs in tomorrow gets free tokens"
        origins = UniversalOriginResolver.resolve_primary_origins(claim)
        origin_names = [o['name'].lower() for o in origins]
        assert not any('world health' in n for n in origin_names), (
            f"Pronoun 'who' incorrectly resolved to WHO entity: {origin_names}"
        )

    def test_genuine_who_full_name_still_detected(self):
        """Genuine 'World Health Organization' text should still classify as health."""
        claim = "World Health Organization issued new global health advisory regarding antibiotic resistant bacteria"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'health_medical', f"Genuine WHO claim should be health_medical, got '{cat}'"

    def test_genuine_who_entity_still_resolved(self):
        """Genuine 'World Health Organization' text should still resolve as primary origin."""
        claim = "World Health Organization confirmed clinical trial results published in Lancet"
        origins = UniversalOriginResolver.resolve_primary_origins(claim)
        origin_names = [o['name'].lower() for o in origins]
        assert any('world health' in n for n in origin_names), (
            f"Genuine WHO mention should resolve to WHO entity, got: {origin_names}"
        )

    def test_uppercase_WHO_in_headline_detected(self):
        """Uppercase 'WHO' in a headline should correctly classify as health."""
        claim = "WHO confirms new variant spreading across Southeast Asia"
        cat = UniversalOriginResolver.infer_category(claim)
        assert cat == 'health_medical', f"Uppercase WHO should trigger health_medical, got '{cat}'"

    # ---- Mars disambiguation (planet vs. chocolate vs. verb) ----

    def test_mars_planet_accepted_with_space_context(self):
        """'Mars' with space/planet context should be accepted."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('mars', 'NASA rover successfully lands on Mars crater') is True

    def test_mars_chocolate_rejected(self):
        """'mars' in chocolate context should be rejected."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('mars', 'Mars chocolate bar wrapper candy snickers') is False

    # ---- Bill disambiguation (legislation vs. person name) ----

    def test_bill_legislation_accepted(self):
        """'bill' with parliamentary context should be accepted."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('bill', 'The new bill was passed in parliament by vote') is True

    def test_bill_person_name_rejected(self):
        """'bill' as a person name without governance context should be rejected."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('bill', 'Bill went to the store to buy groceries') is False

    # ---- Non-ambiguous terms pass through ----

    def test_unambiguous_keyword_always_accepted(self):
        """Keywords not in AMBIGUOUS_TERMS should always pass through."""
        from src.authority_sources import _disambiguate_keyword
        assert _disambiguate_keyword('vaccine', 'Any random text about vaccine') is True
        assert _disambiguate_keyword('isro', 'ISRO launches satellite') is True
        assert _disambiguate_keyword('bitcoin', 'Buy bitcoin now') is True


class TestAlgorithmicCredibilityScorer:
    """Tests for algorithmic domain hierarchy without hardcoding."""

    def test_tier1_government_and_districts(self):
        # All government portals worldwide
        assert get_domain_authority_score("https://coimbatore.nic.in/document/holiday-notice") == 0.99
        assert get_domain_authority_score("https://madurai.nic.in/press-release") == 0.99
        assert get_domain_authority_score("https://agricoop.nic.in/circulars") == 0.99
        assert get_domain_authority_score("https://www.isro.gov.in/chandrayaan.html") == 0.99
        assert get_domain_authority_score("https://www.nasa.gov/news/press-release-2024") == 0.99

    def test_tier1_intergovernmental(self):
        assert get_domain_authority_score("https://www.who.int/news/item/24-05-2024") == 0.99
        assert get_domain_authority_score("https://www.un.org/press/en/2024/sc15589.doc.htm") == 0.99

    def test_tier1_peer_reviewed_academic_journals(self):
        assert get_domain_authority_score("https://pubmed.ncbi.nlm.nih.gov/38291039/") == 0.99
        assert get_domain_authority_score("https://www.nature.com/articles/s41586-024-07589-2") == 0.99
        assert get_domain_authority_score("https://www.thelancet.com/journals/lancet/article/PIIS0140") == 0.99
        assert get_domain_authority_score("https://arxiv.org/abs/2401.12345") == 0.99

    def test_tier2_factcheck_networks(self):
        assert get_domain_authority_score("https://pib.gov.in/factcheck") == 0.99  # Gov + Factcheck
        assert get_domain_authority_score("https://youturn.in/factcheck/viral-claim") == 0.95
        assert get_domain_authority_score("https://www.altnews.in/viral-news/") == 0.95
        assert get_domain_authority_score("https://fullfact.org/health/claim-check/") == 0.95

    def test_tier3_primary_wire_services(self):
        assert get_domain_authority_score("https://www.ptinews.com/story/national/news123") == 0.92
        assert get_domain_authority_score("https://www.aninews.in/news/national/general-news") == 0.92
        assert get_domain_authority_score("https://www.bloomberg.com/news/articles/2024") == 0.92

    def test_tier4_accredited_news(self):
        assert get_domain_authority_score("https://www.thehindu.com/sci-tech/science/") == 0.90
        assert get_domain_authority_score("https://www.space.com/astronomy-news") == 0.90

    def test_tier5_unverified_blogs(self):
        assert get_domain_authority_score("https://cinema-gossip-corner.blogspot.com/post1") == 0.60
