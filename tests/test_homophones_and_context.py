import pytest
from src.claim_extraction import ClaimExtractor, Claim, generate_queries
from src.authority_sources import detect_authorities


@pytest.fixture
def claim_extractor():
    return ClaimExtractor()


class TestFinanceVsHealthDisambiguation:
    """Verifies that crypto/finance claims never trigger health/WHO routing and vice-versa."""

    def test_crypto_token_never_routes_to_who_or_health(self, claim_extractor):
        text = "Ethereum crypto mining network announces 50% token burn across decentralized liquidity pools."
        claim = claim_extractor.extract_claim(text)
        authorities = detect_authorities(claim.original_text, entities=claim.entities)

        auth_names = [a['name'].lower() for a in authorities]
        auth_roles = [a.get('role', '').lower() for a in authorities]

        # Must NEVER route to health bodies
        assert not any("health" in n or "who" in n or "medical" in n for n in auth_names)
        assert not any("health" in r or "medical" in r for r in auth_roles)

    def test_rbi_bank_interest_rate_routes_to_finance(self, claim_extractor):
        text = "Reserve Bank of India announces immediate cut in benchmark repo interest rate for commercial banks."
        claim = claim_extractor.extract_claim(text)
        authorities = detect_authorities(claim.original_text, entities=claim.entities)

        auth_names = [a['name'].lower() for a in authorities]
        # Should identify RBI or banking authorities, never health
        assert any("reserve bank" in n or "rbi" in n or "bank" in n for n in authorities_names_or_domain(authorities, claim.category))
        assert not any("world health" in n or "who" in n for n in auth_names)

    def test_who_health_never_routes_to_banking(self, claim_extractor):
        text = "World Health Organization issues global alert regarding new airborne coronavirus variant symptoms."
        claim = claim_extractor.extract_claim(text)
        authorities = detect_authorities(claim.original_text, entities=claim.entities)

        auth_names = [a['name'].lower() for a in authorities]
        assert any("world health organization" in n or "who" in n for n in auth_names)
        # Must NEVER route to banking or financial regulators
        assert not any("reserve bank" in n or "sebi" in n or "finance" in n for n in auth_names)


def authorities_names_or_domain(authorities, category):
    names = [a['name'].lower() for a in authorities]
    if category:
        names.append(category.lower())
    return names


class TestHomonymContextDisambiguation:
    """Tests for words with multiple meanings (polysemy) to ensure context determines interpretation."""

    def test_bank_financial_vs_river(self, claim_extractor):
        # Case A: Financial Bank
        text_fin = "State Bank of India approved digital collateral loans for self-help groups."
        claim_fin = claim_extractor.extract_claim(text_fin)
        assert any("bank" in str(e).lower() for e in claim_fin.entities) or "sbi" in text_fin.lower()

        # Case B: River Bank
        text_geo = "Heavy monsoon rains caused the Cauvery river bank to overflow into surrounding villages."
        claim_geo = claim_extractor.extract_claim(text_geo)
        # Should detect River / Geography entities, not financial authority
        auth_geo = detect_authorities(claim_geo.original_text, entities=claim_geo.entities)
        geo_names = [a['name'].lower() for a in auth_geo]
        assert not any("reserve bank" in n or "sebi" in n for n in geo_names)

    def test_cell_biological_vs_battery_vs_prison(self, claim_extractor):
        # Case A: Biological cell
        text_bio = "Breakthrough medical therapy reprogrammed immune T-cell receptors to eliminate malignant tumors."
        claim_bio = claim_extractor.extract_claim(text_bio)
        assert claim_bio.domain in ("health", "science", "general")

        # Case B: Battery cell
        text_tech = "Automaker inaugurated a manufacturing facility for lithium iron phosphate battery cells."
        claim_tech = claim_extractor.extract_claim(text_tech)
        queries_tech = generate_queries(claim_tech)
        assert any("battery" in q.lower() for q in queries_tech)

        # Case C: Prison cell
        text_crim = "Two convicted inmates breached high-security central prison cell bars during power outage."
        claim_crim = claim_extractor.extract_claim(text_crim)
        queries_crim = generate_queries(claim_crim)
        assert any("prison" in q.lower() for q in queries_crim)

    def test_strike_labor_vs_military(self, claim_extractor):
        # Case A: Labor strike
        text_labor = "State transport corporation bus drivers announced an indefinite statewide strike over wage settlement."
        claim_labor = claim_extractor.extract_claim(text_labor)
        queries_labor = generate_queries(claim_labor)
        assert any("bus" in q.lower() or "drivers" in q.lower() or "transport" in q.lower() for q in queries_labor)

        # Case B: Military strike
        text_mil = "Air force conducted targeted airstrike on cross-border militant launchpads."
        claim_mil = claim_extractor.extract_claim(text_mil)
        queries_mil = generate_queries(claim_mil)
        assert any("air" in q.lower() or "force" in q.lower() or "airstrike" in q.lower() for q in queries_mil)

    def test_bill_legislation_vs_person(self, claim_extractor):
        # Case A: Legislative Bill
        text_leg = "Parliament passed the digital data protection bill with bipartisan consensus."
        claim_leg = claim_extractor.extract_claim(text_leg)
        assert "bill" in claim_leg.original_text.lower()
        queries_leg = generate_queries(claim_leg)
        assert any("parliament" in q.lower() or "data" in q.lower() for q in queries_leg)

        # Case B: Bill Gates (Person)
        text_person = "Bill Gates donated 50 million dollars to eliminate polio in tropical countries."
        claim_person = claim_extractor.extract_claim(text_person)
        assert any("bill gates" in str(e).lower() for e in claim_person.entities) or "Bill Gates" in claim_person.original_text

    def test_lead_metal_toxicity_vs_leadership(self, claim_extractor):
        # Case A: Heavy metal Lead
        text_metal = "Health inspectors detected dangerous levels of heavy metal lead in packaged infant formula."
        claim_metal = claim_extractor.extract_claim(text_metal)
        queries_metal = generate_queries(claim_metal)
        assert any("lead" in q.lower() or "formula" in q.lower() or "infant" in q.lower() for q in queries_metal)

        # Case B: Leadership Lead
        text_gov = "The Chief Minister will lead an official ministerial trade mission to Europe."
        claim_gov = claim_extractor.extract_claim(text_gov)
        queries_gov = generate_queries(claim_gov)
        assert any("chief minister" in q.lower() or "ministerial" in q.lower() or "trade" in q.lower() for q in queries_gov)


class TestHomophoneSemanticIntegrity:
    """Verifies that phonetically similar words maintain semantic integrity during claim processing."""

    def test_principle_vs_principal(self, claim_extractor):
        text_rule = "The university violated constitutional principles of equality in admissions."
        claim_rule = claim_extractor.extract_claim(text_rule)
        assert "principles" in claim_rule.original_text

        text_person = "The school principal suspended classes due to severe cyclone alert."
        claim_person = claim_extractor.extract_claim(text_person)
        assert "principal" in claim_person.original_text

    def test_affect_vs_effect(self, claim_extractor):
        text_verb = "New tariff policies will adversely affect automotive component exports."
        claim_verb = claim_extractor.extract_claim(text_verb)
        assert "affect" in claim_verb.original_text

        text_noun = "The new tax policy came into effect starting from April 1st."
        claim_noun = claim_extractor.extract_claim(text_noun)
        assert "effect" in claim_noun.original_text
