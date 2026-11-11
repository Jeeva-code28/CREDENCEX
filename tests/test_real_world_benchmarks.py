# tests/test_real_world_benchmarks.py
# End-to-end benchmark tests across core real-world misinformation categories.

import pytest
from src.claim_extraction import ClaimExtractor, generate_queries
from src.authority_sources import detect_authorities
from src.fusion_engine import FusionEngine


@pytest.fixture
def claim_extractor():
    return ClaimExtractor()


@pytest.fixture
def fusion_engine():
    return FusionEngine()


class TestGovernmentWelfareBenchmarks:
    """Benchmark tests on viral government administration rumors."""

    def test_ration_card_cancellation_benchmark(self, claim_extractor, fusion_engine):
        claim_text = "All Tamil Nadu ration cards will be cancelled from next month."
        claim = claim_extractor.extract_claim(claim_text)

        # 1. Claim extraction check
        assert any("tamil nadu" in str(e).lower() for e in claim.entities)
        queries = generate_queries(claim)
        assert queries[0] == claim_text  # Strict claim priority

        # 2. Authority routing
        authorities = detect_authorities(claim.original_text, entities=claim.entities)
        auth_names = [a['name'].lower() for a in authorities]
        assert any("tamil nadu" in n for n in auth_names)

        # 3. Fusion & Truth Analysis
        refuting_evidence = [
            {
                'source': 'tndiprnews',
                'source_tier': 'news_wire',
                'relationship': 'contradiction',
                'text': 'Tamil Nadu Food Department denied reports of ration card cancellation.'
            }
        ]
        nli_mock = {
            'claim': claim_text,
            'aggregated': {
                'primary_verdict': 'contradiction',
                'confidence': 0.91,
                'weighted_probabilities': {'contradiction': 0.91, 'entailment': 0.05, 'neutral': 0.04}
            }
        }
        res = fusion_engine.fuse_verdicts(
            nli_mock,
            primary_origins=authorities,
            evidence=refuting_evidence,
            claim=claim_text
        )

        assert res['final_verdict'] == 'Contradicted'
        assert res['truth_analysis']['status'] == 'rumor'
        assert res['truth_analysis']['badge_label'] == 'DEBUNKED RUMOR / FALSE'
        assert "Food Department denied" in res['truth_analysis']['actual_truth']

    def test_college_student_subsidy_benchmark(self, claim_extractor):
        claim_text = "Tamil Nadu Government will give 50000 to every college student without any application."
        claim = claim_extractor.extract_claim(claim_text)

        queries = generate_queries(claim)
        assert queries[0] == claim_text
        assert any("50000" in q or "college" in q or "student" in q for q in queries)


class TestHealthAndMedicalBenchmarks:
    """Benchmark tests on public health viral hoaxes and pandemic claims."""

    def test_who_digital_passport_benchmark(self, claim_extractor, fusion_engine):
        claim_text = "WHO announces mandatory digital health passports worldwide in 2026."
        claim = claim_extractor.extract_claim(claim_text)

        authorities = detect_authorities(claim.original_text, entities=claim.entities)
        auth_names = [a['name'].lower() for a in authorities]
        assert any("who" in n or "world health" in n for n in auth_names)

        refuting_evidence = [
            {
                'source': 'reuters',
                'source_tier': 'fact_checker',
                'relationship': 'contradiction',
                'text': 'Fact Check: The WHO has no authority to mandate digital health passports for global citizens.'
            }
        ]
        nli_mock = {
            'claim': claim_text,
            'aggregated': {
                'primary_verdict': 'contradiction',
                'confidence': 0.88,
                'weighted_probabilities': {'contradiction': 0.88, 'entailment': 0.06, 'neutral': 0.06}
            }
        }
        res = fusion_engine.fuse_verdicts(
            nli_mock,
            primary_origins=authorities,
            evidence=refuting_evidence,
            claim=claim_text
        )

        assert res['final_verdict'] == 'Contradicted'
        assert res['truth_analysis']['status'] == 'rumor'
        assert "WHO has no authority" in res['truth_analysis']['actual_truth']

    def test_vaccine_microchip_patent_benchmark(self, claim_extractor):
        claim_text = "Bill Gates patented an implantable microchip sensor numbered 060606 for COVID vaccines."
        claim = claim_extractor.extract_claim(claim_text)

        assert any("bill gates" in str(e).lower() for e in claim.entities) or "Bill Gates" in claim.original_text
        queries = generate_queries(claim)
        assert any("microchip" in q.lower() or "060606" in q for q in queries)


class TestFinanceAndCryptoBenchmarks:
    """Benchmark tests on cryptocurrency scams and financial panic rumors."""

    def test_rbi_500_rupee_note_ban_benchmark(self, claim_extractor, fusion_engine):
        claim_text = "Reserve Bank of India has banned all 500 rupee currency notes starting midnight."
        claim = claim_extractor.extract_claim(claim_text)

        authorities = detect_authorities(claim.original_text, entities=claim.entities)
        auth_names = [a['name'].lower() for a in authorities]
        assert any("rbi" in n or "reserve bank" in n for n in auth_names)

        refuting_evidence = [
            {
                'source': 'pib.gov.in',
                'source_tier': 'news_wire',
                'relationship': 'contradiction',
                'text': 'PIB Fact Check: The Reserve Bank of India has issued no order withdrawing 500 rupee notes.'
            }
        ]
        nli_mock = {
            'claim': claim_text,
            'aggregated': {
                'primary_verdict': 'contradiction',
                'confidence': 0.93,
                'weighted_probabilities': {'contradiction': 0.93, 'entailment': 0.03, 'neutral': 0.04}
            }
        }
        res = fusion_engine.fuse_verdicts(
            nli_mock,
            primary_origins=authorities,
            evidence=refuting_evidence,
            claim=claim_text
        )

        assert res['final_verdict'] == 'Contradicted'
        assert res['truth_analysis']['status'] == 'rumor'
        assert "Reserve Bank of India has issued no order" in res['truth_analysis']['actual_truth']

    def test_crypto_giveaway_scam_benchmark(self, claim_extractor):
        claim_text = "Elon Musk is giving away 10000 Bitcoin live on YouTube to all viewers."
        claim = claim_extractor.extract_claim(claim_text)

        assert any("elon musk" in str(e).lower() for e in claim.entities) or "Elon Musk" in claim.original_text
        queries = generate_queries(claim)
        assert any("bitcoin" in q.lower() or "giving" in q.lower() for q in queries)


class TestGenuineVerifiedNewsBenchmarks:
    """Benchmark tests on genuine, verified factual news events."""

    def test_nasa_james_webb_exoplanet_benchmark(self, claim_extractor, fusion_engine):
        claim_text = "NASA James Webb Space Telescope detected water vapor and carbon dioxide on exoplanet K2-18b."
        claim = claim_extractor.extract_claim(claim_text)

        authorities = detect_authorities(claim.original_text, entities=claim.entities)
        auth_names = [a['name'].lower() for a in authorities]
        assert any("nasa" in n for n in auth_names)

        corroborating_evidence = [
            {
                'source': 'nasa.gov',
                'source_tier': 'news_wire',
                'is_official_primary': True,
                'relationship': 'entailment',
                'text': 'NASA Webb telescope reveals carbon-bearing molecules including methane and CO2 on exoplanet K2-18b.'
            }
        ]
        nli_mock = {
            'claim': claim_text,
            'aggregated': {
                'primary_verdict': 'entailment',
                'confidence': 0.94,
                'weighted_probabilities': {'entailment': 0.94, 'neutral': 0.04, 'contradiction': 0.02}
            }
        }
        res = fusion_engine.fuse_verdicts(
            nli_mock,
            primary_origins=authorities,
            evidence=corroborating_evidence,
            claim=claim_text
        )

        assert res['final_verdict'] == 'Supported'
        assert res['truth_analysis']['status'] == 'verified_truth'
        assert res['truth_analysis']['badge_label'] == 'VERIFIED ACTUAL TRUTH'
        assert "verified factual truth" in res['truth_analysis']['actual_truth']
        assert "NASA Webb telescope reveals" in res['truth_analysis']['actual_truth']
