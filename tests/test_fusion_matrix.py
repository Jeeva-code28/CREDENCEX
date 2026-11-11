# tests/test_fusion_matrix.py
# FusionEngine decision rules, multimodal arbitration, and truth matrix tests.

import pytest
from src.fusion_engine import FusionEngine


@pytest.fixture
def engine():
    return FusionEngine()


def _build_nli_result(entail=0.0, neutral=0.0, contra=0.0, claim_text="Test claim"):
    probs = {'entailment': entail, 'neutral': neutral, 'contradiction': contra}
    primary = max(probs, key=probs.get)
    return {
        'claim': claim_text,
        'individual_results': [],
        'aggregated': {
            'weighted_probabilities': probs,
            'primary_verdict': primary,
            'confidence': probs[primary]
        }
    }


class TestFusionEngineDecisionRules:
    """Tests all 7 hierarchical decision rules of the multimodal FusionEngine."""

    def test_rule_1_accredited_contradiction_overrides(self, engine):
        """Rule 1: Accredited news wire or fact-checker refutes without primary confirmation -> Contradicted."""
        nli_res = _build_nli_result(entail=0.4, neutral=0.1, contra=0.5)
        evidence = [
            {
                'source': 'reuters',
                'source_tier': 'news_wire',
                'relationship': 'contradiction',
                'text': 'Reuters fact check confirms government issued no such cancellation order.'
            }
        ]
        primary_origins = [{'name': 'Government of Tamil Nadu', 'role': 'State Government'}]

        res = engine.fuse_verdicts(nli_res, primary_origins=primary_origins, evidence=evidence)
        assert res['final_verdict'] == 'Contradicted'
        assert res['final_confidence'] >= 0.85
        assert "Contradicted by accredited news reporting" in res['reasoning']

    def test_rule_2_primary_origin_confirmation(self, engine):
        """Rule 2: Primary origin official circular confirms claim -> Supported (0.92+ conf)."""
        nli_res = _build_nli_result(entail=0.6, neutral=0.3, contra=0.1)
        evidence = [
            {
                'source': 'tnsocialwelfare.tn.gov.in',
                'is_official_primary': True,
                'relationship': 'entailment',
                'text': 'Official notification: Government announces scheme implementation guidelines.'
            }
        ]
        primary_origins = [{'name': 'Tamil Nadu Social Welfare Dept', 'role': 'Primary Origin'}]

        res = engine.fuse_verdicts(nli_res, primary_origins=primary_origins, evidence=evidence)
        assert res['final_verdict'] == 'Supported'
        assert res['final_confidence'] >= 0.92
        assert "Corroborated by official announcements" in res['reasoning']

    def test_rule_3_image_forensics_contradiction(self, engine):
        """Rule 3: Strong visual manipulation (conf > 0.70) -> Contradicted."""
        nli_res = _build_nli_result(entail=0.5, neutral=0.4, contra=0.1)
        image_res = {
            'authenticity': {'verdict': 'manipulated', 'authenticity_score': 0.85},
            'context': {'verdict': 'contradiction', 'consistency_score': 0.15}
        }

        res = engine.fuse_verdicts(nli_res, image_result=image_res)
        assert res['final_verdict'] == 'Contradicted'
        assert "Visual forensics detected manipulation" in res['reasoning']

    def test_rule_4_strong_text_entailment(self, engine):
        """Rule 4a: Strong text entailment (> 0.75) without image contradiction -> Supported."""
        nli_res = _build_nli_result(entail=0.88, neutral=0.08, contra=0.04)
        res = engine.fuse_verdicts(nli_res)
        assert res['final_verdict'] == 'Supported'
        assert res['final_confidence'] == 0.88
        assert "Supported by multiple independent authoritative sources" in res['reasoning']

    def test_rule_4_strong_text_contradiction(self, engine):
        """Rule 4b: Strong text contradiction (> 0.75) without image contradiction -> Contradicted."""
        nli_res = _build_nli_result(entail=0.02, neutral=0.08, contra=0.90)
        res = engine.fuse_verdicts(nli_res)
        assert res['final_verdict'] == 'Contradicted'
        assert res['final_confidence'] == 0.90
        assert "Refuted by credible factual evidence" in res['reasoning']

    def test_rule_5_conflicting_text_and_image(self, engine):
        """Rule 5: Text says entailment, but image forensics detects contradiction -> Unverified."""
        nli_res = _build_nli_result(entail=0.82, neutral=0.10, contra=0.08)
        image_res = {
            'authenticity': {'verdict': 'manipulated', 'authenticity_score': 0.40},
            'context': {'verdict': 'contradiction', 'consistency_score': 0.35}
        }

        res = engine.fuse_verdicts(nli_res, image_result=image_res)
        assert res['final_verdict'] == 'Unverified'
        assert "Conflicting signals" in res['reasoning']

    def test_rule_6_weak_signals_below_threshold(self, engine):
        """Rule 6: Low text confidence (< 0.60) without primary confirmation -> Unverified."""
        nli_res = _build_nli_result(entail=0.45, neutral=0.35, contra=0.20)
        res = engine.fuse_verdicts(nli_res)
        assert res['final_verdict'] == 'Unverified'
        assert "Insufficient conclusive evidence" in res['reasoning']


class TestTruthAnalysisMatrix:
    """Tests for the structured truth_analysis breakdown comparing rumor vs actual truth."""

    def test_truth_analysis_for_contradicted_rumor(self, engine):
        claim = "All Tamil Nadu ration cards will be cancelled from next month."
        evidence = [
            {
                'source': 'thehindu.com',
                'relationship': 'contradiction',
                'text': 'The Food Department clarified that ration cards are active and distribution is normal.'
            }
        ]
        ta = engine.generate_truth_analysis(
            claim=claim,
            final_verdict='Contradicted',
            final_confidence=0.89,
            evidence=evidence,
            primary_origins=[{'name': 'Government of Tamil Nadu'}]
        )

        assert ta['status'] == 'rumor'
        assert ta['badge_label'] == 'DEBUNKED RUMOR / FALSE'
        assert ta['badge_color'] == 'rose'
        assert ta['rumor_statement'] == claim
        assert "Food Department clarified" in ta['actual_truth']
        assert "This statement is a rumor and NOT actual truth" in ta['verdict_meaning']
        assert "Refuted by Government of Tamil Nadu" in ta['evidence_basis']

    def test_truth_analysis_for_supported_truth(self, engine):
        claim = "James Webb Telescope detected water vapor on exoplanet K2-18b."
        evidence = [
            {
                'source': 'nasa.gov',
                'relationship': 'entailment',
                'text': 'Spectroscopic data reveals methane and carbon dioxide with possible water vapor markers.'
            }
        ]
        ta = engine.generate_truth_analysis(
            claim=claim,
            final_verdict='Supported',
            final_confidence=0.94,
            evidence=evidence,
            primary_origins=[{'name': 'NASA'}]
        )

        assert ta['status'] == 'verified_truth'
        assert ta['badge_label'] == 'VERIFIED ACTUAL TRUTH'
        assert ta['badge_color'] == 'emerald'
        assert "verified factual truth" in ta['actual_truth']
        assert "Spectroscopic data" in ta['actual_truth']
        assert "verified as genuine truth" in ta['verdict_meaning']

    def test_truth_analysis_for_unverified_rumor(self, engine):
        claim = "Alien mothership detected orbiting near Jupiter moons."
        ta = engine.generate_truth_analysis(
            claim=claim,
            final_verdict='Unverified',
            final_confidence=0.45,
            evidence=[],
            primary_origins=[{'name': 'NASA'}]
        )

        assert ta['status'] == 'unverified_rumor'
        assert ta['badge_label'] == 'UNCONFIRMED / POTENTIAL RUMOR'
        assert ta['badge_color'] == 'amber'
        assert "No official notification" in ta['actual_truth']
        assert "lacks verified truth data" in ta['verdict_meaning']

    def test_truth_analysis_empty_evidence_and_origins_fallback(self, engine):
        """Should not crash when evidence and primary origins are None."""
        ta = engine.generate_truth_analysis(
            claim="",
            final_verdict='Contradicted',
            final_confidence=0.80,
            evidence=None,
            primary_origins=None
        )

        assert ta['status'] == 'rumor'
        assert len(ta['actual_truth']) > 0
        assert len(ta['evidence_basis']) > 0
