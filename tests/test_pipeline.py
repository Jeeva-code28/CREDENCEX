# tests/test_pipeline.py
# Integration tests for the full News Verification Pipeline.

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion_engine import FusionEngine
from src.claim_extraction import Claim


class TestFusionEngine:
    """Tests for the FusionEngine verdict fusion logic."""

    @pytest.fixture
    def engine(self):
        return FusionEngine()

    def _make_nli_result(self, entail=0.0, neutral=0.0, contra=0.0):
        """Helper to create an NLI result dict."""
        primary = max(
            [('entailment', entail), ('neutral', neutral), ('contradiction', contra)],
            key=lambda x: x[1]
        )
        return {
            'individual_results': [],
            'aggregated': {
                'weighted_probabilities': {
                    'entailment': entail,
                    'neutral': neutral,
                    'contradiction': contra
                },
                'primary_verdict': primary[0],
                'confidence': primary[1]
            }
        }

    def test_strong_entailment_no_image(self, engine):
        """Strong entailment without image → Supported."""
        nli_result = self._make_nli_result(entail=0.85, neutral=0.10, contra=0.05)
        result = engine.fuse_verdicts(nli_result)

        assert result['final_verdict'] == 'Supported'
        assert result['final_confidence'] > 0.7

    def test_strong_contradiction_no_image(self, engine):
        """Strong contradiction without image → Contradicted."""
        nli_result = self._make_nli_result(entail=0.05, neutral=0.10, contra=0.85)
        result = engine.fuse_verdicts(nli_result)

        assert result['final_verdict'] == 'Contradicted'
        assert result['final_confidence'] > 0.7

    def test_weak_signal_unverified(self, engine):
        """Weak NLI confidence → Unverified."""
        nli_result = self._make_nli_result(entail=0.35, neutral=0.35, contra=0.30)
        result = engine.fuse_verdicts(nli_result)

        assert result['final_verdict'] == 'Unverified'

    def test_image_contradiction_overrides(self, engine):
        """Strong image contradiction should produce Contradicted."""
        nli_result = self._make_nli_result(entail=0.5, neutral=0.3, contra=0.2)
        image_result = {
            'authenticity': {
                'verdict': 'potentially_manipulated',
                'authenticity_score': 0.8
            },
            'context': {
                'verdict': 'context_mismatch',
                'overall_context_score': 0.2
            }
        }

        result = engine.fuse_verdicts(nli_result, image_result)
        assert result['final_verdict'] in ['Contradicted', 'Unverified']

    def test_conflicting_text_image_unverified(self, engine):
        """Entailment text + contradiction image → Unverified."""
        nli_result = self._make_nli_result(entail=0.7, neutral=0.2, contra=0.1)
        image_result = {
            'authenticity': {
                'verdict': 'potentially_manipulated',
                'authenticity_score': 0.4
            },
            'context': {
                'verdict': 'context_mismatch',
                'overall_context_score': 0.3
            }
        }

        result = engine.fuse_verdicts(nli_result, image_result)
        assert result['final_verdict'] == 'Unverified'

    def test_generate_report(self, engine):
        """Report generation should produce a formatted string."""
        nli_result = self._make_nli_result(entail=0.8, neutral=0.1, contra=0.1)
        report = engine.generate_report("Test claim", nli_result)

        assert isinstance(report, str)
        assert "CLAIM VERIFICATION REPORT" in report
        assert "Test claim" in report
        assert "FINAL VERDICT" in report

    def test_generate_report_with_image(self, engine):
        """Report with image results should include image section."""
        nli_result = self._make_nli_result(entail=0.8, neutral=0.1, contra=0.1)
        image_result = {
            'authenticity': {
                'verdict': 'authentic',
                'authenticity_score': 0.9
            },
            'context': {
                'verdict': 'contextually_consistent',
                'overall_context_score': 0.8
            }
        }

        report = engine.generate_report("Test claim", nli_result, image_result)

        assert "IMAGE ANALYSIS" in report
        assert "authentic" in report.lower()


class TestNLIVerifier:
    """Tests for NLI verification (mocked model)."""

    @patch('src.nli_verification.AutoModelForSequenceClassification')
    @patch('src.nli_verification.AutoTokenizer')
    def test_aggregate_empty_results(self, mock_tokenizer, mock_model):
        """Empty results should return neutral verdict."""
        from src.nli_verification import NLIVerifier

        verifier = NLIVerifier.__new__(NLIVerifier)
        verifier.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        aggregated = verifier._aggregate_nli_results([])

        assert aggregated['primary_verdict'] == 'neutral'
        assert aggregated['confidence'] == 0.0

    @patch('src.nli_verification.AutoModelForSequenceClassification')
    @patch('src.nli_verification.AutoTokenizer')
    def test_aggregate_single_result(self, mock_tokenizer, mock_model):
        """Single result should pass through directly."""
        from src.nli_verification import NLIVerifier

        verifier = NLIVerifier.__new__(NLIVerifier)
        verifier.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        results = [{
            'evidence': {'score': 1.0},
            'probabilities': {
                'entailment': 0.8,
                'neutral': 0.1,
                'contradiction': 0.1
            }
        }]

        aggregated = verifier._aggregate_nli_results(results)

        assert aggregated['primary_verdict'] == 'entailment'
        assert abs(aggregated['confidence'] - 0.8) < 0.01

    @patch('src.nli_verification.AutoModelForSequenceClassification')
    @patch('src.nli_verification.AutoTokenizer')
    def test_verify_claim_premise_hypothesis_ordering(self, mock_tokenizer_cls, mock_model_cls):
        """Tokenizer should receive evidence as premise and claim as hypothesis."""
        pytest.importorskip("torch")
        import torch
        from src.nli_verification import NLIVerifier

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.ones(2, 10, dtype=torch.long),
            'attention_mask': torch.ones(2, 10, dtype=torch.long)
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        mock_model.return_value = mock_outputs
        mock_model.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        mock_model_cls.from_pretrained.return_value = mock_model

        verifier = NLIVerifier(model_name="mock-model")

        claim = "Test Claim Statement"
        evidence_list = [
            {'text': 'Evidence passage 1', 'score': 1.0},
            {'text': 'Evidence passage 2', 'score': 1.0},
        ]

        result = verifier.verify_claim_against_evidence(claim, evidence_list, batch_size=2)

        # Check call arguments to tokenizer
        assert mock_tokenizer.call_count == 1
        call_args = mock_tokenizer.call_args[0]
        called_premises = call_args[0]
        called_hypotheses = call_args[1]

        # Verify that premise contains the evidence text and hypothesis contains the claim
        assert called_premises == ['Evidence passage 1', 'Evidence passage 2']
        assert called_hypotheses == [claim, claim]
        assert len(result['individual_results']) == 2


class TestPipelineIntegration:
    """
    Integration tests for the full pipeline.
    These tests require model downloads and are marked as slow.
    Run with: pytest -m integration
    """

    @pytest.mark.integration
    def test_text_verification(self):
        """Full pipeline should process a text claim end-to-end."""
        from main import NewsVerificationPipeline

        pipeline = NewsVerificationPipeline()
        result = pipeline.verify(
            "The Earth is flat.",
            input_type='text'
        )

        assert 'final_verdict' in result
        assert result['final_verdict']['final_verdict'] in [
            'Supported', 'Contradicted', 'Unverified'
        ]

    @pytest.mark.integration
    def test_image_verification(self):
        """Pipeline should handle image input."""
        from main import NewsVerificationPipeline

        test_image = Path('data/test_images/test.jpg')
        if not test_image.exists():
            pytest.skip("Test image not available")

        pipeline = NewsVerificationPipeline()
        result = pipeline.verify(str(test_image), input_type='image')

        assert 'image_result' in result
        if result['image_result']:
            assert result['image_result']['authenticity']['verdict'] in [
                'authentic', 'potentially_manipulated'
            ]
