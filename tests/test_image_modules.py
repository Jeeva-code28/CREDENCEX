# tests/test_image_modules.py
# Unit tests for image authenticity detection and image context verification modules.

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
try:
    import torch
except ImportError:
    torch = MagicMock()
from PIL import Image

from src.image_authenticity import ImageAuthenticityDetector
from src.image_context import ImageContextVerifier


@pytest.fixture
def sample_test_image(tmp_path):
    """Create a temporary valid JPEG image for testing."""
    img_path = tmp_path / "test_sample.jpg"
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(str(img_path), "JPEG", quality=95)
    return str(img_path)


class TestImageAuthenticityDetector:
    """Tests for ImageAuthenticityDetector."""

    def test_init_without_checkpoint(self):
        """Should disable CNN manipulation scoring when no fine-tuned weights exist."""
        detector = ImageAuthenticityDetector(
            model_name='resnet50',
            checkpoint_path="non_existent_weights.pth"
        )
        assert detector.cnn_available is False
        assert detector.model is None

    def test_cnn_prediction_unavailable_when_no_checkpoint(self, sample_test_image):
        """cnn_prediction should return unavailable rather than random guesses."""
        detector = ImageAuthenticityDetector(
            model_name='resnet50',
            checkpoint_path="non_existent_weights.pth"
        )
        result = detector.cnn_prediction(sample_test_image)

        assert result['available'] is False
        assert result['prediction'] == 'unavailable'
        assert result['authentic_prob'] is None
        assert result['manipulated_prob'] is None

    def test_error_level_analysis(self, sample_test_image):
        """ELA should produce a valid 2D error map and normalized score in [0, 1]."""
        detector = ImageAuthenticityDetector(checkpoint_path="none.pth")
        error_map, error_level = detector.error_level_analysis(sample_test_image)

        assert isinstance(error_map, np.ndarray)
        assert 0.0 <= error_level <= 1.0

    def test_spectral_analysis(self, sample_test_image):
        """Spectral analysis should return valid magnitude and anomaly score."""
        detector = ImageAuthenticityDetector(checkpoint_path="none.pth")
        spectral = detector.spectral_analysis(sample_test_image)

        assert 'avg_magnitude' in spectral
        assert 'std_magnitude' in spectral
        assert 'anomaly_score' in spectral
        assert spectral['anomaly_score'] >= 0.0

    def test_assess_authenticity_fallback(self, sample_test_image):
        """Without CNN weights, assessment should safely use deterministic ELA and Spectral analysis."""
        detector = ImageAuthenticityDetector(checkpoint_path="none.pth")
        assessment = detector.assess_authenticity(sample_test_image)

        assert 'authenticity_score' in assessment
        assert 0.0 <= assessment['authenticity_score'] <= 1.0
        assert assessment['verdict'] in ['authentic', 'potentially_manipulated']
        assert assessment['cnn_prediction']['available'] is False


class TestImageContextVerifier:
    """Tests for ImageContextVerifier."""

    def test_legacy_model_name_normalization(self):
        """Legacy name 'ViT-B/32' should map to Hugging Face 'openai/clip-vit-base-patch32'."""
        pytest.importorskip("transformers")
        with patch('transformers.CLIPModel.from_pretrained') as mock_model, \
             patch('transformers.CLIPProcessor.from_pretrained') as mock_proc, \
             patch('src.image_context.SentenceTransformer'):
            
            mock_model.return_value = MagicMock()
            mock_proc.return_value = MagicMock()

            verifier = ImageContextVerifier(clip_model_name="ViT-B/32")
            assert verifier._clip_available is True
            mock_model.assert_called_with("openai/clip-vit-base-patch32")

    def test_check_temporal_consistency_contemporary(self):
        """Contemporary image with mentioned dates should score consistent."""
        verifier = ImageContextVerifier.__new__(ImageContextVerifier)
        attributes = {'time_period': {'predicted': 'contemporary'}}
        temporal_info = {'mentioned_dates': ['2024-01-15']}

        result = verifier._check_temporal_consistency(attributes, temporal_info)
        assert result['score'] >= 0.6

    def test_check_entity_consistency_with_people(self):
        """Entity consistency should score high when people are detected for person claims."""
        verifier = ImageContextVerifier.__new__(ImageContextVerifier)
        attributes = {'objects': {'predicted': 'people'}}
        claim_text = "Bill Gates announced a program"
        entities = ["Bill Gates"]

        result = verifier._check_entity_consistency(claim_text, entities, attributes)
        assert result['score'] >= 0.7

    def test_fallback_when_clip_unavailable(self, sample_test_image):
        """When CLIP is unavailable, alignment and attributes should return safe fallback dicts."""
        verifier = ImageContextVerifier.__new__(ImageContextVerifier)
        verifier._clip_available = False

        alignment = verifier.compute_image_text_alignment(sample_test_image, "test claim")
        assert alignment['similarity_score'] == 0.5
        assert 'CLIP not available' in alignment['alignment']

        attrs = verifier.extract_image_attributes(sample_test_image)
        assert attrs['location_type']['predicted'] == 'unknown'
