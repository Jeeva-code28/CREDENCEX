# tests/test_image_ocr.py
# Unit tests for Multimodal Image Claim Extractor (EasyOCR).

import os
import tempfile
import pytest
from PIL import Image, ImageDraw

from src.image_ocr import ImageClaimExtractor


@pytest.fixture
def sample_image():
    """Create a temporary image with high-contrast text for OCR testing."""
    img = Image.new('RGB', (600, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Simple plain text in black
    d.text((30, 80), "RBI NOTIFICATION 2024", fill=(0, 0, 0))

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    img.save(temp_path)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestImageClaimExtractor:
    """Tests for ImageClaimExtractor."""

    def test_init_extractor(self):
        extractor = ImageClaimExtractor()
        assert 'ta' in extractor.languages
        assert 'en' in extractor.languages

    def test_extract_from_nonexistent_image(self):
        extractor = ImageClaimExtractor()
        res = extractor.extract_claim_from_image("nonexistent_path_12345.jpg")
        assert res['has_text'] is False
        assert res['extracted_text'] == ""

    def test_extract_from_text_image(self, sample_image):
        extractor = ImageClaimExtractor()
        res = extractor.extract_claim_from_image(sample_image)
        assert isinstance(res, dict)
        assert 'primary_claim' in res
        assert 'confidence' in res
        assert 'has_text' in res
        # If EasyOCR weights are available, it reads text; otherwise fallback works safely
        assert len(res['primary_claim']) > 0
