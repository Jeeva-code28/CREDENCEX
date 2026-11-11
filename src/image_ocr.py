# src/image_ocr.py
# Multimodal Image Claim Extractor supporting Tamil ('ta') and English ('en').
# Extracts textual claims, circulars, and news headlines from uploaded images,
# breaking news card templates, social media banners, and government orders.

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class ImageClaimExtractor:
    """
    Extracts text claims and headlines from news images, circulars, and posters
    using EasyOCR with multilingual support (Tamil and English).
    """

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Initialize the ImageClaimExtractor.

        Args:
            languages: List of language codes. Defaults to ['ta', 'en'] (Tamil & English).
            use_gpu: Whether to run OCR on GPU. If None, auto-detects CUDA availability.
        """
        self.languages = languages or ['ta', 'en']
        self.use_gpu = use_gpu if use_gpu is not None else (HAS_TORCH and torch and torch.cuda.is_available())
        self._reader = None
        self._reader_available = None

        # Known Tamil and Indian news channel card branding markers
        self.news_templates = [
            'puthiyathalaimurai', 'puthiya thalaimurai', 'புதிய தலைமுறை',
            'thanthi tv', 'thanthi', 'daily thanthi', 'தந்தி டிவி', 'தந்தி',
            'polimer news', 'polimer', 'பாலிமர்',
            'news18 tamil nadu', 'news18', 'நியூஸ்18',
            'sun news', 'சன் நியூஸ்',
            'dinamalar', 'தினமலர்',
            'dinakaran', 'தினகரன்',
            'kalaignar news', 'கலைஞர் செய்தி',
            'breaking news', 'பிரேக்கிங் நியூஸ்', 'முக்கிய செய்தி'
        ]

    def _get_reader(self):
        """Lazy-load the EasyOCR reader with robust fallback to ensure fast and reliable startup."""
        if self._reader is not None:
            return self._reader

        if self._reader_available is False:
            return None

        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages {self.languages} (GPU={self.use_gpu})...")
            try:
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            except Exception as e_lang:
                logger.warning(
                    f"EasyOCR with languages {self.languages} failed ({e_lang}). "
                    "Falling back to English OCR engine."
                )
                self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)

            self._reader_available = True
            logger.info("EasyOCR engine successfully initialized.")
            return self._reader
        except Exception as e:
            logger.warning(f"EasyOCR could not be initialized ({e}). Fallback to lightweight image text heuristics.")
            self._reader_available = False
            return None

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy for social media and WhatsApp compressed cards:
        - Resizes small images
        - Normalizes contrast and applies bilateral filtering to suppress compression noise.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed image as a numpy array.
        """
        img = cv2.imread(image_path)
        if img is None:
            # Fallback for PIL-compatible formats
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Upscale if image is too small for small fonts
        if w < 600:
            scale = 600.0 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Bilateral filter to smooth noise while keeping text edges sharp
        filtered = cv2.bilateralFilter(img, 7, 50, 50)
        return filtered

    @timer
    def extract_claim_from_image(self, image_path: str, min_confidence: float = 0.20) -> Dict:
        """
        Extract textual claim from an image file.

        Args:
            image_path: Path to the news image, poster, circular, or screenshot.
            min_confidence: Minimum OCR confidence threshold for candidate text tokens.

        Returns:
            Dict containing:
                - extracted_text: Full clean extracted text
                - primary_claim: Most salient headline/claim statement
                - detected_template: Identified news channel or card template (if any)
                - confidence: Average confidence score
                - has_text: Boolean indicating whether text was found
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {
                'extracted_text': '',
                'primary_claim': '',
                'detected_template': None,
                'confidence': 0.0,
                'has_text': False
            }

        reader = self._get_reader()
        if reader is None:
            # Fallback if OCR library is unavailable: use clean image stem
            stem = Path(image_path).stem.replace('_', ' ').replace('-', ' ')
            return {
                'extracted_text': stem,
                'primary_claim': stem,
                'detected_template': None,
                'confidence': 0.0,
                'has_text': bool(stem.strip())
            }

        try:
            preprocessed_img = self.preprocess_image(image_path)
            # EasyOCR detail=1 returns (bbox, text, confidence)
            raw_ocr = reader.readtext(preprocessed_img, detail=1)

            extracted_lines = []
            confidences = []
            detected_template = None

            # Ignore typical mobile screenshot header artifacts
            mobile_header_patterns = [
                r'^\d{1,2}:\d{2}', r'^\d+%', r'^volte', r'^4g', r'^5g',
                r'^lte', r'^wifi', r'^am$', r'^pm$', r'^battery'
            ]

            for item in raw_ocr:
                bbox, text, conf = item[0], item[1].strip(), float(item[2])

                if conf < min_confidence or not text:
                    continue

                # Check for mobile UI junk
                lower_t = text.lower()
                if any(re.match(p, lower_t) for p in mobile_header_patterns):
                    continue

                # Detect news card branding
                for tmpl in self.news_templates:
                    if tmpl in lower_t:
                        detected_template = tmpl.title()
                        break

                extracted_lines.append(text)
                confidences.append(conf)

            full_text = " ".join(extracted_lines)
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            # Identify primary headline: longest meaningful block or first 2 substantial lines
            significant_lines = [l for l in extracted_lines if len(l.split()) >= 3 or len(l) >= 15]
            if significant_lines:
                primary_claim = " ".join(significant_lines[:3])
            elif extracted_lines:
                primary_claim = " ".join(extracted_lines)
            else:
                primary_claim = Path(image_path).stem.replace('_', ' ').replace('-', ' ')

            # Extract poster & circular specific signals: handles, milestones, production entities
            detected_handles = list(dict.fromkeys(re.findall(r'@[A-Za-z0-9_]{3,30}', full_text)))
            detected_milestones = list(dict.fromkeys(re.findall(r'\b\d+\+?\s*(?:crores?|cr|crore rs|lakhs?|billion|million)\b', full_text, re.IGNORECASE)))
            
            # Detect studio / production signatures in poster
            detected_production_entities = []
            known_studios = [
                'dream warrior pictures', 'dream warrior', 'sun pictures', 'lyca productions',
                'lyca', '2d entertainment', 'red giant movies', 'studio green', 'mythri movie makers',
                'hombale films', 'dharma productions', 'yash raj films', 'yrf', 'seven screen studio',
                'ags entertainment', 'director', 'produced by', 'directed by', 'starring'
            ]
            lower_full = full_text.lower()
            for ks in known_studios:
                if ks in lower_full:
                    detected_production_entities.append(ks.title())

            has_text = len(extracted_lines) > 0

            logger.info(
                f"Image OCR extracted {len(extracted_lines)} lines "
                f"(avg_conf={avg_confidence:.2f}, template={detected_template}, "
                f"entities={detected_production_entities}, milestones={detected_milestones}): '{primary_claim[:80]}...'"
            )

            return {
                'extracted_text': full_text,
                'primary_claim': primary_claim,
                'detected_template': detected_template,
                'detected_production_entities': detected_production_entities,
                'detected_milestones': detected_milestones,
                'detected_handles': detected_handles,
                'confidence': avg_confidence,
                'has_text': has_text
            }

        except Exception as e:
            logger.error(f"Error during image OCR extraction: {e}")
            stem = Path(image_path).stem.replace('_', ' ').replace('-', ' ')
            return {
                'extracted_text': stem,
                'primary_claim': stem,
                'detected_template': None,
                'confidence': 0.0,
                'has_text': False
            }
