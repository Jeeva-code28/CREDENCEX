# main.py
# Phase 7: End-to-End News Verification Pipeline
# Orchestrates all modules: claim extraction → evidence retrieval →
# evidence ranking → NLI verification → image analysis → fusion.

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.claim_extraction import ClaimExtractor, generate_queries
from src.evidence_retrieval import EvidenceRetriever
from src.evidence_ranking import EvidenceRanker
from src.nli_verification import NLIVerifier
from src.image_authenticity import ImageAuthenticityDetector
from src.image_context import ImageContextVerifier
from src.image_ocr import ImageClaimExtractor
from src.fusion_engine import FusionEngine
from src.authority_sources import detect_authorities
from src.utils import setup_logger, load_config

logger = setup_logger(__name__)


class NewsVerificationPipeline:
    """
    End-to-end pipeline for claim-centric multimodal news verification.

    Supports three input types:
    - text: Raw claim text
    - url: URL to a news article (extracts text via BeautifulSoup)
    - image: Image file path (runs image analysis in addition to text pipeline)
    """

    def __init__(self, config_path: str = 'configs/model_config.yaml'):
        """
        Initialize all pipeline components.

        Args:
            config_path: Path to the YAML configuration file.
        """
        print("=" * 60)
        print("  Initializing News Verification Pipeline...")
        print("=" * 60)

        # Load configuration
        self.config = load_config(config_path)

        # Initialize components
        print("  [1/8] Loading Claim Extractor (spaCy NER)...")
        self.claim_extractor = ClaimExtractor(
            ner_model=self.config['claim_extraction']['ner_model']
        )

        print("  [2/8] Loading Evidence Retriever (sentence-transformers)...")
        self.evidence_retriever = EvidenceRetriever(
            model_name=self.config['evidence_retrieval']['semantic_model']
        )

        print("  [3/8] Initializing Evidence Ranker...")
        self.evidence_ranker = EvidenceRanker(
            semantic_model=self.evidence_retriever.semantic_model,
            weights=self.config['evidence_ranking']['weights']
        )

        print("  [4/8] Loading NLI Verifier (DeBERTa)...")
        self.nli_verifier = NLIVerifier(
            model_name=self.config['nli_model']['model_name']
        )

        print("  [5/8] Loading Image Authenticity Detector (ResNet50 / ELA / FFT)...")
        self.image_detector = ImageAuthenticityDetector(
            model_name=self.config['image_auth']['cnn_model'],
            checkpoint_path=self.config['image_auth'].get('checkpoint_path')
        )

        print("  [6/8] Loading Image Context Verifier (HuggingFace CLIP)...")
        self.image_verifier = ImageContextVerifier(
            clip_model_name=self.config['image_context']['clip_model'],
            text_encoder_name=self.config['image_context']['text_encoder'],
            text_encoder=self.evidence_retriever.semantic_model
        )

        print("  [7/8] Loading Multimodal Image OCR Engine (EasyOCR for Tamil & English)...")
        self.image_ocr = ImageClaimExtractor()

        print("  [8/8] Initializing Fusion Engine...")
        self.fusion_engine = FusionEngine()

        print("\n  [OK] Pipeline initialization complete!")
        print("=" * 60)

    def verify(
        self,
        input_data: str,
        input_type: str = 'text',
        max_evidence: Optional[int] = None,
        claim_text: Optional[str] = None
    ) -> dict:
        """
        Main verification function.

        Args:
            input_data: Text claim, URL, or image file path.
            input_type: One of 'text', 'url', or 'image'.
            max_evidence: Optional maximum number of evidence items to rank and verify.
            claim_text: Optional claim text when verifying images.

        Returns:
            Dict with claim, evidence, nli_result, image_result,
            extracted_ocr, final_verdict, and human-readable report.
        """
        print(f"\n{'=' * 60}")
        print(f"  Processing {input_type.upper()} input...")
        print(f"{'=' * 60}")

        # -- Step 1: Extract and structure claim --
        print("\n  [1/8] Extracting and structuring claim...")
        extracted_ocr = None
        if input_type == 'url':
            import requests
            import re
            from bs4 import BeautifulSoup
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/124.0.0.0 Safari/537.36'
                )
            }
            response = requests.get(input_data, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Strip non-content boilerplate elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'form', 'svg']):
                tag.decompose()

            # Locate primary headline
            headline = ""
            h1 = soup.find('h1')
            if h1:
                headline = h1.get_text(strip=True)

            # Locate primary article container
            article = soup.find('article') or soup.find('main') or soup.find(class_=re.compile(r'article[-_]?body|story[-_]?content|post[-_]?content|entry[-_]?content', re.I))
            search_scope = article if article else soup

            paragraphs = [
                p.get_text(strip=True) for p in search_scope.find_all('p')
                if len(p.get_text(strip=True)) > 25
            ]

            if paragraphs:
                body_content = ' '.join(paragraphs[:8])
                text = f"{headline}. {body_content}" if headline and headline not in body_content else body_content
            elif headline:
                text = f"{headline}. {search_scope.get_text(separator=' ', strip=True)[:1500]}"
            else:
                text = soup.get_text(separator=' ', strip=True)[:1500]
        elif input_type == 'image':
            # Always scan image for poster text, production credits, handles & numbers
            if os.path.exists(input_data):
                print("    [OCR] Scanning image/poster for headlines, text, credits & watermarks...")
                try:
                    ocr_res = self.image_ocr.extract_claim_from_image(input_data)
                    if ocr_res.get('has_text') and ocr_res.get('extracted_text'):
                        extracted_ocr = ocr_res
                        print(f"    [OCR OK] Detected text: '{ocr_res['primary_claim'][:80]}...'")
                        if ocr_res.get('detected_production_entities'):
                            print(f"    [OCR OK] Detected entities: {ocr_res['detected_production_entities']}")
                except Exception as e:
                    logger.debug(f"OCR extraction error: {e}")

            if claim_text and claim_text.strip():
                if extracted_ocr and extracted_ocr.get('extracted_text'):
                    text = f"{claim_text.strip()} (Poster details: {extracted_ocr['extracted_text']})"
                else:
                    text = claim_text.strip()
            elif extracted_ocr and extracted_ocr.get('primary_claim'):
                text = extracted_ocr['primary_claim']
            else:
                stem = Path(input_data).stem
                text = stem.replace('_', ' ').replace('-', ' ') or "News image verification"
        else:
            text = input_data

        claim = self.claim_extractor.extract_claim(text, ocr_data=extracted_ocr)
        print(f"    [OK] Extracted entities: {', '.join(claim.entities[:3]) or 'None'}")
        print(f"    [OK] Main event: {claim.event}")
        print(f"    [OK] Domain: {claim.domain}")
        if getattr(claim, 'primary_origins', None):
            po_names = [po['name'] for po in claim.primary_origins]
            print(f"    [OK] Primary Origins Detected: {', '.join(po_names)}")

        # Determine effective top_k
        top_k = max_evidence if (max_evidence is not None and max_evidence > 0) else self.config['evidence_retrieval'].get('top_k', 10)

        # -- Step 2: Generate retrieval queries --
        print("\n  [2/8] Generating retrieval queries with Primary Origin Priority...")
        queries = generate_queries(claim)
        print(f"    [OK] Generated {len(queries)} queries")

        # -- Step 3: Retrieve evidence from official authorities & trusted sources --
        print("\n  [3/8] Retrieving evidence from official authorities, primary channels & multi-source databases...")
        evidence = self.evidence_retriever.retrieve_evidence(
            queries,
            max_results=max(top_k * 4, 24),
            claim_text=claim.original_text,
            entities=claim.entities,
            primary_origins=getattr(claim, 'primary_origins', None)
        )
        print(f"    [OK] Retrieved {len(evidence)} evidence passages")

        # -- Step 4: Rank evidence with primary authority priority --
        print("\n  [4/8] Ranking evidence (Primary Official Origins & Accredited Wires weighted highest)...")
        dates = claim.temporal.get('mentioned_dates', []) if (claim and claim.temporal) else []
        claim_date = dates[0] if dates else None
        ranked_evidence = self.evidence_ranker.rank_evidence(
            claim.original_text if (claim and claim.original_text) else ' '.join(queries),
            evidence,
            claim_date=claim_date,
            primary_origins=getattr(claim, 'primary_origins', None)
        )
        ranked_evidence = ranked_evidence[:top_k]
        print(f"    [OK] Top-{top_k} evidence selected")

        # -- Step 5: NLI Verification --
        print("\n  [5/8] Performing NLI verification...")
        nli_result = self.nli_verifier.verify_claim_against_evidence(
            claim.original_text,
            ranked_evidence,
            max_evidence=top_k
        )
        # Propagate per-evidence NLI stance & probabilities onto ranked_evidence
        # so FusionEngine, truth analysis, and downstream reports have full context
        indiv_by_text = {}
        for res in nli_result.get('individual_results', []):
            pred = res.get('prediction', 'neutral')
            probs = res.get('probabilities', {})
            conf = float(probs.get(pred, 0.0)) if probs else 0.0
            ev_ref = res.get('evidence')
            if isinstance(ev_ref, dict):
                ev_ref['relationship'] = pred
                ev_ref['stance'] = pred
                ev_ref['nli_probabilities'] = probs
                ev_ref['nli_confidence'] = conf
                indiv_by_text[ev_ref.get('text', '').strip()] = (pred, probs, conf)

        for ev in ranked_evidence:
            if 'relationship' not in ev:
                key = ev.get('text', '').strip()
                if key in indiv_by_text:
                    pred, probs, conf = indiv_by_text[key]
                    ev['relationship'] = pred
                    ev['stance'] = pred
                    ev['nli_probabilities'] = probs
                    ev['nli_confidence'] = conf

        print(f"    [OK] Verdict: {nli_result['aggregated']['primary_verdict'].upper()}")
        print(f"    [OK] Confidence: {nli_result['aggregated']['confidence']:.2%}")

        # -- Step 6: Image Analysis (if applicable) --
        image_result = None
        if input_type == 'image' and os.path.exists(input_data):
            print("\n  [6a/8] Analyzing image authenticity...")
            authenticity = self.image_detector.assess_authenticity(input_data)
            print(f"    [OK] Authenticity: {authenticity['verdict']}")
            print(f"    [OK] Score: {authenticity['authenticity_score']:.2%}")

            print("\n  [6b/8] Verifying image context...")
            context = self.image_verifier.verify_image_context(
                input_data,
                claim.original_text,
                claim.entities,
                claim.temporal
            )
            print(f"    [OK] Context verdict: {context['verdict']}")

            image_result = {
                'authenticity': authenticity,
                'context': context,
                'ocr': extracted_ocr
            }
        else:
            print("\n  [6/8] No image input - skipping image analysis")

        # -- Step 7: Fusion and final verdict --
        print("\n  [7/8] Fusing multimodal evidence with Primary Origin Corroboration...")
        final_result = self.fusion_engine.fuse_verdicts(
            nli_result,
            image_result,
            primary_origins=getattr(claim, 'primary_origins', None),
            evidence=ranked_evidence,
            claim=claim.original_text
        )
        print(f"    [OK] Final verdict: {final_result['final_verdict']}")
        print(f"    [OK] Confidence: {final_result['final_confidence']:.2%}")

        # Generate report
        report = self.fusion_engine.generate_report(
            claim.original_text,
            nli_result,
            image_result,
            primary_origins=getattr(claim, 'primary_origins', None),
            evidence=ranked_evidence,
            fusion_result=final_result
        )

        return {
            'claim': claim,
            'evidence': ranked_evidence,
            'nli_result': nli_result,
            'image_result': image_result,
            'extracted_ocr': extracted_ocr,
            'final_verdict': final_result,
            'report': report
        }


def main():
    """CLI entry point for the News Verification Pipeline."""
    parser = argparse.ArgumentParser(
        description="Claim-Centric Multimodal News Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "Bill Gates announced a new vaccine on January 15, 2024"
  python main.py --url "https://example.com/news-article"
  python main.py --image "path/to/suspicious_image.jpg"
  python main.py --text "The Earth is flat" --output result.json
        """
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text', type=str, help='Text claim to verify'
    )
    input_group.add_argument(
        '--url', type=str, help='URL of a news article to verify'
    )
    input_group.add_argument(
        '--image', type=str, help='Path to an image to verify'
    )

    # Optional arguments
    parser.add_argument(
        '--config', type=str, default='configs/model_config.yaml',
        help='Path to configuration YAML file (default: configs/model_config.yaml)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save JSON result (optional)'
    )

    args = parser.parse_args()

    # Determine input type and data
    if args.text:
        input_type = 'text'
        input_data = args.text
    elif args.url:
        input_type = 'url'
        input_data = args.url
    elif args.image:
        input_type = 'image'
        input_data = args.image

    # Run pipeline
    pipeline = NewsVerificationPipeline(config_path=args.config)
    result = pipeline.verify(input_data, input_type=input_type)

    # Print report
    print(result['report'])

    # Save result if output path specified
    output_path = args.output or 'verification_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'verdict': result['final_verdict'],
            'claim': result['claim'].original_text,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\n  Result saved to: {output_path}")


if __name__ == '__main__':
    main()
