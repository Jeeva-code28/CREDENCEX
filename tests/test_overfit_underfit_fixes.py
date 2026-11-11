# tests/test_overfit_underfit_fixes.py
# Regression tests for overfitting, underfitting, calibration, and integration fixes.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.authority_sources import UniversalOriginResolver, get_domain_authority_score
from src.claim_extraction import Claim, generate_queries
from src.evidence_retrieval import EvidenceRetriever
from src.fusion_engine import FusionEngine
from src.image_authenticity import ImageAuthenticityDetector
from src.image_context import ImageContextVerifier
from src.nli_verification import NLIVerifier


class TestAuthorityScoringFixes:
    """Ensure domain & handle authority matching cannot be spoofed by URL slugs or subdomains."""

    def test_handle_path_prefix_only_applies_to_social_hosts(self):
        nasa_origin = [{
            'name': 'NASA',
            'official_handles': ['@NASA'],
            'official_domains': ['nasa.gov']
        }]
        blog_url = "https://www.thetechedvocate.org/nasas-groundbreaking-detection-of-molecules/"
        score = get_domain_authority_score(blog_url, primary_entities=nasa_origin)
        assert score == 0.60, f"Blog slug starting with /nasa got spoofed authority {score}"

    def test_official_social_handle_exact_segment_required(self):
        nasa_origin = [{
            'name': 'NASA',
            'official_handles': ['@NASA'],
            'official_domains': ['nasa.gov']
        }]
        assert get_domain_authority_score("https://x.com/NASA/status/123", primary_entities=nasa_origin) == 0.99
        assert get_domain_authority_score("https://x.com/nasa_fanclub/status/123", primary_entities=nasa_origin) == 0.45

    def test_arbitrary_fact_check_path_not_inflated_to_ifcn_tier(self):
        url = "https://www.usatoday.com/story/news/factcheck/2023/01/10/earth-flat/12345/"
        assert get_domain_authority_score(url) == 0.90

    def test_domain_suffix_requires_dot_boundary(self):
        assert get_domain_authority_score("https://fakemyrbi.org.in.evil.com/notice") == 0.60
        assert get_domain_authority_score("https://www.rbi.org.in/press-release") == 0.99

    def test_primary_origin_queries_are_topic_anchored(self):
        claim = "NASA James Webb Space Telescope detected water vapor and carbon dioxide on exoplanet K2-18b."
        origins = UniversalOriginResolver.resolve_primary_origins(claim, entities=['NASA', 'James Webb Space', 'K2-18b'])
        assert origins
        joined = " | ".join(origins[0]['search_queries'])
        assert "K2-18b" in joined
        assert "site:nasa.gov" in joined
        assert "official announcement" not in joined.lower()
        assert "official (circular" not in joined.lower()

        c_obj = Claim(
            entities=['Earth', 'NASA'],
            event='Earth is round',
            temporal={'mentioned_dates': [], 'relative_times': []},
            domain='science',
            original_text='The Earth is round and NASA images show its curvature.',
            primary_origins=origins
        )
        queries = generate_queries(c_obj)
        assert len(queries) >= 2
        assert "curvature" in queries[1].lower() or "round" in queries[1].lower()


class TestRetrievalFilteringAndDedupFixes:
    """Ensure ad click trackers and staging/date-prefixed duplicates are filtered."""

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_duckduckgo_filters_bing_aclick_ads_and_staging(self, mock_st):
        with patch('duckduckgo_search.DDGS') as mock_ddgs_cls:
            mock_inst = MagicMock()
            mock_inst.text.return_value = [
                {
                    'title': 'Ad Fluoride Removal System',
                    'body': 'Fluoride removal water filtration systems available online now.',
                    'href': 'https://www.bing.com/aclick?ld=abc123&u=https%3A%2F%2Fexample.com'
                },
                {
                    'title': 'Staging mirror',
                    'body': 'Photos from space show that the Earth is round and not flat.',
                    'href': 'https://staging.politifact.com/factchecks/2018/earth-flat/'
                },
                {
                    'title': 'Official Fact Check',
                    'body': 'Photos from space show that the Earth is round and not flat.',
                    'href': 'https://www.politifact.com/factchecks/2024/earth-flat/'
                }
            ]
            mock_ddgs_cls.return_value = mock_inst

            retriever = EvidenceRetriever()
            results = retriever.retrieve_from_duckduckgo("earth is flat")

            urls = [r['url'] for r in results]
            assert len(urls) == 1
            assert urls[0] == 'https://www.politifact.com/factchecks/2024/earth-flat/'

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_dedup_prefers_longer_snippet_and_retains_wikipedia_deterministically(self, mock_st):
        retriever = EvidenceRetriever()
        long_wiki = (
            "Exoplanet K2-18b is a sub-Neptune discovered by Kepler and observed by JWST. "
            "JWST discovered water vapour, carbon dioxide and methane in its atmosphere."
        )
        short_nasa = "Sep 11, 2023 · A new investigation with NASA's James Webb Space Telescope revealed methane."
        long_nasa = (
            "Sep 11, 2023 · A new investigation with NASA's James Webb Space Telescope into K2-18 b "
            "revealed carbon dioxide and methane, supporting liquid water ocean hypotheses."
        )

        # Simulate fast general_web + truncated primary arriving first, and richer primary + Wikipedia arriving last
        ddg_fast = [
            {'source': 'duckduckgo', 'url': f'https://blog{i}.example.org/post-{i}', 'text': f'General commentary about exoplanet #{i} astronomy research.'}
            for i in range(8)
        ] + [
            {'source': 'duckduckgo_factcheck', 'url': 'https://science.nasa.gov/missions/webb/k2-18-b/', 'text': short_nasa}
        ]
        slow_rich_nasa = [
            {'source': 'duckduckgo', 'url': 'https://science.nasa.gov/missions/webb/k2-18-b/', 'text': long_nasa}
        ]

        call_count = {'n': 0}

        def fake_ddg(q, max_results=5, factcheck_targeted=False):
            call_count['n'] += 1
            return ddg_fast if call_count['n'] == 1 else slow_rich_nasa

        with patch.object(retriever, 'retrieve_from_duckduckgo', side_effect=fake_ddg), \
             patch.object(retriever, 'retrieve_from_google_factcheck', return_value=[]), \
             patch.object(retriever, 'retrieve_from_snopes', return_value=[]), \
             patch.object(retriever, 'retrieve_from_politifact', return_value=[]), \
             patch.object(retriever, 'retrieve_from_factcheck', return_value=[]), \
             patch.object(retriever, 'retrieve_from_web_search', return_value=[]), \
             patch.object(retriever, 'retrieve_from_wikipedia', return_value=[
                 {'source': 'wikipedia', 'url': 'https://en.wikipedia.org/wiki/K2-18b', 'title': 'K2-18b - Wikipedia', 'text': long_wiki}
             ]):
            res = retriever.retrieve_evidence(
                ["NASA James Webb Space Telescope detected water vapor and carbon dioxide on exoplanet K2-18b."],
                max_results=5,
                claim_text="NASA James Webb Space Telescope detected water vapor and carbon dioxide on exoplanet K2-18b.",
                entities=["NASA", "James Webb", "K2-18b"]
            )

        urls = [r.get('url') for r in res]
        assert len(res) == 5
        assert 'https://science.nasa.gov/missions/webb/k2-18-b/' in urls
        assert 'https://en.wikipedia.org/wiki/K2-18b' in urls
        nasa_item = next(r for r in res if 'science.nasa.gov' in (r.get('url') or ''))
        assert nasa_item['text'] == long_nasa


class TestNLIVerifierCalibrationAndFallback:
    """Verify heuristic fallback polarity/numbers and symmetric aggregator corroboration."""

    @pytest.fixture
    def heuristic_nli(self):
        nli = NLIVerifier.__new__(NLIVerifier)
        nli.has_neural_model = False
        nli.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        return nli

    def test_heuristic_handles_negation_numbers_and_entity_swaps(self, heuristic_nli):
        # 1. Opposite negation -> contradiction
        res1 = heuristic_nli.verify_claim_against_evidence(
            "The Earth is flat.",
            [{'text': "NASA confirms the Earth is not flat.", 'source': 'nasa.gov', 'source_tier': 'primary_origin'}]
        )['individual_results'][0]
        assert res1['prediction'] == 'contradiction'

        # 2. Matching negation -> entailment
        res2 = heuristic_nli.verify_claim_against_evidence(
            "The Earth is not flat.",
            [{'text': "Scientists agree that the Earth is not flat.", 'source': 'reuters', 'source_tier': 'news_wire'}]
        )['individual_results'][0]
        assert res2['prediction'] == 'entailment'

        # 3. Conflicting numerical claim -> contradiction
        res3 = heuristic_nli.verify_claim_against_evidence(
            "Tamil Nadu Government gives 50000 rupees to every college student.",
            [{'text': "Tamil Nadu Government provides 1000 rupees per month to eligible girl students.", 'source': 'thehindu', 'source_tier': 'mainstream_press'}]
        )['individual_results'][0]
        assert res3['prediction'] == 'contradiction'

        # 4. Prepositional entity swap -> contradiction
        res4 = heuristic_nli.verify_claim_against_evidence(
            "Apollo 11 astronauts landed on Mars in July 1969.",
            [{'text': "NASA's Apollo 11 mission landed the first humans on the Moon in July 1969.", 'source': 'nasa', 'source_tier': 'primary_origin'}]
        )['individual_results'][0]
        assert res4['prediction'] == 'contradiction'

        # 5. Disjoint partial overlap -> neutral
        res5 = heuristic_nli.verify_claim_against_evidence(
            "ISRO launched a manned mission to Venus in 2025.",
            [{'text': "ISRO is headquartered in Bengaluru and launched Chandrayaan-3 to the Moon in 2023.", 'source': 'isro', 'source_tier': 'primary_origin'}]
        )['individual_results'][0]
        assert res5['prediction'] == 'neutral'

    def test_single_general_web_contradiction_does_not_override_neutral_majority(self, heuristic_nli):
        results = [
            {
                'evidence': {'source': 'web1', 'source_tier': 'general_web', 'score': 0.7},
                'probabilities': {'entailment': 0.01, 'neutral': 0.96, 'contradiction': 0.03},
                'prediction': 'neutral'
            },
            {
                'evidence': {'source': 'web2', 'source_tier': 'general_web', 'score': 0.7},
                'probabilities': {'entailment': 0.02, 'neutral': 0.95, 'contradiction': 0.03},
                'prediction': 'neutral'
            },
            {
                'evidence': {'source': 'web3', 'source_tier': 'general_web', 'score': 0.7},
                'probabilities': {'entailment': 0.02, 'neutral': 0.95, 'contradiction': 0.03},
                'prediction': 'neutral'
            },
            {
                'evidence': {'source': 'web4', 'source_tier': 'general_web', 'score': 0.7},
                'probabilities': {'entailment': 0.02, 'neutral': 0.95, 'contradiction': 0.03},
                'prediction': 'neutral'
            },
            {
                'evidence': {'source': 'web5', 'source_tier': 'general_web', 'score': 0.7},
                'probabilities': {'entailment': 0.01, 'neutral': 0.24, 'contradiction': 0.75},
                'prediction': 'contradiction'
            },
        ]
        agg = heuristic_nli._aggregate_nli_results(results)
        assert agg['primary_verdict'] == 'neutral'


class TestFusionAndImageForensicsFixes:
    """Verify Fusion templates and calibrated spectral FFT / block ELA / CLIP compatibility."""

    def test_fusion_uses_nli_individual_results_and_avoids_cinema_circular_templates(self):
        engine = FusionEngine()
        evidence = [
            {'source': 'reuters', 'source_tier': 'news_wire', 'text': 'Fact check refutes flat Earth claim.'}
        ]
        nli_res = {
            'claim': 'The Earth is flat.',
            'individual_results': [
                {
                    'evidence': evidence[0],
                    'prediction': 'contradiction',
                    'probabilities': {'entailment': 0.01, 'neutral': 0.02, 'contradiction': 0.97}
                }
            ],
            'aggregated': {
                'primary_verdict': 'contradiction',
                'confidence': 0.97,
                'weighted_probabilities': {'entailment': 0.01, 'neutral': 0.02, 'contradiction': 0.97}
            }
        }
        res = engine.fuse_verdicts(
            nli_res,
            primary_origins=[{'name': 'NASA', 'category': 'space_astronomy'}],
            evidence=evidence,
            claim='The Earth is flat.'
        )
        assert res['final_verdict'] == 'Contradicted'
        assert "poster/figure" not in res['reasoning'].lower()
        assert "Fact check refutes flat Earth claim" in res['truth_analysis']['actual_truth']

    def test_clean_photo_not_misclassified_as_synthetic_ai(self, tmp_path):
        rng = np.random.default_rng(7)
        base = np.zeros((320, 480, 3), dtype=np.uint8)
        for y in range(320):
            base[y, :, 0] = int(50 + 150 * (y / 320))
            base[y, :, 1] = 120
            base[y, :, 2] = 180
        noisy = np.clip(base.astype(np.float32) + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
        clean_path = tmp_path / "clean_photo.jpg"
        cv2.imwrite(str(clean_path), noisy, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = ImageAuthenticityDetector(checkpoint_path="none.pth")
        res = det.assess_authenticity(str(clean_path))
        assert res['spectral_details']['synthetic_risk'] == 'clean'
        assert res['manipulation_type'] == 'authentic_capture'
        assert res['verdict'] == 'authentic'
        assert res['authenticity_score'] > 0.70

    def test_spliced_news_card_detected_by_block_ela(self, tmp_path):
        img = Image.new("RGB", (640, 400), color=(26, 22, 18))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 640, 60], fill=(180, 40, 40))
        draw.text((30, 20), "BREAKING NEWS", fill=(255, 255, 255))
        card_path = tmp_path / "spliced_card.jpg"
        img.save(str(card_path), "JPEG", quality=95)

        arr = cv2.imread(str(card_path))
        patch = np.full((64, 400, 3), (20, 20, 200), dtype=np.uint8)
        cv2.putText(patch, "FAKE SPLICE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        _, enc = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 20])
        arr[320:384, 100:500] = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        cv2.imwrite(str(card_path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = ImageAuthenticityDetector(checkpoint_path="none.pth")
        res = det.assess_authenticity(str(card_path))
        assert res['verdict'] == 'potentially_manipulated'
        assert res['ela_details']['splice_risk'] in ('moderate', 'high')
        assert res['ela_details']['block_spike_ratio'] >= 0.01
        assert res['authenticity_score'] <= 0.25

        # High-risk manipulation must trigger FusionEngine Rule 3 -> Contradicted
        engine = FusionEngine()
        fused = engine.fuse_verdicts(
            {
                'claim': 'Breaking news official banner.',
                'aggregated': {'primary_verdict': 'neutral', 'confidence': 0.98}
            },
            image_result={'authenticity': res, 'context': {'verdict': 'contextually_consistent', 'consistency_score': 0.62}},
            evidence=[],
            claim='Breaking news official banner.'
        )
        assert fused['final_verdict'] == 'Contradicted'
        assert fused['final_confidence'] >= 0.75

    def test_spliced_patch_on_compressed_photo_detected(self, tmp_path):
        rng = np.random.default_rng(12)
        base = np.clip(rng.normal(128, 12, (360, 480, 3)), 0, 255).astype(np.uint8)
        bg_path = tmp_path / "bg_q75.jpg"
        cv2.imwrite(str(bg_path), base, [cv2.IMWRITE_JPEG_QUALITY, 75])

        arr = cv2.imread(str(bg_path))
        patch = np.full((64, 220, 3), (20, 20, 200), dtype=np.uint8)
        cv2.putText(patch, "TAMPERED", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        arr[140:204, 130:350] = patch
        out_path = tmp_path / "spliced_photo.jpg"
        cv2.imwrite(str(out_path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = ImageAuthenticityDetector(checkpoint_path="none.pth")
        res = det.assess_authenticity(str(out_path))
        assert res['verdict'] == 'potentially_manipulated'
        assert res['ela_details']['block_spike_ratio'] >= 0.01

    def test_clip_extract_feature_tensor_supports_transformers_v5_output(self, tmp_path):
        img_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(str(img_path), "JPEG")

        verifier = ImageContextVerifier.__new__(ImageContextVerifier)
        verifier._clip_available = True
        verifier.device = "cpu"
        verifier.attributes = {'location_type': ['outdoor', 'indoor']}
        verifier._cached_attribute_features = {}

        def _fake_processor(**kwargs):
            out = {}
            if 'images' in kwargs:
                out['pixel_values'] = torch.zeros((1, 3, 224, 224))
            if 'text' in kwargs:
                n = len(kwargs['text'])
                out['input_ids'] = torch.ones((n, 4), dtype=torch.long)
                out['attention_mask'] = torch.ones((n, 4), dtype=torch.long)
            holder = MagicMock()
            holder.__getitem__ = lambda self, k: out[k]
            holder.get = lambda k, d=None: out.get(k, d)
            holder.to = lambda dev: holder
            return holder

        mock_clip = MagicMock()
        img_vec = torch.zeros((1, 512))
        img_vec[0, 0] = 1.0
        txt_vec = torch.zeros((1, 512))
        txt_vec[0, 0] = 0.28
        txt_vec[0, 1] = 0.96

        mock_clip.get_image_features.return_value = SimpleNamespace(pooler_output=img_vec)
        mock_clip.get_text_features.return_value = SimpleNamespace(pooler_output=txt_vec)

        verifier.clip_processor = _fake_processor
        verifier.clip_model = mock_clip

        alignment = verifier.compute_image_text_alignment(str(img_path), "Outdoor landscape")
        assert 0.27 <= alignment['similarity_score'] <= 0.29
        assert alignment['alignment'] == 'consistent'
        assert alignment['calibrated_score'] > 0.70
