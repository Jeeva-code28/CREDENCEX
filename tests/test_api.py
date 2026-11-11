# tests/test_api.py
# Unit tests for the FastAPI service, cache, and endpoints.

import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.cache import ClaimCache, normalize_claim_key


class TestClaimCache:
    """Tests for the thread-safe LRU/TTL claim cache."""

    def test_normalize_claim_key(self):
        """Should lowercase, strip punctuation, and normalize whitespace."""
        raw = "  Bill Gates Announced a Vaccine!! On January 15, 2024?  "
        norm = normalize_claim_key(raw)
        assert norm == "bill gates announced a vaccine on january 15 2024"

    def test_cache_set_and_get(self):
        """Should store and retrieve values."""
        cache = ClaimCache(max_size=10, ttl_seconds=60)
        cache.set("The Earth is flat", {"verdict": "Contradicted"})

        res = cache.get("The Earth is flat")
        assert res == {"verdict": "Contradicted"}

        # Normalized query should also hit cache
        res_norm = cache.get("the earth is flat!")
        assert res_norm == {"verdict": "Contradicted"}

    def test_cache_miss(self):
        """Should return None on missing key."""
        cache = ClaimCache()
        assert cache.get("unknown claim") is None

    def test_cache_expiration(self):
        """Should return None when TTL expires."""
        cache = ClaimCache(max_size=10, ttl_seconds=0.01)
        cache.set("test", {"verdict": "Unverified"})
        time.sleep(0.02)
        assert cache.get("test") is None

    def test_cache_lru_eviction(self):
        """Should evict oldest entries when max_size is reached."""
        cache = ClaimCache(max_size=2, ttl_seconds=60)
        cache.set("claim 1", {"val": 1})
        cache.set("claim 2", {"val": 2})
        cache.set("claim 3", {"val": 3})

        assert cache.size() == 2
        assert cache.get("claim 1") is None
        assert cache.get("claim 2") is not None
        assert cache.get("claim 3") is not None

    def test_cache_clear(self):
        """Should remove all entries."""
        cache = ClaimCache()
        cache.set("c1", {"v": 1})
        cache.set("c2", {"v": 2})
        assert cache.clear() == 2
        assert cache.size() == 0


@pytest.fixture
def api_client():
    """Create a TestClient with a lightweight mock pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.evidence_ranker.source_credibility = {
        'google_factcheck': 0.98,
        'snopes': 0.95,
        'reuters': 0.95,
        'ap_news': 0.95,
        'duckduckgo': 0.65
    }

    mock_claim = MagicMock()
    mock_claim.original_text = "Test claim text"

    mock_pipeline.verify.return_value = {
        'claim': mock_claim,
        'evidence': [
            {
                'source': 'snopes',
                'title': 'Fact Check',
                'text': 'Debunking excerpt',
                'url': 'https://snopes.com/test',
                'score': 0.88,
                'scores_breakdown': {'semantic': 0.5, 'reliability': 0.3, 'temporal': 0.08}
            }
        ],
        'final_verdict': {
            'final_verdict': 'Contradicted',
            'final_confidence': 0.88,
            'text_verdict': 'contradiction',
            'text_confidence': 0.88,
            'image_verdict': None,
            'image_confidence': 0.0
        },
        'report': "Test Report Summary"
    }

    app.state.pipeline = mock_pipeline
    app.state.cache = ClaimCache(max_size=100, ttl_seconds=3600)
    app.state.start_time = time.time()
    app.state.jobs = {}

    client = TestClient(app)
    return client


class TestAPIEndpoints:
    """Tests for FastAPI HTTP endpoints."""

    def test_root_endpoint(self, api_client):
        """GET /?format=json or with Accept: application/json should return service info and links."""
        response = api_client.get("/", headers={"Accept": "application/json"})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'online'
        assert "/docs" in data['documentation']

    def test_health_check_endpoint(self, api_client):
        """GET /api/v1/health should return healthy status."""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'models_loaded' in data
        assert 'uptime_seconds' in data

    def test_get_sources_endpoint(self, api_client):
        """GET /api/v1/sources should return list of sources with credibility scores."""
        response = api_client.get("/api/v1/sources")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 4
        source_names = [s['source'] for s in data]
        assert 'google_factcheck' in source_names
        assert 'snopes' in source_names

    def test_verify_sync_json(self, api_client):
        """POST /api/v1/verify should return synchronous verdict."""
        payload = {"text": "Test claim text"}
        response = api_client.post("/api/v1/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data['final_verdict'] == 'Contradicted'
        assert data['final_confidence'] == 0.88
        assert data['cached'] is False
        assert data['evidence_count'] == 1
        assert 'truth_analysis' in data
        assert data['truth_analysis'] is not None
        assert data['truth_analysis']['status'] in ('rumor', 'verified_truth', 'unverified_rumor')
        assert 'actual_truth' in data['truth_analysis']
        assert 'rumor_statement' in data['truth_analysis']

    def test_verify_truth_analysis_present(self, api_client):
        """API should return structured truth_analysis comparing rumor vs actual truth."""
        payload = {"text": "All ration cards will be cancelled"}
        response = api_client.post("/api/v1/verify", json=payload)
        assert response.status_code == 200
        ta = response.json().get('truth_analysis')
        assert ta is not None
        assert ta['status'] == 'rumor'
        assert 'badge_label' in ta
        assert 'DEBUNKED RUMOR' in ta['badge_label']
        assert 'actual_truth' in ta
        assert 'rumor_statement' in ta
        assert 'evidence_basis' in ta
        assert 'verdict_meaning' in ta

    def test_verify_cache_hit(self, api_client):
        """Second call with same claim should return cached result."""
        payload = {"text": "A viral rumor to cache"}
        res1 = api_client.post("/api/v1/verify", json=payload)
        assert res1.status_code == 200
        assert res1.json()['cached'] is False

        # Repeat identical claim
        res2 = api_client.post("/api/v1/verify", json=payload)
        assert res2.status_code == 200
        assert res2.json()['cached'] is True

    def test_verify_missing_input_400(self, api_client):
        """POST /api/v1/verify with empty payload should return 400."""
        response = api_client.post("/api/v1/verify", json={})
        assert response.status_code == 400
        assert "At least one input required" in response.json()['detail']

    def test_verify_async_job_and_poll(self, api_client):
        """POST /api/v1/verify/async should return 202 and pollable job_id."""
        payload = {"text": "Async test claim"}
        response = api_client.post("/api/v1/verify/async", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert 'job_id' in data
        assert data['status'] in ('queued', 'processing', 'completed')

        # Poll job
        job_id = data['job_id']
        poll_res = api_client.get(f"/api/v1/jobs/{job_id}")
        assert poll_res.status_code == 200
        assert poll_res.json()['job_id'] == job_id

    def test_job_not_found_404(self, api_client):
        """GET /api/v1/jobs/invalid-id should return 404."""
        response = api_client.get("/api/v1/jobs/non-existent-uuid")
        assert response.status_code == 404

    def test_clear_cache_endpoint(self, api_client):
        """DELETE /api/v1/cache should clear cache."""
        api_client.post("/api/v1/verify", json={"text": "To be cleared"})
        assert app.state.cache.size() >= 1

        del_res = api_client.delete("/api/v1/cache")
        assert del_res.status_code == 200
        assert app.state.cache.size() == 0

    def test_serve_web_dashboard(self, api_client):
        """GET / (default) and GET /app should serve the interactive web dashboard HTML."""
        res_root = api_client.get("/")
        assert res_root.status_code == 200
        assert "CREDENCEX" in res_root.text
        assert "Trust the evidence. Verify the truth." in res_root.text
        assert "Multimodal AI-Powered News Verification" in res_root.text

        res_app = api_client.get("/app")
        assert res_app.status_code == 200
        assert "CREDENCEX" in res_app.text

    def test_serve_static_assets(self, api_client):
        """GET /static/styles.css and /static/app.js should return static web assets."""
        res_css = api_client.get("/static/styles.css")
        assert res_css.status_code == 200
        assert "--accent-primary" in res_css.text

        res_js = api_client.get("/static/app.js")
        assert res_js.status_code == 200
        assert "checkHealth" in res_js.text

