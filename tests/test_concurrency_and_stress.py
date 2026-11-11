# tests/test_concurrency_and_stress.py
# Concurrency, multi-threading, LRU cache stress, and async worker lifecycle tests.

import time
import concurrent.futures
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.cache import ClaimCache, normalize_claim_key


@pytest.fixture
def thread_safe_client():
    """Create a thread-safe TestClient for concurrent verification testing."""
    mock_pipeline = MagicMock()
    mock_pipeline.evidence_ranker.source_credibility = {'reuters': 0.95}

    def _mock_verify(input_data, input_type='text', claim_text=None, **kwargs):
        mock_claim = MagicMock()
        mock_claim.original_text = input_data
        mock_claim.entities = ['TestEntity']
        mock_claim.domain = 'general'
        return {
            'claim': mock_claim,
            'evidence': [
                {
                    'source': 'reuters',
                    'title': 'Wire Report',
                    'text': 'Fact check verification text.',
                    'url': 'https://reuters.com/news',
                    'score': 0.85,
                    'scores_breakdown': {'reliability': 0.95}
                }
            ],
            'final_verdict': {
                'final_verdict': 'Contradicted',
                'final_confidence': 0.88,
                'text_verdict': 'contradiction',
                'text_confidence': 0.88,
                'truth_analysis': {
                    'status': 'rumor',
                    'badge_label': 'DEBUNKED RUMOR / FALSE',
                    'actual_truth': 'Reuters fact-check confirms claim is false.',
                    'rumor_statement': input_data,
                    'evidence_basis': 'Refuted by wire reports.',
                    'verdict_meaning': 'This is a rumor and NOT actual truth.'
                }
            },
            'report': "Automated Test Report"
        }

    mock_pipeline.verify.side_effect = _mock_verify

    app.state.pipeline = mock_pipeline
    app.state.cache = ClaimCache(max_size=200, ttl_seconds=3600)
    app.state.jobs = {}
    app.state.start_time = time.time()

    return TestClient(app)


class TestConcurrentAPIRequests:
    """Stress tests simulating concurrent incoming requests to the verification endpoint."""

    def test_concurrent_text_verifications(self, thread_safe_client):
        """15 concurrent threads should verify claims without deadlock or data race."""
        claims = [f"Viral assertion claim number {i} circulating on social media" for i in range(15)]

        def _send_request(claim_text):
            response = thread_safe_client.post("/api/v1/verify", json={"text": claim_text})
            return response.status_code, response.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_send_request, c) for c in claims]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 15
        for status_code, data in results:
            assert status_code == 200
            assert data['final_verdict'] == 'Contradicted'
            assert 'truth_analysis' in data
            assert data['truth_analysis']['status'] == 'rumor'

    def test_concurrent_cache_reads_and_writes(self):
        """High-frequency concurrent reads/writes on ClaimCache should remain thread-safe."""
        cache = ClaimCache(max_size=50, ttl_seconds=60)

        def _worker(thread_id):
            for i in range(25):
                key = f"claim_variant_{i % 5}"
                cache.set(key, {"id": f"{thread_id}_{i}"})
                res = cache.get(key)
                assert res is not None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_worker, tid) for tid in range(10)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert cache.size() <= 50


class TestCacheResilienceAndLifecycle:
    """Detailed lifecycle tests for caching, normalization, and LRU eviction."""

    def test_cache_key_normalization_invariance(self, thread_safe_client):
        """Claims with varying casing, whitespace, and punctuation should hit cache."""
        payload1 = {"text": "All Tamil Nadu ration cards will be cancelled!"}
        res1 = thread_safe_client.post("/api/v1/verify", json=payload1)
        assert res1.status_code == 200
        assert res1.json()['cached'] is False

        # Exact normalized match
        payload2 = {"text": "  ALL TAMIL NADU RATION CARDS WILL BE CANCELLED...  "}
        res2 = thread_safe_client.post("/api/v1/verify", json=payload2)
        assert res2.status_code == 200
        assert res2.json()['cached'] is True

    def test_bypass_cache_flag(self, thread_safe_client):
        """bypass_cache=True must ignore cached result and execute fresh verification."""
        payload = {"text": "A viral claim to test cache bypass"}
        res1 = thread_safe_client.post("/api/v1/verify", json=payload)
        assert res1.status_code == 200
        assert res1.json()['cached'] is False

        # Bypass cache
        res2 = thread_safe_client.post(
            "/api/v1/verify",
            json={"text": "A viral claim to test cache bypass", "bypass_cache": True}
        )
        assert res2.status_code == 200
        assert res2.json()['cached'] is False

    def test_lru_eviction_under_load(self):
        """Cache should strictly enforce max_size and evict least-recently-used entry."""
        cache = ClaimCache(max_size=3, ttl_seconds=300)
        cache.set("A", {"v": 1})
        cache.set("B", {"v": 2})
        cache.set("C", {"v": 3})

        # Access A so B becomes oldest
        _ = cache.get("A")

        # Insert D -> B must be evicted
        cache.set("D", {"v": 4})

        assert cache.size() == 3
        assert cache.get("B") is None
        assert cache.get("A") is not None
        assert cache.get("C") is not None
        assert cache.get("D") is not None

    def test_cache_purge_api_endpoint(self, thread_safe_client):
        """DELETE /api/v1/cache should clear all stored entries."""
        thread_safe_client.post("/api/v1/verify", json={"text": "Item to cache before purge"})
        assert app.state.cache.size() >= 1

        del_res = thread_safe_client.delete("/api/v1/cache")
        assert del_res.status_code == 200
        assert del_res.json()['status'] == 'cleared'
        assert app.state.cache.size() == 0


class TestAsyncJobLifecycle:
    """Tests for the asynchronous verification job queue and polling mechanism."""

    def test_async_job_submission_and_completion(self, thread_safe_client):
        """Submitting with async_mode=True returns job_id which eventually completes."""
        payload = {"text": "Async background verification job claim", "async_mode": True}
        post_res = thread_safe_client.post("/api/v1/verify", json=payload)

        assert post_res.status_code == 202
        body = post_res.json()
        assert "job_id" in body
        assert body["status"] == "queued"

        job_id = body["job_id"]

        # Poll job status
        max_attempts = 20
        completed = False
        final_job_data = None

        for _ in range(max_attempts):
            time.sleep(0.05)
            poll_res = thread_safe_client.get(f"/api/v1/jobs/{job_id}")
            assert poll_res.status_code == 200
            data = poll_res.json()
            if data["status"] == "completed":
                completed = True
                final_job_data = data
                break

        assert completed is True
        assert final_job_data is not None
        assert final_job_data["result"] is not None
        assert final_job_data["result"]["final_verdict"] == "Contradicted"
        assert "truth_analysis" in final_job_data["result"]
