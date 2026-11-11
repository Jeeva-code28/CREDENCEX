# tests/test_retrieval.py
# Tests for Evidence Retrieval and Ranking modules.

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evidence_ranking import EvidenceRanker


class TestEvidenceRanker:
    """Tests for the EvidenceRanker class."""

    @pytest.fixture
    def mock_semantic_model(self):
        """Create a mock sentence transformer model."""
        model = MagicMock()
        # Return a dummy tensor for encoding
        try:
            import torch
            model.encode.return_value = torch.randn(384)
        except ImportError:
            model.encode.return_value = MagicMock()
        return model

    @pytest.fixture
    def ranker(self, mock_semantic_model):
        """Create an EvidenceRanker with mock model."""
        return EvidenceRanker(semantic_model=mock_semantic_model)

    def test_default_weights(self, ranker):
        """Should use default weights if none specified."""
        assert ranker.weights == {
            'semantic': 0.5,
            'reliability': 0.3,
            'temporal': 0.2
        }

    def test_custom_weights(self, mock_semantic_model):
        """Should accept custom weights."""
        custom_weights = {'semantic': 0.6, 'reliability': 0.2, 'temporal': 0.2}
        ranker = EvidenceRanker(
            semantic_model=mock_semantic_model,
            weights=custom_weights
        )
        assert ranker.weights == custom_weights

    def test_source_reliability_known(self, ranker):
        """Known sources should return their credibility scores."""
        assert ranker.compute_source_reliability('snopes') == 0.95
        assert ranker.compute_source_reliability('reuters') == 0.95
        assert ranker.compute_source_reliability('wikipedia') == 0.80
        assert ranker.compute_source_reliability('web_search') == 0.60

    def test_source_reliability_unknown(self, ranker):
        """Unknown sources should return 0.5."""
        assert ranker.compute_source_reliability('random_blog') == 0.5

    def test_temporal_relevance_same_day(self, ranker):
        """Evidence from the same day as claim should score high."""
        now = datetime.now()
        score = ranker.compute_temporal_relevance(now, now, now)
        assert score >= 0.9

    def test_temporal_relevance_old_evidence(self, ranker):
        """Very old evidence should score lower."""
        now = datetime.now()
        old_date = now - timedelta(days=400)
        score = ranker.compute_temporal_relevance(old_date, now, now)
        assert score < 0.6

    def test_temporal_relevance_missing_dates(self, ranker):
        """Missing dates should return neutral 0.5."""
        score = ranker.compute_temporal_relevance(None, None)
        assert score == 0.5

    def test_rank_evidence_ordering(self, ranker):
        """Evidence should be sorted by score descending."""
        evidence_list = [
            {'text': 'Low relevance text', 'source': 'web_search',
             'retrieved_at': datetime.now()},
            {'text': 'High relevance from snopes', 'source': 'snopes',
             'retrieved_at': datetime.now()},
        ]

        ranked = ranker.rank_evidence('test query', evidence_list)

        assert len(ranked) == 2
        assert all('score' in e for e in ranked)
        assert all('scores_breakdown' in e for e in ranked)
        # Should be sorted descending
        assert ranked[0]['score'] >= ranked[1]['score']

    def test_rank_evidence_empty_list(self, ranker):
        """Empty evidence list should return empty."""
        ranked = ranker.rank_evidence('test', [])
        assert ranked == []

    def test_rank_evidence_preserves_data(self, ranker):
        """Ranking should preserve original evidence data."""
        evidence_list = [
            {'text': 'Test evidence', 'source': 'snopes',
             'url': 'http://example.com', 'retrieved_at': datetime.now()},
        ]

        ranked = ranker.rank_evidence('test', evidence_list)

        assert ranked[0]['text'] == 'Test evidence'
        assert ranked[0]['source'] == 'snopes'
        assert ranked[0]['url'] == 'http://example.com'

    def test_domain_aware_source_reliability(self, ranker):
        """URL domain should boost reliability even for generic web search."""
        score_reuters = ranker.compute_source_reliability('web_search', 'https://www.reuters.com/article/123')
        assert score_reuters == 0.95

        score_politifact = ranker.compute_source_reliability('duckduckgo', 'https://www.politifact.com/factchecks/2024/jan/15/')
        assert score_politifact == 0.95

        score_bbc = ranker.compute_source_reliability('web_search', 'https://www.bbc.com/news/world-123')
        assert score_bbc == 0.90


class TestEvidenceRetriever:
    """Tests for the EvidenceRetriever class (mocked network calls)."""

    @patch('src.evidence_retrieval.requests.get')
    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_snopes_success(self, mock_st, mock_get):
        """Should parse Snopes search results."""
        from src.evidence_retrieval import EvidenceRetriever

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html><body>
            <article>
                <h2>Fact Check Title</h2>
                <p>This is a fact check excerpt.</p>
                <a href="https://snopes.com/fact-check/test">Link</a>
            </article>
        </body></html>
        """
        mock_get.return_value = mock_response

        retriever = EvidenceRetriever()
        results = retriever.retrieve_from_snopes("test query")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['title'] == 'Fact Check Title'
        assert results[0]['source'] == 'snopes'

    @patch('src.evidence_retrieval.requests.get')
    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_snopes_failure(self, mock_st, mock_get):
        """Should handle network errors gracefully."""
        from src.evidence_retrieval import EvidenceRetriever

        mock_get.side_effect = Exception("Network error")

        retriever = EvidenceRetriever()
        results = retriever.retrieve_from_snopes("test query")

        assert results == []

    @patch('src.evidence_retrieval.requests.get')
    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_google_factcheck_success(self, mock_st, mock_get):
        """Should parse Google Fact Check API response with ClaimReview markup."""
        from src.evidence_retrieval import EvidenceRetriever

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'claims': [
                {
                    'text': 'Bill Gates created a new vaccine',
                    'claimant': 'Social media post',
                    'claimReview': [
                        {
                            'publisher': {'name': 'Reuters Fact Check'},
                            'textualRating': 'False',
                            'title': 'Fact Check: Gates vaccine claims are false',
                            'url': 'https://www.reuters.com/fact-check/test'
                        }
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy_key'}):
            retriever = EvidenceRetriever()
            results = retriever.retrieve_from_google_factcheck("Bill Gates vaccine")

            assert len(results) == 1
            assert results[0]['source'] == 'google_factcheck'
            assert results[0]['rating'] == 'False'
            assert results[0]['publisher'] == 'Reuters Fact Check'
            assert "Rating is 'False'" in results[0]['text']

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_google_factcheck_no_key(self, mock_st):
        """Should return empty list when GOOGLE_API_KEY is not set."""
        from src.evidence_retrieval import EvidenceRetriever

        with patch.dict(os.environ, {}, clear=True):
            retriever = EvidenceRetriever()
            results = retriever.retrieve_from_google_factcheck("test")
            assert results == []

    @patch('src.evidence_retrieval.requests.get')
    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_politifact_success(self, mock_st, mock_get):
        """Should parse PolitiFact search results."""
        from src.evidence_retrieval import EvidenceRetriever

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html><body>
            <div class="c-text-teaser">
                <h2><a href="/factchecks/2024/jan/15/bill-gates-test/">Bill Gates Claim</a></h2>
                <p class="c-text-teaser__statement">The statement about the vaccine is completely false.</p>
            </div>
        </body></html>
        """
        mock_get.return_value = mock_response

        retriever = EvidenceRetriever()
        results = retriever.retrieve_from_politifact("Bill Gates")

        assert len(results) == 1
        assert results[0]['source'] == 'politifact'
        assert 'Bill Gates Claim' in results[0]['title']

    @patch('src.evidence_retrieval.requests.get')
    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_factcheck_org_success(self, mock_st, mock_get):
        """Should parse FactCheck.org search results."""
        from src.evidence_retrieval import EvidenceRetriever

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html><body>
            <article>
                <h3><a href="https://www.factcheck.org/2024/01/test-claim/">Vaccine Rumor False</a></h3>
                <div class="entry-content">No evidence supports this vaccine rumor.</div>
            </article>
        </body></html>
        """
        mock_get.return_value = mock_response

        retriever = EvidenceRetriever()
        results = retriever.retrieve_from_factcheck("vaccine rumor")

        assert len(results) == 1
        assert results[0]['source'] == 'factcheck'
        assert results[0]['title'] == 'Vaccine Rumor False'

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_duckduckgo_mocked(self, mock_st):
        """Should parse DuckDuckGo search results."""
        from src.evidence_retrieval import EvidenceRetriever

        with patch('duckduckgo_search.DDGS') as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = [
                {
                    'title': 'Test Fact Check',
                    'body': 'This claim has been debunked.',
                    'href': 'https://example.com/debunk'
                }
            ]
            mock_ddgs_cls.return_value = mock_instance

            retriever = EvidenceRetriever()
            results = retriever.retrieve_from_duckduckgo("test", factcheck_targeted=False)

            assert len(results) == 1
            assert results[0]['source'] == 'duckduckgo'
            assert results[0]['text'] == 'This claim has been debunked.'

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_web_search_fallback_to_duckduckgo(self, mock_st):
        """Should automatically fall back to DuckDuckGo when Google Search credentials are not set."""
        from src.evidence_retrieval import EvidenceRetriever

        with patch.dict(os.environ, {}, clear=True):
            retriever = EvidenceRetriever()
            with patch.object(retriever, 'retrieve_from_duckduckgo', return_value=[{'source': 'duckduckgo', 'text': 'fallback text'}]) as mock_ddg:
                results = retriever.retrieve_from_web_search("test")
                assert len(results) == 1
                assert results[0]['text'] == 'fallback text'
                mock_ddg.assert_called_once()

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_duckduckgo_filters_social_vanity_metrics(self, mock_st):
        """Should filter out social media profiles that only contain follower counts and vanity metrics."""
        from src.evidence_retrieval import EvidenceRetriever

        with patch('duckduckgo_search.DDGS') as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = [
                {
                    'title': 'Instagram Profile',
                    'body': '180K Followers, 7 Following, 2,208 Posts - NTK Official Handle on Instagram',
                    'href': 'https://www.instagram.com/naamtamilarkatchi/'
                },
                {
                    'title': 'Real News Article',
                    'body': 'Naam Tamilar Katchi was founded as a political party and is led by chief coordinator Seeman.',
                    'href': 'https://www.thehindu.com/news/national/tamil-nadu/ntk-party-report/article123.ece'
                }
            ]
            mock_ddgs_cls.return_value = mock_instance

            retriever = EvidenceRetriever()
            results = retriever.retrieve_from_duckduckgo("Naam Tamilar Katchi")

            assert len(results) == 1
            assert results[0]['title'] == 'Real News Article'
            assert 'Instagram' not in results[0]['title']

    @patch('src.evidence_retrieval.SentenceTransformer')
    def test_retrieve_from_wikipedia_success(self, mock_st):
        """Should retrieve encyclopedic summaries from Wikipedia."""
        from src.evidence_retrieval import EvidenceRetriever

        with patch('wikipedia.search', return_value=['Seeman']):
            with patch('wikipedia.summary', return_value='Seeman is an Indian politician and filmmaker who founded Naam Tamilar Katchi.'):
                retriever = EvidenceRetriever()
                results = retriever.retrieve_from_wikipedia("Seeman")

                assert len(results) == 1
                assert results[0]['source'] == 'wikipedia'
                assert 'Seeman is an Indian politician' in results[0]['text']
                assert 'wikipedia.org' in results[0]['url']
