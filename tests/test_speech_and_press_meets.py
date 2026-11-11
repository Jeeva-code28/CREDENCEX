import pytest
from src.claim_extraction import ClaimExtractor, generate_queries, Claim
from src.authority_sources import UniversalOriginResolver, get_domain_authority_score
from src.evidence_ranking import EvidenceRanker
from src.fusion_engine import FusionEngine


class TestSpeechAndPressMeetVerification:
    """
    Test suite verifying the real-world journalistic paradigm:
    Celebrities, politicians, leaders, and public figures give press meets,
    interviews, rallies, and speeches that are documented live by TV news channels,
    accredited newspapers, and video proofs — NOT posted on personal social media handles.
    """

    @pytest.fixture
    def extractor(self):
        return ClaimExtractor(ner_model="en_core_web_sm")

    @pytest.fixture
    def ranker(self):
        return EvidenceRanker()

    @pytest.fixture
    def fusion_engine(self):
        return FusionEngine()

    def test_speech_claim_detection(self, extractor):
        """Verify speeches, press meets, and public addresses are correctly detected."""
        speech_claims = [
            "photo shows vijay's recent palimentry speech",
            "Vijay gave a fiery speech at the TVK rally in Vikravandi",
            "Chief Minister held a press meet regarding flood relief",
            "Actress spoke at the press conference about her upcoming project",
            "Kamal Haasan addressed the public rally yesterday",
            "Prime Minister stated during parliamentary address",
            "தலைவர் செய்தியாளர் சந்திப்பில் பேசியது"
        ]

        for claim_text in speech_claims:
            claim = extractor.extract_claim(claim_text)
            assert claim.is_speech_claim is True, f"Failed to detect speech claim for: {claim_text}"

    def test_non_speech_claim_detection(self, extractor):
        """Verify institutional, scientific, or government circulars are NOT marked as speech claims."""
        non_speech_claims = [
            "RBI announced revision in repo rate by 25 basis points",
            "ISRO successfully launched Chandrayaan-3 lunar rover",
            "WHO declared new medical health guidelines for booster vaccines",
            "Chennai District Collector declared school holiday due to heavy rain"
        ]

        for claim_text in non_speech_claims:
            claim = extractor.extract_claim(claim_text)
            assert claim.is_speech_claim is False, f"Erroneously detected speech claim for: {claim_text}"

    def test_speech_queries_prioritize_news_and_avoid_social_handles(self, extractor):
        """Verify queries for speeches target accredited TV channels & newspapers, NOT personal Twitter handles."""
        claim_text = "photo shows vijay's recent palimentry speech"
        claim = extractor.extract_claim(claim_text)
        assert claim.is_speech_claim is True

        queries = generate_queries(claim)
        assert len(queries) > 0

        # Must not contain bare personal handle queries like site:x.com/actorvijay
        for q in queries:
            assert "site:x.com/actorvijay" not in q
            assert "site:twitter.com/actorvijay" not in q

        # Must contain accredited news media queries
        queries_joined = " ".join(queries)
        assert any(domain in queries_joined for domain in ["thehindu.com", "puthiyathalaimurai.com", "thanthitv.com", "dinamalar.com", "indianexpress.com", "speech news"])

    def test_unrelated_nic_domain_not_scored_as_primary_origin_for_celebrity_speech(self):
        """Verify unrelated government exam domains (e.g. cuet.nta.nic.in) do NOT receive 0.99 credibility for actor speech."""
        actor_origin = [{
            'key': 'actor_vijay',
            'name': 'Thalapathy Vijay',
            'role': 'Lead Actor & Leader',
            'category': 'tvk_party',
            'official_handles': ['@actorvijay', '@tvkvijayhq'],
            'official_domains': ['tvk.family']
        }]

        score = get_domain_authority_score(
            "https://cuet.nta.nic.in/score-card-2026",
            primary_entities=actor_origin
        )
        # Should be downgraded to standard web score (0.60), NOT 0.99
        assert score == 0.60, f"Unrelated .nic.in domain received inappropriate authority score: {score}"

    def test_semantic_relevance_gate_suppresses_off_topic_evidence(self, ranker):
        """Verify that evidence with near-zero semantic similarity is penalized even if prestige domain."""
        query = "photo shows vijay's recent palimentry speech"
        evidence_list = [
            {
                'text': 'Jun 23, 2026 - Score Card for CUET (UG) - 2026 Publish Date: June 23, 2026 Score Card for CUET (UG)',
                'url': 'https://cuet.nta.nic.in/scorecard',
                'source': 'web_search',
                'retrieved_at': None
            },
            {
                'text': 'Thalapathy Vijay addressed party cadre at Chennai meeting discussing political roadmap and public policies',
                'url': 'https://www.thehindu.com/news/national/tamil-nadu/vijay-address/article.ece',
                'source': 'web_search',
                'retrieved_at': None
            }
        ]

        ranked = ranker.rank_evidence(query, evidence_list)
        assert len(ranked) == 2
        # The relevant news story must rank higher than the off-topic exam portal
        assert "thehindu.com" in ranked[0]['url']
        assert ranked[1]['score'] < ranked[0]['score']

    def test_truth_analysis_speech_synthesis(self, fusion_engine):
        """Verify that truth analysis for speeches uses accredited news & video broadcast phrasing rather than circulars."""
        truth_analysis = fusion_engine.generate_truth_analysis(
            claim="photo shows vijay's recent palimentry speech",
            final_verdict="Contradicted",
            final_confidence=0.88,
            primary_origins=[{'name': 'Thalapathy Vijay'}],
            is_speech_claim=True
        )

        assert truth_analysis['status'] == 'rumor'
        # Must NOT mention bureaucratic circulars or policies
        assert "circular" not in truth_analysis['actual_truth'].lower()
        assert "policy" not in truth_analysis['actual_truth'].lower()
        # Must mention accredited news channels, newspapers, or live broadcasts
        assert any(kw in truth_analysis['actual_truth'].lower() for kw in ['news channels', 'broadcast', 'newspaper', 'speech', 'video'])
        assert any(kw in truth_analysis['evidence_basis'].lower() for kw in ['tv news', 'broadcast', 'newspapers', 'video proof'])
