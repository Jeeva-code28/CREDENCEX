# src/evidence_ranking.py
# Phase 3: Evidence Ranking
# Ranks retrieved evidence using composite scoring:
#   Se = 0.5 * Ssem + 0.3 * Srel + 0.2 * Stemp

from datetime import datetime
from typing import List, Dict, Optional

import re
import numpy as np
try:
    from sentence_transformers.util import pytorch_cos_sim
    HAS_COS_SIM = True
except ImportError:
    pytorch_cos_sim = None
    HAS_COS_SIM = False

from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class EvidenceRanker:
    """
    Ranks evidence using a weighted combination of:
    - Semantic similarity (cosine similarity via sentence-transformers)
    - Source reliability (pre-defined credibility scores)
    - Temporal relevance (proximity to claim date + recency)
    """

    def __init__(self, semantic_model=None, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the evidence ranker.

        Args:
            semantic_model: A SentenceTransformer model instance
                            (shared with EvidenceRetriever).
            weights: Dict with keys 'semantic', 'reliability', 'temporal'.
                     Defaults to {0.5, 0.3, 0.2}.
        """
        if semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            except (ImportError, Exception):
                self.semantic_model = None
        else:
            self.semantic_model = semantic_model

        # Default weights: Se = 0.5*Ssem + 0.3*Srel + 0.2*Stemp
        self.weights = weights or {
            'semantic': 0.5,
            'reliability': 0.3,
            'temporal': 0.2
        }

        # Source credibility scores (can be learned from training data)
        self.source_credibility = {
            'google_factcheck': 0.98,
            'snopes': 0.95,
            'factcheck': 0.95,
            'politifact': 0.95,
            'reuters': 0.95,
            'ap_news': 0.95,
            'duckduckgo_factcheck': 0.90,
            'wikipedia': 0.80,  # Encyclopedic reference context
            'duckduckgo': 0.65,
            'web_search': 0.60,
        }

        logger.info(f"EvidenceRanker initialized with weights: {self.weights}")

    def compute_semantic_similarity(self, query: str, evidence_text: str) -> float:
        """
        Compute cosine similarity between query and evidence embeddings,
        or token-overlap similarity if semantic embedding model is not present.

        Args:
            query: The search query or claim text.
            evidence_text: The evidence passage text.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not query or not evidence_text:
            return 0.0

        if self.semantic_model is not None and pytorch_cos_sim is not None:
            try:
                q_embed = self.semantic_model.encode(query, convert_to_tensor=True)
                e_embed = self.semantic_model.encode(evidence_text, convert_to_tensor=True)
                sim = pytorch_cos_sim(q_embed, e_embed).item()
                return max(0.0, min(1.0, sim))
            except Exception:
                pass

        # Fallback: token overlap & Jaccard similarity
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        e_tokens = set(re.findall(r'\b\w+\b', evidence_text.lower()))
        if not q_tokens or not e_tokens:
            return 0.0
        overlap = len(q_tokens & e_tokens)
        jaccard = overlap / len(q_tokens | e_tokens)
        coverage = overlap / len(q_tokens)
        sim = min(1.0, 0.7 * coverage + 0.3 * jaccard)
        return sim

    def compute_source_reliability(
        self,
        source: str,
        url: Optional[str] = None,
        primary_origins: Optional[List[Dict]] = None
    ) -> float:
        """
        Retrieve source credibility score with domain-aware authority evaluation
        and primary entity origin awareness.
        """
        if url:
            if 'wikipedia.org' in url.lower():
                return 0.80
            try:
                from src.authority_sources import get_domain_authority_score
                domain_score = get_domain_authority_score(
                    url,
                    source_type=source,
                    primary_entities=primary_origins
                )
                if domain_score > 0.60 or domain_score <= 0.50:
                    return domain_score
            except Exception:
                pass

        if source in self.source_credibility:
            return self.source_credibility[source]

        return 0.5

    def compute_temporal_relevance(
        self,
        evidence_date: Optional[datetime],
        claim_date: Optional[datetime],
        current_date: Optional[datetime] = None
    ) -> float:
        """
        Compute temporal relevance based on:
        - Proximity of evidence date to claim date
        - Recency of the evidence

        Args:
            evidence_date: When the evidence was published/retrieved.
            claim_date: When the claim event occurred.
            current_date: Reference date (defaults to now).

        Returns:
            Temporal relevance score in [0, 1].
        """
        if current_date is None:
            current_date = datetime.now()

        if not evidence_date or not claim_date:
            return 0.5  # Neutral score if dates unavailable

        try:
            time_diff = abs((evidence_date - claim_date).days)
            time_since_evidence = (current_date - evidence_date).days

            # Penalize very old evidence
            if time_since_evidence > 365:
                recency_score = 0.3
            elif time_since_evidence > 180:
                recency_score = 0.6
            else:
                recency_score = 0.9

            # Proximity to claim date
            if time_diff == 0:
                proximity_score = 1.0
            elif time_diff <= 7:
                proximity_score = 0.9
            elif time_diff <= 30:
                proximity_score = 0.7
            else:
                proximity_score = 0.4

            return (recency_score + proximity_score) / 2

        except Exception:
            return 0.5

    @timer
    def rank_evidence(
        self,
        query: str,
        evidence_list: List[Dict],
        claim_date: Optional[datetime] = None,
        primary_origins: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Rank evidence using the composite scoring formula:
            Se = w_sem * Ssem + w_rel * Srel + w_temp * Stemp

        Optimized with vectorized batch semantic encoding:
        encodes the query once and batch-encodes all evidence texts in a single pass.

        Args:
            query: The claim or combined query text.
            evidence_list: List of evidence dicts from retrieval.
            claim_date: Date of the claim event (defaults to now).
            primary_origins: Optional list of identified primary origin entities.

        Returns:
            Evidence list sorted by score (descending), each augmented
            with 'score' and 'scores_breakdown' keys.
        """
        if not evidence_list:
            return []

        if claim_date is None:
            claim_date = datetime.now()

        # Batch compute semantic similarity across all evidence passages
        s_sem_scores = [0.0] * len(evidence_list)
        valid_indices = []
        texts_to_encode = []

        for i, ev in enumerate(evidence_list):
            txt = ev.get('text', '')
            if txt and txt.strip():
                valid_indices.append(i)
                texts_to_encode.append(txt.strip())

        if texts_to_encode and query and query.strip():
            try:
                # 1. Encode query once
                query_emb = self.semantic_model.encode(query, convert_to_tensor=True)
                if hasattr(query_emb, 'ndim') and query_emb.ndim == 1:
                    query_emb = query_emb.unsqueeze(0)

                # 2. Batch encode all evidence passages
                ev_embs = self.semantic_model.encode(
                    texts_to_encode,
                    batch_size=32,
                    convert_to_tensor=True
                )
                if hasattr(ev_embs, 'ndim') and ev_embs.ndim == 1:
                    ev_embs = ev_embs.unsqueeze(0)

                # 3. Vectorized cosine similarity
                cos_sims = pytorch_cos_sim(query_emb, ev_embs)
                raw_sims = cos_sims[0].cpu().numpy() if hasattr(cos_sims, 'cpu') else cos_sims[0]

                if np.isscalar(raw_sims):
                    raw_sims = [float(raw_sims)]
                elif hasattr(raw_sims, 'tolist'):
                    raw_sims = raw_sims.tolist()

                for idx, sim_val in zip(valid_indices, raw_sims):
                    s_sem_scores[idx] = float(max(0.0, float(sim_val)))

            except Exception as e:
                logger.warning(f"Batch semantic encoding failed, falling back to individual scoring: {e}")
                for i in valid_indices:
                    s_sem_scores[i] = self.compute_semantic_similarity(
                        query, evidence_list[i].get('text', '')
                    )

        scored_evidence = []

        for i, evidence in enumerate(evidence_list):
            s_sem = s_sem_scores[i]
            s_rel = self.compute_source_reliability(
                evidence.get('source', ''),
                evidence.get('url'),
                primary_origins=primary_origins
            )
            s_temp = self.compute_temporal_relevance(
                evidence.get('retrieved_at'),
                claim_date
            )

            # Weighted combination
            final_score = (
                self.weights['semantic'] * s_sem +
                self.weights['reliability'] * s_rel +
                self.weights['temporal'] * s_temp
            )

            # Semantic relevance gate:
            # If an evidence snippet has very low semantic relevance (< 0.18), it is completely off-topic
            # (e.g. CUET entrance exams or unrelated film teasers for a speech/news claim).
            # Heavily suppress it so it can never outrank on-topic news coverage.
            if s_sem < 0.18:
                final_score *= 0.20

            evidence_with_score = evidence.copy()
            evidence_with_score['score'] = final_score
            evidence_with_score['scores_breakdown'] = {
                'semantic': s_sem,
                'reliability': s_rel,
                'reliability_score': s_rel,
                'temporal': s_temp
            }

            scored_evidence.append(evidence_with_score)

            logger.debug(
                f"Evidence [{i}] score={final_score:.3f} "
                f"(sem={s_sem:.3f}, rel={s_rel:.3f}, temp={s_temp:.3f})"
            )

        # Sort by score (descending)
        scored_evidence.sort(key=lambda x: x['score'], reverse=True)

        # Apply source diversity cap: max 1 encyclopedic record in top positions
        diversified_evidence = []
        deferred_wiki = []
        wiki_count = 0

        for ev in scored_evidence:
            is_wiki = ev.get('source') == 'wikipedia' or 'wikipedia.org' in (ev.get('url') or '').lower()
            if is_wiki:
                if wiki_count < 1:
                    diversified_evidence.append(ev)
                    wiki_count += 1
                else:
                    deferred_wiki.append(ev)
            else:
                diversified_evidence.append(ev)

        diversified_evidence.extend(deferred_wiki)

        logger.info(
            f"Ranked {len(diversified_evidence)} evidence pieces. "
            f"Top score: {diversified_evidence[0]['score']:.3f}" if diversified_evidence else
            "No evidence to rank."
        )

        return diversified_evidence
