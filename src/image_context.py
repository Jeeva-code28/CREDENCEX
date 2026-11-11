# src/image_context.py
# Phase 5: Image-Claim Context Verification
# Uses CLIP for image-text alignment and visual attribute extraction
# to verify whether an image is contextually consistent with a claim.

from typing import Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class ImageContextVerifier:
    """
    Verifies whether an image is contextually consistent with a claim by:
    1. Computing image-text semantic alignment via CLIP
    2. Extracting visual attributes (location, time period, event type, objects)
    3. Checking temporal and entity consistency
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        text_encoder: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the image context verifier.

        Args:
            clip_model_name: HuggingFace CLIP model variant to use.
            text_encoder_name: Sentence transformer for text encoding.
            text_encoder: Optional pre-loaded SentenceTransformer instance to avoid redundant loading.
        """
        logger.info("Initializing ImageContextVerifier...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if (HAS_TORCH and torch) else 'cpu'

        # Normalize legacy OpenAI names (e.g. ViT-B/32) to HuggingFace identifiers
        hf_model_name = clip_model_name
        if clip_model_name in ["ViT-B/32", "ViT-B-32"]:
            hf_model_name = "openai/clip-vit-base-patch32"
        elif clip_model_name in ["ViT-L/14", "ViT-L-14"]:
            hf_model_name = "openai/clip-vit-large-patch14"

        # Load HuggingFace CLIP model
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(hf_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(hf_model_name)
            self.clip_model.eval()
            self._clip_available = True
            logger.info(f"HuggingFace CLIP model '{hf_model_name}' loaded on {self.device}")
        except Exception as e:
            logger.warning(
                f"HuggingFace CLIP model could not be loaded ({e}). "
                "Image context verification will use fallback scoring."
            )
            self.clip_model = None
            self.clip_processor = None
            self._clip_available = False

        # Text encoder: reuse existing instance if passed, otherwise load once
        if text_encoder is not None:
            self.text_encoder = text_encoder
        elif SentenceTransformer is not None:
            try:
                self.text_encoder = SentenceTransformer(text_encoder_name)
            except Exception:
                self.text_encoder = None
        else:
            self.text_encoder = None

        # Precompute and cache static visual attribute text embeddings
        self.attributes = {
            'location_type': ['urban', 'rural', 'indoor', 'outdoor', 'natural'],
            'time_period': ['ancient', 'medieval', 'modern', 'contemporary', 'future'],
            'event_type': ['protest', 'disaster', 'celebration', 'conflict', 'sports'],
            'objects': ['vehicles', 'people', 'buildings', 'nature', 'technology']
        }
        self._cached_attribute_features = {}

        if self._clip_available and self.clip_model is not None and self.clip_processor is not None:
            inference_ctx = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
            try:
                with inference_ctx():
                    for attr_type, attr_values in self.attributes.items():
                        prompts = [f"a photo showing {val}" for val in attr_values]
                        text_inputs = self.clip_processor(
                            text=prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                        raw_feat = self.clip_model.get_text_features(
                            input_ids=text_inputs['input_ids'],
                            attention_mask=text_inputs.get('attention_mask')
                        )
                        features = self._extract_feature_tensor(raw_feat)
                        self._cached_attribute_features[attr_type] = features / features.norm(p=2, dim=-1, keepdim=True)
            except Exception as e:
                logger.debug(f"Attribute embedding precomputation failed: {e}")

        logger.info("ImageContextVerifier initialized with cached visual attribute embeddings")

    @staticmethod
    def _extract_feature_tensor(model_output):
        """Normalize HuggingFace CLIP feature outputs across transformers 4.x (Tensor) and 5.x (BaseModelOutputWithPooling)."""
        if model_output is None:
            return None
        if HAS_TORCH and torch is not None and isinstance(model_output, torch.Tensor):
            return model_output
        for attr in ('image_embeds', 'text_embeds', 'pooler_output', 'last_hidden_state'):
            if hasattr(model_output, attr):
                val = getattr(model_output, attr)
                if val is not None and (HAS_TORCH and torch is not None and isinstance(val, torch.Tensor)):
                    if attr == 'last_hidden_state' and val.ndim == 3:
                        return val[:, 0, :]
                    return val
        if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
            return model_output[0]
        return model_output

    def compute_image_text_alignment(self, image_path: str, text: str) -> Dict:
        """
        Use CLIP to compute semantic alignment between image and text.

        Args:
            image_path: Path to the image file.
            text: Claim or description text.

        Returns:
            Dict with similarity_score, calibrated_score, and alignment verdict.
        """
        if not self._clip_available:
            return {
                'similarity_score': 0.5,
                'calibrated_score': 0.5,
                'alignment': 'unknown (CLIP not available)'
            }

        image = Image.open(image_path).convert('RGB')
        text_truncated = (text or "")[:300]
        inference_ctx = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad

        with inference_ctx():
            inputs = self.clip_processor(
                text=[text_truncated],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            raw_img_feat = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])
            raw_txt_feat = self.clip_model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask')
            )
            image_features = self._extract_feature_tensor(raw_img_feat)
            text_features = self._extract_feature_tensor(raw_txt_feat)

            # Normalize embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            similarity = float(torch.nn.functional.cosine_similarity(image_features, text_features).item())

        # Calibrate ViT-B/32 cosine band (~0.16 unrelated .. ~0.32 strong match) into [0, 1]
        calibrated_score = max(0.0, min(1.0, (similarity - 0.16) / (0.32 - 0.16)))
        alignment = 'consistent' if similarity > 0.23 else 'mismatch'

        logger.debug(f"Image-text alignment: raw={similarity:.3f}, cal={calibrated_score:.3f} → {alignment}")

        return {
            'similarity_score': float(similarity),
            'calibrated_score': float(calibrated_score),
            'alignment': alignment
        }

    def extract_image_attributes(self, image_path: str) -> Dict:
        """
        Extract visual attributes from the image using CLIP zero-shot classification.

        Categories:
            - location_type: urban, rural, indoor, outdoor, natural
            - time_period: ancient, medieval, modern, contemporary, future
            - event_type: protest, disaster, celebration, conflict, sports
            - objects: vehicles, people, buildings, nature, technology

        Args:
            image_path: Path to the image file.

        Returns:
            Dict of attribute categories, each with predicted value and confidence.
        """
        if not self._clip_available:
            return {
                'location_type': {'predicted': 'unknown', 'confidence': 0.0},
                'time_period': {'predicted': 'unknown', 'confidence': 0.0},
                'event_type': {'predicted': 'unknown', 'confidence': 0.0},
                'objects': {'predicted': 'unknown', 'confidence': 0.0},
            }

        image = Image.open(image_path).convert('RGB')
        inference_ctx = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
        results = {}

        with inference_ctx():
            # Get normalized image embedding once
            img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            raw_img_feat = self.clip_model.get_image_features(pixel_values=img_inputs['pixel_values'])
            image_features = self._extract_feature_tensor(raw_img_feat)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            for attr_type, attr_values in self.attributes.items():
                if attr_type in self._cached_attribute_features:
                    text_features = self._cached_attribute_features[attr_type]
                else:
                    prompts = [f"a photo showing {val}" for val in attr_values]
                    text_inputs = self.clip_processor(
                        text=prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    raw_txt_feat = self.clip_model.get_text_features(
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs.get('attention_mask')
                    )
                    text_features = self._extract_feature_tensor(raw_txt_feat)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                similarities = torch.nn.functional.cosine_similarity(
                    image_features, text_features
                ).cpu().numpy()

                best_idx = int(similarities.argmax())
                best_match = attr_values[best_idx]

                results[attr_type] = {
                    'predicted': best_match,
                    'confidence': float(similarities[best_idx])
                }

        logger.debug(f"Image attributes: {results}")

        return results

    @timer
    def verify_image_context(
        self,
        image_path: str,
        claim_text: str,
        claim_entities: List[str],
        temporal_info: Dict
    ) -> Dict:
        """
        Full image context verification pipeline.

        Combines:
        - Image-text alignment (CLIP similarity)
        - Visual attribute extraction
        - Temporal consistency check
        - Entity consistency check

        Scoring:
            context_score = 0.5 * alignment + 0.3 * temporal + 0.2 * entity

        Args:
            image_path: Path to the image file.
            claim_text: The claim text.
            claim_entities: List of entity strings from claim extraction.
            temporal_info: Temporal dict from claim extraction.

        Returns:
            Dict with alignment, attributes, consistency checks,
            overall_context_score, and verdict.
        """
        logger.info(f"Verifying image context: {image_path}")

        # Image-text alignment
        alignment = self.compute_image_text_alignment(image_path, claim_text)

        # Image attributes
        attributes = self.extract_image_attributes(image_path)

        # Check temporal consistency
        temporal_consistency = self._check_temporal_consistency(
            attributes, temporal_info
        )

        # Check entity consistency
        entity_consistency = self._check_entity_consistency(
            claim_text, claim_entities, attributes
        )

        # Overall context verification using calibrated CLIP score
        sim = float(alignment['similarity_score'])
        cal_sim = float(alignment.get('calibrated_score', max(0.0, min(1.0, (sim - 0.16) / (0.32 - 0.16)))))
        temp_score = float(temporal_consistency['score'])
        ent_score = float(entity_consistency['score'])

        context_score = (
            0.55 * cal_sim +
            0.25 * temp_score +
            0.20 * ent_score
        )

        # Decontextualization check (Real image used with fake/mismatched caption)
        pred_event = attributes.get('event_type', {}).get('predicted', 'general')
        pred_loc = attributes.get('location_type', {}).get('predicted', 'outdoor')
        is_decontextualized = False
        decontext_reason = "Visual semantics align with claim context."

        if sim < 0.22:
            is_decontextualized = True
            decontext_reason = (
                f"Severe cross-modal discrepancy: image depicts '{pred_loc}' / '{pred_event}', "
                "which diverges from the accompanying claim narrative (indicative of recycled or miscaptioned media)."
            )
        elif sim < 0.25 and (temp_score < 0.55 or ent_score < 0.5):
            is_decontextualized = True
            decontext_reason = "Marginal semantic correlation; image may be reused out of original temporal or event context."

        verdict = (
            'contextually_consistent' if (context_score > 0.58 and not is_decontextualized)
            else 'context_mismatch'
        )

        logger.info(f"Context verdict: {verdict} (score={context_score:.3f}, decontextualized={is_decontextualized})")

        return {
            'alignment': alignment,
            'attributes': attributes,
            'temporal_consistency': temporal_consistency,
            'entity_consistency': entity_consistency,
            'overall_context_score': float(context_score),
            'consistency_score': float(context_score),
            'is_decontextualized': is_decontextualized,
            'decontextualization_reason': decontext_reason,
            'verdict': verdict
        }

    def _check_temporal_consistency(
        self,
        attributes: Dict,
        temporal_info: Dict
    ) -> Dict:
        """
        Verify temporal consistency between image visual appearance
        and claimed time period.

        Args:
            attributes: Visual attributes from CLIP analysis.
            temporal_info: Temporal info from claim extraction.

        Returns:
            Dict with score and human-readable details.
        """
        image_time_period = attributes.get('time_period', {}).get('predicted', '')
        claim_dates = temporal_info.get('mentioned_dates', [])

        # Heuristic: contemporary setting + recent claim = consistent
        if image_time_period == 'contemporary' and claim_dates:
            score = 0.7
        elif image_time_period in ('modern', 'contemporary'):
            score = 0.6
        else:
            score = 0.5

        return {
            'score': score,
            'details': f"Image appears {image_time_period}"
        }

    def _check_entity_consistency(
        self,
        claim_text: str,
        entities: List[str],
        attributes: Dict
    ) -> Dict:
        """
        Verify if image visual content aligns with entities mentioned
        in the claim.

        Args:
            claim_text: The claim text.
            entities: Entity strings from claim extraction.
            attributes: Visual attributes from CLIP analysis.

        Returns:
            Dict with score and human-readable details.
        """
        people_in_claim = bool(entities) and any(
            ent.lower() in claim_text.lower() for ent in entities
        )
        objects_in_image = attributes.get('objects', {}).get('predicted', '')
        loc_in_image = attributes.get('location_type', {}).get('predicted', '')
        claim_lower = (claim_text or '').lower()

        natural_terms = {'mountain', 'ocean', 'sea', 'forest', 'volcano', 'river', 'flood', 'sky', 'nature', 'landscape'}
        doc_terms = {'circular', 'notification', 'notice', 'order', 'government', 'department', 'ministry', 'official', 'document'}

        if 'people' in objects_in_image and people_in_claim:
            score = 0.8
        elif objects_in_image == 'nature' and any(t in claim_lower for t in natural_terms):
            score = 0.75
        elif any(t in claim_lower for t in doc_terms) and loc_in_image == 'indoor':
            score = 0.7
        elif objects_in_image:
            score = 0.55
        else:
            score = 0.45

        return {
            'score': score,
            'details': f"Image contains {objects_in_image}"
        }
