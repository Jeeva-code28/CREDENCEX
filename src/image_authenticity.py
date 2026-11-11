# src/image_authenticity.py
# Phase 5: Image Authenticity Detection
# Detects image manipulation using Error Level Analysis (ELA),
# spectral analysis, and CNN-based prediction.

import io
import base64
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import cv2
import numpy as np
from PIL import Image, ExifTags
try:
    import torch
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    torch = None
    models = None
    transforms = None
    HAS_TORCH = False

from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class ImageAuthenticityDetector:
    """
    Detects whether an image has been manipulated using three methods:
    1. Error Level Analysis (ELA) — detects JPEG compression inconsistencies
    2. Spectral/FFT Analysis — identifies splicing or copy-move forgery
    3. CNN Prediction — ResNet50-based binary classification (authentic/manipulated)
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the image authenticity detector.

        Args:
            model_name: CNN architecture or HuggingFace model for manipulation detection.
            pretrained: Whether to use ImageNet pretrained backbone weights.
            checkpoint_path: Optional path to fine-tuned manipulation weights.
        """
        logger.info(f"Initializing ImageAuthenticityDetector with {model_name}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if (HAS_TORCH and torch) else 'cpu'
        self.model = None
        self.cnn_available = False
        self.checkpoint_path = checkpoint_path

        # Check default checkpoint location if none specified
        default_ckpt = Path("models/image_classifiers/resnet50_manipulation.pth")
        active_ckpt = checkpoint_path or (str(default_ckpt) if default_ckpt.exists() else None)

        if HAS_TORCH and models is not None and active_ckpt and Path(active_ckpt).exists():
            try:
                if model_name == 'resnet50':
                    self.model = models.resnet50(
                        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                    )
                    self.model.fc = torch.nn.Linear(2048, 2)
                    state_dict = torch.load(active_ckpt, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()
                    self.cnn_available = True
                    logger.info(f"Loaded fine-tuned manipulation checkpoint: {active_ckpt}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {active_ckpt}: {e}")
                self.cnn_available = False
        else:
            logger.info(
                "No fine-tuned manipulation checkpoint found or PyTorch offline. "
                "CNN manipulation scoring disabled to avoid random predictions. "
                "Authenticity scoring will use deterministic ELA and Spectral FFT analysis."
            )

        # Image preprocessing pipeline
        if HAS_TORCH and transforms is not None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = None

        logger.info(f"ImageAuthenticityDetector ready (CNN active: {self.cnn_available}) on {self.device}")

    def error_level_analysis(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Error Level Analysis (ELA): Detects JPEG compression inconsistencies
        indicating possible manipulation using fast in-memory recompression.

        Args:
            image_path: Path to the input image.

        Returns:
            Tuple of (error_map ndarray, normalized error_level float).
        """
        img = Image.open(image_path)
        rgb_img = img.convert('RGB')

        # Fast in-memory recompression to avoid disk I/O
        buffer = io.BytesIO()
        rgb_img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        # Compute pixel-level difference
        original_array = np.array(rgb_img, dtype=np.float64)
        recompressed_array = np.array(recompressed, dtype=np.float64)

        # Error level map
        error_map = np.abs(original_array - recompressed_array)
        error_level = float(np.mean(error_map) / 255.0)  # Normalize to [0, 1]

        logger.debug(f"ELA error level: {error_level:.4f}")

        return error_map, error_level

    def generate_ela_heatmap(
        self,
        image_path: str,
        quality: int = 90,
        scale_multiplier: float = 16.0
    ) -> Dict[str, Any]:
        """
        Generate high-resolution Error Level Analysis (ELA) false-color heatmap.
        Blends localized JPEG compression error differences with the original image
        to visually pinpoint spliced regions, pasted text, and modified heads/logos.

        Args:
            image_path: Path to the input image.
            quality: Recompression JPEG quality factor (default 90).
            scale_multiplier: Enhancement gain for visualizing subtle error gradients.

        Returns:
            Dict containing heatmap_base64, error_level, max_error, hotspot_ratio, and splice_risk.
        """
        try:
            img = Image.open(image_path)
            rgb_img = img.convert('RGB')

            # Fast in-memory recompression
            buffer = io.BytesIO()
            rgb_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            # Compute pixel-level difference
            orig_arr = np.array(rgb_img, dtype=np.float32)
            recomp_arr = np.array(recompressed, dtype=np.float32)
            diff = np.abs(orig_arr - recomp_arr)

            # Average across color channels
            gray_diff = np.mean(diff, axis=2)
            error_level = float(np.mean(diff) / 255.0)
            max_error = float(np.max(diff) / 255.0)

            # Hotspot ratio: percentage of pixels with localized error > 3x median
            median_err = float(np.median(gray_diff))
            hotspot_threshold = max(18.0, median_err * 3.2)
            hotspot_ratio = float(np.mean(gray_diff > hotspot_threshold))

            # Block-level 16x16 ELA spike detection to catch localized news-card/graphic splices
            h_diff, w_diff = gray_diff.shape[:2]
            block_spike_ratio = 0.0
            if h_diff >= 32 and w_diff >= 32:
                bh, bw = h_diff // 16, w_diff // 16
                diff_crop = gray_diff[:bh * 16, :bw * 16]
                block_means = diff_crop.reshape(bh, 16, bw, 16).mean(axis=(1, 3)).ravel()
                b_med = float(np.median(block_means))
                block_spike_ratio = float(np.mean(block_means > max(3.5, b_med * 3.0 + 1.5)))

            # False-color thermal heatmap (INFERNO palette)
            scaled_err = np.clip(gray_diff * scale_multiplier, 0, 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(scaled_err, cv2.COLORMAP_INFERNO)

            # Blend with original for context (35% original + 65% thermal heatmap)
            orig_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(orig_bgr, 0.35, heatmap_bgr, 0.65, 0)

            # Resize if overly large to optimize network payload
            h, w = blended.shape[:2]
            if max(h, w) > 900:
                scale = 900 / max(h, w)
                blended = cv2.resize(blended, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

            splice_risk = 'high' if (
                hotspot_ratio > 0.08 or error_level > 0.12 or block_spike_ratio >= 0.025
            ) else (
                'moderate' if (
                    hotspot_ratio > 0.03 or error_level > 0.06 or block_spike_ratio >= 0.010
                ) else 'low'
            )

            return {
                'heatmap_base64': b64_str,
                'error_level': round(error_level, 4),
                'max_error': round(max_error, 4),
                'hotspot_ratio': round(hotspot_ratio, 4),
                'block_spike_ratio': round(block_spike_ratio, 4),
                'splice_risk': splice_risk
            }
        except Exception as e:
            logger.error(f"Error generating ELA heatmap: {e}")
            return {
                'heatmap_base64': None,
                'error_level': 0.0,
                'max_error': 0.0,
                'hotspot_ratio': 0.0,
                'splice_risk': 'unknown'
            }

    def noise_consistency_analysis(self, image_path: str, grid_size: int = 8) -> Dict[str, Any]:
        """
        Partition image into an 8x8 tile grid (64 patches) and compute local Laplacian
        noise variance. Detects composited elements, pasted logos, or spliced faces that have
        mismatched sensor noise profiles compared to the background.

        Args:
            image_path: Path to the input image.
            grid_size: Number of tiles per axis (default 8 -> 64 tiles).

        Returns:
            Dict with uniformity_score, inconsistent_tiles, variance_spread, and risk.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img)

            # Resize to standardized grid evaluation dimension
            standardized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            laplacian = cv2.Laplacian(standardized, cv2.CV_64F)

            tile_h = 512 // grid_size
            tile_w = 512 // grid_size

            variances = []
            for r in range(grid_size):
                for c in range(grid_size):
                    patch = laplacian[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
                    variances.append(float(np.var(patch)))

            variances = np.array(variances)
            median_var = float(np.median(variances))
            std_var = float(np.std(variances))

            # Outlier blocks with variance deviating significantly from sensor baseline
            high_outliers = np.sum(variances > (median_var + 2.4 * (std_var + 1e-6)))
            low_outliers = np.sum(variances < max(0.0, median_var - 2.0 * (std_var + 1e-6)))
            total_inconsistent = int(high_outliers + low_outliers)

            inconsistency_ratio = float(total_inconsistent / (grid_size * grid_size))
            uniformity_score = max(0.0, min(1.0, 1.0 - (inconsistency_ratio * 3.5)))

            risk = 'high' if inconsistency_ratio > 0.15 else ('moderate' if inconsistency_ratio > 0.07 else 'clean')

            return {
                'uniformity_score': round(float(uniformity_score), 4),
                'inconsistent_tiles': total_inconsistent,
                'total_tiles': grid_size * grid_size,
                'variance_spread': round(float(std_var / (median_var + 1e-6)), 4),
                'noise_risk': risk
            }
        except Exception as e:
            logger.error(f"Error in noise consistency analysis: {e}")
            return {
                'uniformity_score': 0.85,
                'inconsistent_tiles': 0,
                'total_tiles': 64,
                'variance_spread': 0.0,
                'noise_risk': 'clean'
            }

    def extract_exif_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract EXIF and provenance metadata. Detects camera make/model, original timestamps,
        and flags editing software signatures (Photoshop, Canva, PicsArt, PixelLab, GIMP).
        Detects if metadata was completely stripped (typical of social media compression).

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with has_exif, camera, software, timestamp, provenance_status, and details.
        """
        known_software_markers = [
            'photoshop', 'gimp', 'picsart', 'pixellab', 'canva', 'snapseed',
            'lightroom', 'midjourney', 'stable diffusion', 'dall-e', 'affinity',
            'paint.net', 'corel', 'procreate'
        ]

        try:
            img = Image.open(image_path)
            raw_exif = getattr(img, '_getexif', lambda: None)()

            if not raw_exif:
                return {
                    'has_exif': False,
                    'camera': 'None (Stripped)',
                    'software': 'None (Social Media / WhatsApp Recompression)',
                    'timestamp': 'None (Stripped)',
                    'provenance_status': 'stripped_social',
                    'editing_software_detected': False,
                    'details': {}
                }

            tag_dict = {}
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(val, (str, int, float)):
                    tag_dict[tag_name] = str(val).strip()

            make = tag_dict.get('Make', '')
            model = tag_dict.get('Model', '')
            camera_str = f"{make} {model}".strip() or 'None'
            software_str = tag_dict.get('Software', 'None')
            dt = tag_dict.get('DateTimeOriginal', tag_dict.get('DateTime', 'None'))

            # Check for editing software signatures
            software_lower = software_str.lower()
            editing_detected = any(marker in software_lower for marker in known_software_markers)

            if editing_detected:
                provenance = 'edited_software'
            elif camera_str != 'None':
                provenance = 'original_camera'
            else:
                provenance = 'standard'

            return {
                'has_exif': True,
                'camera': camera_str,
                'software': software_str,
                'timestamp': dt,
                'provenance_status': provenance,
                'editing_software_detected': editing_detected,
                'details': tag_dict
            }
        except Exception as e:
            logger.debug(f"EXIF parsing info: {e}")
            return {
                'has_exif': False,
                'camera': 'None',
                'software': 'None',
                'timestamp': 'None',
                'provenance_status': 'stripped_social',
                'editing_software_detected': False,
                'details': {}
            }

    def spectral_analysis(self, image_path: str) -> Dict:
        """
        Analyze the frequency domain via log-magnitude FFT to detect periodic
        synthetic generation artifacts or splicing anomalies while accounting for
        normal DC power-law falloff in natural photographs.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with avg_magnitude, std_magnitude, anomaly_score, and synthetic_risk.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None or img.size == 0:
            logger.warning(f"Could not read image: {image_path}")
            return {
                'avg_magnitude': 0.0,
                'std_magnitude': 0.0,
                'anomaly_score': 0.0,
                'synthetic_risk': 'clean'
            }

        h, w = img.shape[:2]
        if h < 16 or w < 16:
            return {
                'avg_magnitude': float(np.mean(img)),
                'std_magnitude': float(np.std(img)),
                'anomaly_score': 0.0,
                'synthetic_risk': 'clean'
            }

        # Log-scaled FFT magnitude spectrum (normalizes 1/f natural photo decay)
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        log_spectrum = np.log1p(np.abs(f_shift))

        avg_magnitude = float(np.mean(log_spectrum))
        std_magnitude = float(np.std(log_spectrum))

        # Mask DC center and primary border-wrap axes when measuring off-axis anomalies
        cy, cx = h // 2, w // 2
        r_margin = max(2, min(h, w) // 32)
        yy, xx = np.ogrid[:h, :w]
        off_axis_mask = (np.abs(yy - cy) > r_margin) & (np.abs(xx - cx) > r_margin)
        valid_ac = log_spectrum[off_axis_mask] if np.any(off_axis_mask) else log_spectrum.ravel()

        ac_mean = float(np.mean(valid_ac))
        ac_std = float(np.std(valid_ac))
        cv_log = ac_std / (ac_mean + 1e-6)

        ac_med = float(np.median(valid_ac))
        ac_mad = float(np.median(np.abs(valid_ac - ac_med))) + 1e-6
        p9995 = float(np.percentile(valid_ac, 99.95)) if valid_ac.size >= 100 else float(np.max(valid_ac))
        spike_z = (p9995 - ac_med) / (1.4826 * ac_mad)

        # Calibrated spectral anomaly score: natural photos/documents typically score 0.0–1.5
        anomaly_score = max(
            0.0,
            (cv_log - 0.08) * 12.0 + max(0.0, spike_z - 5.5) * 0.8
        )

        synthetic_risk = 'high' if (cv_log > 0.35 or spike_z > 9.5 or anomaly_score >= 6.5) else (
            'moderate' if (cv_log > 0.25 or spike_z > 7.5 or anomaly_score >= 4.0) else 'clean'
        )

        logger.debug(
            f"Spectral analysis — log_avg: {avg_magnitude:.2f}, cv_log: {cv_log:.3f}, "
            f"spike_z: {spike_z:.2f}, anomaly: {anomaly_score:.4f}, synthetic: {synthetic_risk}"
        )

        return {
            'avg_magnitude': float(avg_magnitude),
            'std_magnitude': float(std_magnitude),
            'anomaly_score': round(float(anomaly_score), 4),
            'synthetic_risk': synthetic_risk
        }

    def cnn_prediction(self, image_path: str) -> Dict:
        """
        CNN-based manipulation detection using ResNet50.

        If a fine-tuned checkpoint is loaded, computes authentic vs manipulated
        probabilities. Otherwise, reports unavailable to avoid returning
        random untrained guesses.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with available, authentic_prob, manipulated_prob, and prediction.
        """
        if not self.cnn_available or self.model is None:
            logger.debug("CNN manipulation model unavailable or untrained; skipping CNN inference.")
            return {
                'available': False,
                'authentic_prob': None,
                'manipulated_prob': None,
                'prediction': 'unavailable'
            }

        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=-1)[0].cpu().numpy()

        prediction = 'authentic' if probs[0] > probs[1] else 'manipulated'

        logger.debug(
            f"CNN prediction: {prediction} "
            f"(authentic={probs[0]:.3f}, manipulated={probs[1]:.3f})"
        )

        return {
            'available': True,
            'authentic_prob': float(probs[0]),
            'manipulated_prob': float(probs[1]),
            'prediction': prediction
        }

    @timer
    def assess_authenticity(self, image_path: str) -> Dict[str, Any]:
        """
        Full digital media forensics pipeline combining:
        1. Error Level Analysis (ELA) with false-color tampering heatmap
        2. Sensor noise consistency (8x8 tile Laplacian variance)
        3. EXIF and provenance software audit
        4. Spectral/FFT frequency anomaly & synthetic media detection
        5. Deep CNN prediction (when fine-tuned weights are present)

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with authenticity_score, ela_details, noise_details, exif_details,
            spectral_anomaly, manipulation_type, and forensic verdict.
        """
        logger.info(f"Executing comprehensive digital media forensics on: {image_path}")

        # 1. ELA Analysis & Heatmap
        ela_map, ela_score = self.error_level_analysis(image_path)
        ela_heatmap_data = self.generate_ela_heatmap(image_path)

        # 2. Noise consistency
        noise_data = self.noise_consistency_analysis(image_path)

        # 3. EXIF & Provenance
        exif_data = self.extract_exif_metadata(image_path)

        # 4. Spectral analysis
        spectral = self.spectral_analysis(image_path)

        # 5. CNN prediction
        cnn_result = self.cnn_prediction(image_path)

        # Normalized components
        norm_spectral = max(0.0, 1.0 - min(spectral['anomaly_score'] / 8.0, 1.0))
        block_spike = float(ela_heatmap_data.get('block_spike_ratio', 0.0))
        norm_ela = max(0.0, 1.0 - min(ela_score * 3.0 + block_spike * 18.0, 1.0))
        norm_noise = noise_data['uniformity_score']

        # Composite score weighting:
        # 35% ELA, 30% Noise Uniformity, 25% Spectral, 10% Provenance/CNN
        if cnn_result.get('available') and cnn_result.get('authentic_prob') is not None:
            base_score = (
                0.35 * norm_ela +
                0.25 * norm_noise +
                0.20 * norm_spectral +
                0.20 * cnn_result['authentic_prob']
            )
        else:
            base_score = (
                0.40 * norm_ela +
                0.35 * norm_noise +
                0.25 * norm_spectral
            )

        # Software tampering penalty if editing tool signatures are explicitly found
        if exif_data.get('editing_software_detected'):
            base_score = min(base_score, 0.45)

        authenticity_score = max(0.0, min(1.0, float(base_score)))

        # Determine manipulation diagnosis
        if exif_data.get('editing_software_detected'):
            manipulation_type = 'edited_software'
            verdict = 'potentially_manipulated'
        elif (
            ela_heatmap_data.get('splice_risk') == 'high'
            or noise_data.get('noise_risk') == 'high'
            or (ela_heatmap_data.get('splice_risk') == 'moderate' and block_spike >= 0.010)
        ):
            manipulation_type = 'spliced_composition'
            verdict = 'potentially_manipulated'
        elif spectral.get('synthetic_risk') == 'high':
            manipulation_type = 'synthetic_ai'
            verdict = 'potentially_manipulated'
        elif authenticity_score > 0.65:
            manipulation_type = 'authentic_capture'
            verdict = 'authentic'
        else:
            manipulation_type = 'compression_anomaly'
            verdict = 'potentially_manipulated'

        if verdict == 'potentially_manipulated':
            high_confidence_tamper = bool(
                manipulation_type in ('edited_software', 'spliced_composition', 'synthetic_ai')
                or exif_data.get('editing_software_detected')
                or ela_heatmap_data.get('splice_risk') == 'high'
                or noise_data.get('noise_risk') == 'high'
                or spectral.get('synthetic_risk') == 'high'
            )
            authenticity_score = min(
                authenticity_score,
                0.25 if high_confidence_tamper else 0.48
            )

        logger.info(
            f"Forensics verdict: {verdict} | Type: {manipulation_type} | "
            f"Score: {authenticity_score:.3f} | ELA Risk: {ela_heatmap_data.get('splice_risk')}"
        )

        return {
            'authenticity_score': float(authenticity_score),
            'ela_score': float(ela_score),
            'ela_details': ela_heatmap_data,
            'noise_consistency': noise_data,
            'exif_metadata': exif_data,
            'spectral_anomaly': spectral['anomaly_score'],
            'spectral_details': spectral,
            'cnn_prediction': cnn_result,
            'manipulation_type': manipulation_type,
            'verdict': verdict
        }
