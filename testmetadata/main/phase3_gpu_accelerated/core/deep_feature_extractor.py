"""
Phase 3: Deep Feature Extraction Module

Advanced feature extraction using deep learning models for comprehensive document analysis.
Supports multiple neural architectures and feature fusion techniques.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class DeepFeatureExtractor:
    """Advanced deep feature extraction with multiple neural architectures."""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.extractors = {}
        self._load_models()

    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                device = "cuda"
            elif HAS_TENSORFLOW and tf.config.list_physical_devices('GPU'):
                device = "tensorflow_gpu"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Deep Feature Extractor using device: {device}")
        return device

    def _load_models(self):
        """Load multiple deep learning models for comprehensive feature extraction."""
        if HAS_TORCH:
            self._load_pytorch_models()
        if HAS_TENSORFLOW:
            self._load_tensorflow_models()
        if HAS_TRANSFORMERS:
            self._load_transformer_models()

    def _load_pytorch_models(self):
        """Load PyTorch-based models."""
        try:
            # Vision models
            vision_models = {
                'resnet50': models.resnet50(pretrained=True),
                'resnet101': models.resnet101(pretrained=True),
                'efficientnet_b4': models.efficientnet_b4(pretrained=True),
                'vit_b16': models.vit_b_16(pretrained=True),
            }

            for name, model in vision_models.items():
                # Remove classification head to get features
                if hasattr(model, 'fc'):
                    model.fc = nn.Identity()
                elif hasattr(model, 'classifier'):
                    model.classifier = nn.Identity()

                model.to(self.device)
                model.eval()
                self.models[f'pytorch_{name}'] = model

            # Text models
            self.models['pytorch_bert'] = models.bert(pretrained_model_name_or_path='bert-base-uncased')

        except Exception as e:
            logger.warning(f"Failed to load PyTorch models: {e}")

    def _load_tensorflow_models(self):
        """Load TensorFlow/Keras models."""
        try:
            # Vision models
            tf_vision_models = {
                'mobilenet_v3': tf.keras.applications.MobileNetV3Large(include_top=False),
                'nasnet_mobile': tf.keras.applications.NASNetMobile(include_top=False),
            }

            for name, model in tf_vision_models.items():
                self.models[f'tensorflow_{name}'] = model

        except Exception as e:
            logger.warning(f"Failed to load TensorFlow models: {e}")

    def _load_transformer_models(self):
        """Load transformer-based models."""
        try:
            # Vision transformers
            vision_transformers = [
                'google/vit-base-patch16-224',
                'facebook/dino-vitb16',
            ]

            for model_name in vision_transformers:
                try:
                    model = AutoModel.from_pretrained(model_name)
                    extractor = AutoFeatureExtractor.from_pretrained(model_name)
                    self.models[f'transformer_{model_name.split("/")[-1]}'] = model
                    self.extractors[f'transformer_{model_name.split("/")[-1]}'] = extractor
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load transformer models: {e}")

    def extract_vision_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract deep vision features from image."""
        features = {}

        if not HAS_TORCH:
            return features

        try:
            # Prepare image tensor
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            tensor = transform(image).unsqueeze(0).to(self.device)

            # Extract features from different models
            with torch.no_grad():
                for model_name, model in self.models.items():
                    if model_name.startswith('pytorch_') and 'bert' not in model_name:
                        try:
                            output = model(tensor)
                            if isinstance(output, torch.Tensor):
                                features[model_name] = output.squeeze().cpu().numpy()
                            elif hasattr(output, 'pooler_output'):
                                features[model_name] = output.pooler_output.squeeze().cpu().numpy()
                        except Exception as e:
                            logger.warning(f"Feature extraction failed for {model_name}: {e}")

        except Exception as e:
            logger.warning(f"Vision feature extraction failed: {e}")

        return features

    def extract_text_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract deep text features from document text."""
        features = {}

        if not text or not HAS_TRANSFORMERS:
            return features

        try:
            # Use BERT for text features
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')

            inputs = tokenizer(text[:512], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                features['bert_text_features'] = outputs.pooler_output.squeeze().cpu().numpy()

        except Exception as e:
            logger.warning(f"Text feature extraction failed: {e}")

        return features

    def extract_structural_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract advanced structural and layout features."""
        features = {}

        try:
            # Document layout analysis
            height, width = image.shape[:2]

            # Divide image into grid and analyze each region
            grid_size = 8
            h_step, w_step = height // grid_size, width // grid_size

            grid_features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    region = image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                    if len(region.shape) == 3:
                        region = np.mean(region, axis=2)

                    # Extract region statistics
                    region_stats = {
                        'mean': float(np.mean(region)),
                        'std': float(np.std(region)),
                        'min': float(np.min(region)),
                        'max': float(np.max(region)),
                        'entropy': self._calculate_entropy(region),
                    }
                    grid_features.append(region_stats)

            features['grid_analysis'] = grid_features
            features['layout_complexity'] = np.std([r['entropy'] for r in grid_features])

        except Exception as e:
            logger.warning(f"Structural feature extraction failed: {e}")

        return features

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image region."""
        histogram = np.histogram(image, bins=32, range=(0, 256))[0]
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram)) if len(histogram) > 0 else 0.0

    def fuse_features(self, feature_dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Fuse features from multiple sources using various techniques."""
        fused_features = {}

        if not feature_dicts:
            return fused_features

        try:
            # Concatenation fusion
            all_features = []
            feature_names = []

            for feat_dict in feature_dicts:
                for name, features in feat_dict.items():
                    if isinstance(features, np.ndarray):
                        all_features.append(features.flatten())
                        feature_names.append(name)

            if all_features:
                # Simple concatenation
                fused_features['concatenated'] = np.concatenate(all_features)

                # Statistical fusion (mean, std, etc.)
                fused_features['mean_fusion'] = np.mean(all_features, axis=0)
                fused_features['max_fusion'] = np.max(all_features, axis=0)
                fused_features['std_fusion'] = np.std(all_features, axis=0)

        except Exception as e:
            logger.warning(f"Feature fusion failed: {e}")

        return fused_features

    def extract_comprehensive_features(self, image: np.ndarray, text: Optional[str] = None) -> Dict[str, Any]:
        """Extract comprehensive features combining vision, text, and structural analysis."""
        comprehensive_features = {}

        # Vision features
        vision_features = self.extract_vision_features(image)
        comprehensive_features.update(vision_features)

        # Text features
        if text:
            text_features = self.extract_text_features(text)
            comprehensive_features.update(text_features)

        # Structural features
        structural_features = self.extract_structural_features(image)
        comprehensive_features.update(structural_features)

        # Feature fusion
        feature_sources = [vision_features]
        if text_features:
            feature_sources.append(text_features)

        fused_features = self.fuse_features(feature_sources)
        comprehensive_features['fused_features'] = fused_features

        # Metadata
        comprehensive_features['feature_metadata'] = {
            'vision_models_used': list(vision_features.keys()),
            'text_models_used': list(text_features.keys()) if text_features else [],
            'structural_analysis': bool(structural_features),
            'fusion_applied': bool(fused_features),
            'total_feature_dimensions': sum(
                f.size if hasattr(f, 'size') else len(f) if hasattr(f, '__len__') else 1
                for f in comprehensive_features.values()
                if isinstance(f, (np.ndarray, list, dict))
            )
        }

        return comprehensive_features


class AdvancedFeatureProcessor:
    """Advanced feature processing with dimensionality reduction and optimization."""

    def __init__(self):
        self.pca_components = {}
        self.scalers = {}
        self.feature_selectors = {}

    def optimize_features(self, features: Dict[str, np.ndarray], target_dim: int = 256) -> Dict[str, np.ndarray]:
        """Optimize feature dimensions using PCA and selection techniques."""
        optimized_features = {}

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import SelectKBest, f_classif

            for name, feature_vector in features.items():
                if not isinstance(feature_vector, np.ndarray):
                    continue

                # Flatten if needed
                if feature_vector.ndim > 1:
                    feature_vector = feature_vector.flatten()

                # Standardize
                if name not in self.scalers:
                    self.scalers[name] = StandardScaler()

                scaled_features = self.scalers[name].fit_transform(feature_vector.reshape(1, -1))

                # Apply PCA for dimensionality reduction
                if scaled_features.shape[1] > target_dim:
                    if name not in self.pca_components:
                        self.pca_components[name] = PCA(n_components=min(target_dim, scaled_features.shape[1]))

                    optimized_features[f'{name}_optimized'] = self.pca_components[name].fit_transform(scaled_features).flatten()
                else:
                    optimized_features[f'{name}_optimized'] = scaled_features.flatten()

        except ImportError:
            logger.warning("sklearn not available for feature optimization")
            optimized_features = {f"{k}_raw": v for k, v in features.items()}

        return optimized_features

    def extract_semantic_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract semantic features for document understanding."""
        semantic_features = {}

        try:
            # Document type classification features
            semantic_features['document_type_scores'] = self._classify_document_type(image)

            # Layout analysis features
            semantic_features['layout_features'] = self._analyze_layout(image)

            # Content density features
            semantic_features['content_density'] = self._calculate_content_density(image)

        except Exception as e:
            logger.warning(f"Semantic feature extraction failed: {e}")

        return semantic_features

    def _classify_document_type(self, image: np.ndarray) -> Dict[str, float]:
        """Classify document type based on visual features."""
        # Simple heuristic-based classification
        height, width = image.shape[:2]
        aspect_ratio = width / height

        # Calculate text density (rough approximation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        text_density = np.mean(gray < 200)  # Assume dark pixels are text

        scores = {
            'invoice': 0.0,
            'form': 0.0,
            'letter': 0.0,
            'document': 0.0
        }

        # Invoice-like characteristics
        if 0.5 < aspect_ratio < 2.0 and text_density > 0.3:
            scores['invoice'] = 0.8
        elif aspect_ratio > 2.0 and text_density > 0.4:
            scores['form'] = 0.7
        elif aspect_ratio < 0.8 and text_density > 0.2:
            scores['letter'] = 0.6
        else:
            scores['document'] = 0.5

        return scores

    def _analyze_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze document layout structure."""
        layout_features = {}

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Edge detection for layout analysis
            edges = cv2.Canny(gray, 50, 150)

            # Find lines (potential table borders, dividers)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

            layout_features['line_count'] = len(lines) if lines is not None else 0
            layout_features['has_tables'] = layout_features['line_count'] > 10

            # Connected component analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8)
            layout_features['component_count'] = num_labels
            layout_features['text_blocks'] = len([s for s in stats if s[4] > 100])  # Filter small components

        except Exception as e:
            logger.warning(f"Layout analysis failed: {e}")

        return layout_features

    def _calculate_content_density(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate content density metrics."""
        density_features = {}

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Overall density
            density_features['overall_density'] = np.mean(gray < 128)

            # Regional density analysis
            h, w = gray.shape
            regions = {
                'top': gray[:h//3, :],
                'middle': gray[h//3:2*h//3, :],
                'bottom': gray[2*h//3:, :],
                'left': gray[:, :w//3],
                'center': gray[:, w//3:2*w//3],
                'right': gray[:, 2*w//3:],
            }

            for region_name, region in regions.items():
                density_features[f'{region_name}_density'] = np.mean(region < 128)

        except Exception as e:
            logger.warning(f"Content density calculation failed: {e}")

        return density_features


# Convenience functions
def extract_deep_features(image: np.ndarray, text: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for deep feature extraction."""
    extractor = DeepFeatureExtractor()
    return extractor.extract_comprehensive_features(image, text)


def extract_optimized_features(image: np.ndarray, text: Optional[str] = None, target_dim: int = 256) -> Dict[str, Any]:
    """Convenience function for optimized feature extraction."""
    extractor = DeepFeatureExtractor()
    processor = AdvancedFeatureProcessor()

    # Extract comprehensive features
    features = extractor.extract_comprehensive_features(image, text)

    # Extract semantic features
    semantic = processor.extract_semantic_features(image)
    features.update(semantic)

    # Optimize dimensions
    vision_features = {k: v for k, v in features.items() if isinstance(v, np.ndarray)}
    optimized = processor.optimize_features(vision_features, target_dim)
    features['optimized_features'] = optimized

    return features