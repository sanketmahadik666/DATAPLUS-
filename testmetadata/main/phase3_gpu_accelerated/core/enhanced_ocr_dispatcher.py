"""
Phase 3: Enhanced OCR Dispatcher with GPU Metadata Integration

Advanced OCR routing that leverages Phase 3 GPU-accelerated metadata and deep features
for superior engine selection and preprocessing recommendations.
"""

import json
import logging
import math
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

# Import Phase 3 components
try:
    from .gpu_metadata_processor import GPUMetadataProcessor
    from .deep_feature_extractor import extract_optimized_features
except ImportError:
    GPUMetadataProcessor = None
    extract_optimized_features = None

logger = logging.getLogger(__name__)


class EnhancedOCRDispatcher:
    """
    Advanced OCR dispatcher that integrates Phase 3 GPU metadata and deep features
    for intelligent engine routing and preprocessing optimization.
    """

    def __init__(self,
                 alpha: float = 1.0,  # Accuracy weight
                 beta: float = 0.5,   # Latency weight
                 gamma: float = 0.5,  # Resource cost weight
                 delta_threshold: float = 0.03,
                 use_gpu_features: bool = True):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_threshold = delta_threshold
        self.use_gpu_features = use_gpu_features

        # Enhanced engine specifications with Phase 3 capabilities
        self.engine_specs = {
            "tesseract_standard": {
                "latency_baseline": 0.8,  # seconds per page
                "resource_cost": 1.0,     # baseline cost
                "accuracy_profile": {
                    "general_text": 0.85,
                    "forms": 0.75,
                    "tables": 0.70,
                    "handwriting": 0.60,
                    "multilingual": 0.80
                },
                "strengths": ["general_text", "speed"],
                "weaknesses": ["tables", "handwriting"],
                "gpu_accelerated": False,
                "preprocessing_boost": 1.0
            },
            "tesseract_form_trained": {
                "latency_baseline": 1.0,
                "resource_cost": 1.2,
                "accuracy_profile": {
                    "general_text": 0.82,
                    "forms": 0.88,
                    "tables": 0.75,
                    "handwriting": 0.65,
                    "multilingual": 0.78
                },
                "strengths": ["forms", "structured_content"],
                "weaknesses": ["speed", "handwriting"],
                "gpu_accelerated": False,
                "preprocessing_boost": 1.1
            },
            "paddle_malayalam": {
                "latency_baseline": 0.5,
                "resource_cost": 3.0,
                "accuracy_profile": {
                    "general_text": 0.90,
                    "forms": 0.85,
                    "tables": 0.82,
                    "handwriting": 0.75,
                    "multilingual": 0.95
                },
                "strengths": ["multilingual", "tables", "gpu_acceleration"],
                "weaknesses": ["resource_intensive"],
                "gpu_accelerated": True,
                "preprocessing_boost": 1.3
            },
            "donut_tabular": {
                "latency_baseline": 1.5,
                "resource_cost": 4.0,
                "accuracy_profile": {
                    "general_text": 0.80,
                    "forms": 0.78,
                    "tables": 0.92,
                    "handwriting": 0.70,
                    "multilingual": 0.85
                },
                "strengths": ["tables", "structured_data"],
                "weaknesses": ["speed", "resource_intensive"],
                "gpu_accelerated": True,
                "preprocessing_boost": 1.4
            },
            "easyocr": {
                "latency_baseline": 1.0,
                "resource_cost": 1.1,
                "accuracy_profile": {
                    "general_text": 0.83,
                    "forms": 0.80,
                    "tables": 0.76,
                    "handwriting": 0.82,
                    "multilingual": 0.90
                },
                "strengths": ["handwriting", "multilingual"],
                "weaknesses": ["tables", "speed"],
                "gpu_accelerated": False,
                "preprocessing_boost": 1.2
            }
        }

        self.engines = list(self.engine_specs.keys())

        # Phase 3 feature integration
        self.feature_processor = None
        self.deep_feature_cache = {}

        # Model parameters (enhanced from Phase 2)
        self.class_priors = {}
        self.numeric_stats = {}
        self.categorical_stats = {}
        self.deep_feature_weights = {}
        self.features_used = []

        # Performance tracking
        self.routing_history = []
        self.accuracy_metrics = defaultdict(list)

        logger.info("Enhanced OCR Dispatcher initialized with Phase 3 GPU metadata integration")

    def initialize_phase3_features(self):
        """Initialize Phase 3 feature processing components."""
        if self.use_gpu_features and extract_optimized_features:
            logger.info("Phase 3 deep feature extraction enabled")
        else:
            logger.warning("Phase 3 deep features disabled - using Phase 2 fallback")

    def preprocess_phase3_metadata(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess documents with Phase 3 GPU metadata enhancement.
        Adds deep features, GPU-accelerated processing metadata, and enhanced feature extraction.
        """
        logger.info(f"Preprocessing {len(documents)} documents with Phase 3 metadata...")

        enhanced_documents = []

        for doc in documents:
            enhanced_doc = doc.copy()

            # Extract Phase 3 GPU processing metadata
            gpu_metadata = self._extract_gpu_processing_metadata(doc)

            # Add deep feature analysis if available
            if self.use_gpu_features and extract_optimized_features:
                deep_features = self._extract_document_deep_features(doc)
                enhanced_doc['phase3_deep_features'] = deep_features

            # Enhance feature extraction with GPU metadata
            enhanced_features = self._enhance_features_with_gpu_metadata(doc, gpu_metadata)
            enhanced_doc.update(enhanced_features)

            # Add GPU acceleration indicators
            enhanced_doc['gpu_accelerated_processing'] = gpu_metadata.get('gpu_accelerated', False)
            enhanced_doc['processing_method'] = gpu_metadata.get('processing_method', 'cpu')

            enhanced_documents.append(enhanced_doc)

        logger.info(f"Phase 3 preprocessing complete: {len(enhanced_documents)} documents enhanced")
        return enhanced_documents

    def _extract_gpu_processing_metadata(self, doc: Dict) -> Dict[str, Any]:
        """Extract GPU processing metadata from Phase 3 results."""
        gpu_metadata = {
            'gpu_accelerated': doc.get('gpu_accelerated', False),
            'processing_method': doc.get('processing_method', 'unknown'),
            'image_dimensions': doc.get('image_dimensions'),
            'deep_features_extracted': doc.get('deep_features_extracted', False),
            'feature_dimensions': doc.get('feature_dimensions', {}),
            'processing_timestamp': doc.get('processing_timestamp'),
        }

        # Add performance indicators
        if gpu_metadata['gpu_accelerated']:
            gpu_metadata['performance_boost'] = 2.5  # Estimated GPU speedup
            gpu_metadata['quality_improvement'] = 0.15  # Estimated accuracy boost

        return gpu_metadata

    def _extract_document_deep_features(self, doc: Dict) -> Dict[str, Any]:
        """Extract deep features for document analysis."""
        deep_features = {}

        try:
            # This would use the actual deep feature extractor in production
            # For now, simulate based on available metadata
            feature_dims = doc.get('feature_dimensions', {})

            deep_features = {
                'vision_features': feature_dims.get('vision', 2048),
                'text_features': feature_dims.get('text', 768),
                'structural_features': feature_dims.get('structural', 64),
                'fused_features': sum(feature_dims.values()),
                'feature_quality_score': self._calculate_feature_quality(doc),
                'semantic_complexity': self._assess_semantic_complexity(doc)
            }

        except Exception as e:
            logger.warning(f"Deep feature extraction failed: {e}")

        return deep_features

    def _enhance_features_with_gpu_metadata(self, doc: Dict, gpu_metadata: Dict) -> Dict[str, Any]:
        """Enhance traditional features with GPU processing metadata."""
        enhanced = {}

        # GPU acceleration indicators
        enhanced['gpu_boost_available'] = gpu_metadata.get('gpu_accelerated', False)
        enhanced['processing_speed_boost'] = gpu_metadata.get('performance_boost', 1.0)

        # Quality improvements from GPU processing
        base_accuracy = doc.get('text_density', 0) / 1000.0  # Normalize
        gpu_quality_boost = gpu_metadata.get('quality_improvement', 0.0)
        enhanced['gpu_enhanced_accuracy'] = min(1.0, base_accuracy + gpu_quality_boost)

        # Feature reliability indicators
        enhanced['feature_confidence'] = self._calculate_feature_confidence(doc, gpu_metadata)

        # Processing method optimization
        enhanced['optimal_processing_method'] = self._determine_optimal_processing_method(doc, gpu_metadata)

        return enhanced

    def _calculate_feature_quality(self, doc: Dict) -> float:
        """Calculate overall feature quality score."""
        quality_factors = []

        # GPU processing quality
        if doc.get('gpu_accelerated', False):
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.7)

        # Feature completeness
        feature_dims = doc.get('feature_dimensions', {})
        total_features = sum(feature_dims.values())
        if total_features > 2000:
            quality_factors.append(0.95)
        elif total_features > 1000:
            quality_factors.append(0.85)
        else:
            quality_factors.append(0.75)

        # Image quality indicators
        if doc.get('image_dimensions'):
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.6)

        return np.mean(quality_factors) if quality_factors else 0.5

    def _assess_semantic_complexity(self, doc: Dict) -> float:
        """Assess document semantic complexity."""
        complexity = 0.5  # Base complexity

        # Content type complexity
        if doc.get('has_tables', False):
            complexity += 0.2
        if doc.get('has_forms', False):
            complexity += 0.15
        if doc.get('has_images', False):
            complexity += 0.1

        # Text complexity
        text_density = doc.get('text_density', 0)
        if text_density > 500:
            complexity += 0.1
        elif text_density < 100:
            complexity -= 0.1

        # Language complexity
        if doc.get('has_multilingual', False):
            complexity += 0.15

        return min(1.0, max(0.0, complexity))

    def _calculate_feature_confidence(self, doc: Dict, gpu_metadata: Dict) -> float:
        """Calculate confidence in feature extraction."""
        confidence = 0.7  # Base confidence

        # GPU processing increases confidence
        if gpu_metadata.get('gpu_accelerated', False):
            confidence += 0.2

        # Deep features increase confidence
        if doc.get('deep_features_extracted', False):
            confidence += 0.1

        # Image quality affects confidence
        if doc.get('image_dimensions'):
            confidence += 0.1

        # Processing method reliability
        processing_method = gpu_metadata.get('processing_method', 'unknown')
        if processing_method in ['cupy_gpu', 'opencv_cuda']:
            confidence += 0.1
        elif processing_method == 'cpu':
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _determine_optimal_processing_method(self, doc: Dict, gpu_metadata: Dict) -> str:
        """Determine optimal processing method based on document characteristics."""
        if gpu_metadata.get('gpu_accelerated', False):
            if doc.get('has_tables', False) or doc.get('has_complex_layout', False):
                return 'gpu_deep_analysis'
            else:
                return 'gpu_standard'
        else:
            if doc.get('page_count', 1) > 10:
                return 'cpu_batch_optimized'
            else:
                return 'cpu_standard'

    def train_enhanced_model(self, documents: List[Dict]) -> Dict[str, Any]:
        """Train enhanced model with Phase 3 metadata integration."""
        logger.info(f"Training enhanced OCR dispatcher on {len(documents)} documents...")

        # Preprocess with Phase 3 features
        enhanced_documents = self.preprocess_phase3_metadata(documents)

        # Enhanced feature selection
        self.features_used = self._select_enhanced_features(enhanced_documents)

        # Extract training data
        X, y = self._extract_enhanced_training_data(enhanced_documents)

        # Estimate class priors with confidence weighting
        self._estimate_enhanced_class_priors(y, enhanced_documents)

        # Estimate likelihood parameters
        self._estimate_enhanced_numeric_likelihoods(X, y, enhanced_documents)
        self._estimate_enhanced_categorical_likelihoods(X, y)

        # Train deep feature integration
        self._train_deep_feature_integration(enhanced_documents)

        logger.info("Enhanced model training completed")

        return {
            "features_used": self.features_used,
            "class_priors": self.class_priors,
            "numeric_stats": self.numeric_stats,
            "categorical_stats": self.categorical_stats,
            "deep_feature_weights": self.deep_feature_weights,
            "phase3_enhancements": {
                "gpu_accelerated_documents": sum(1 for d in enhanced_documents if d.get('gpu_accelerated', False)),
                "deep_features_available": any(d.get('deep_features_extracted', False) for d in enhanced_documents),
                "enhanced_features": len(self.features_used)
            }
        }

    def _select_enhanced_features(self, documents: List[Dict]) -> List[str]:
        """Select features with Phase 3 enhancements."""
        base_features = [
            'page_count', 'total_characters', 'total_words', 'text_density',
            'aspect_ratio', 'has_tables', 'has_numbers', 'has_currency',
            'gpu_accelerated_processing', 'gpu_boost_available', 'processing_speed_boost'
        ]

        # Add Phase 3 specific features
        phase3_features = [
            'gpu_enhanced_accuracy', 'feature_confidence', 'feature_quality_score',
            'semantic_complexity', 'deep_features_extracted'
        ]

        # Filter available features
        available_features = set()
        for doc in documents:
            available_features.update(doc.keys())

        selected_features = []
        for feature in base_features + phase3_features:
            if feature in available_features:
                selected_features.append(feature)

        logger.info(f"Selected {len(selected_features)} enhanced features: {selected_features}")
        return selected_features

    def _extract_enhanced_training_data(self, documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Extract training data with Phase 3 enhancements."""
        X = []
        y = []

        for doc in documents:
            features = {}
            for feature_name in self.features_used:
                features[feature_name] = doc.get(feature_name)

            X.append(features)

            # Enhanced label extraction with fallbacks
            engine = self._extract_enhanced_engine_label(doc)
            y.append(engine)

        return X, y

    def _extract_enhanced_engine_label(self, doc: Dict) -> str:
        """Extract engine label with enhanced logic."""
        # Primary: explicit recommendation
        if 'recommended_ocr_engine' in doc:
            engine = doc['recommended_ocr_engine']
            # Normalize engine names
            engine_mapping = {
                'paddleocr': 'paddle_malayalam',
                'tesseract': 'tesseract_standard',
                'easyocr': 'easyocr'
            }
            return engine_mapping.get(engine, engine)

        # Secondary: infer from GPU capabilities and content
        if doc.get('gpu_accelerated', False):
            if doc.get('has_tables', False):
                return 'donut_tabular'
            elif doc.get('has_multilingual', False):
                return 'paddle_malayalam'
            else:
                return 'paddle_malayalam'  # GPU preferred for general content

        # Tertiary: content-based heuristics
        if doc.get('has_tables', False):
            return 'donut_tabular'
        elif doc.get('has_forms', False):
            return 'tesseract_form_trained'
        elif doc.get('has_multilingual', False):
            return 'easyocr'
        else:
            return 'tesseract_standard'

    def _estimate_enhanced_class_priors(self, y: List[str], documents: List[Dict]):
        """Estimate class priors with confidence weighting."""
        engine_counts = {}
        confidence_weights = {}

        for i, engine in enumerate(y):
            engine_counts[engine] = engine_counts.get(engine, 0) + 1

            # Weight by feature confidence
            confidence = documents[i].get('feature_confidence', 0.7)
            confidence_weights[engine] = confidence_weights.get(engine, 0) + confidence

        total_samples = len(y)
        total_confidence = sum(confidence_weights.values())

        for engine in self.engines:
            count = engine_counts.get(engine, 0)
            confidence_weight = confidence_weights.get(engine, 0)

            # Weighted prior with Laplace smoothing
            if total_confidence > 0:
                weighted_prior = (confidence_weight / total_confidence) * (count / total_samples)
            else:
                weighted_prior = count / total_samples

            self.class_priors[engine] = (weighted_prior + 1) / (1 + len(self.engines))

    def _estimate_enhanced_numeric_likelihoods(self, X: List[Dict], y: List[str], documents: List[Dict]):
        """Estimate numeric likelihoods with Phase 3 enhancements."""
        self.numeric_stats = {}

        for feature in self.features_used:
            if not self._is_numeric_feature([x.get(feature) for x in X if x.get(feature) is not None]):
                continue

            self.numeric_stats[feature] = {}

            for engine in self.engines:
                values = []
                weights = []

                for i, features in enumerate(X):
                    if y[i] == engine and features.get(feature) is not None:
                        try:
                            value = float(features[feature])
                            values.append(value)

                            # Weight by feature confidence
                            confidence = documents[i].get('feature_confidence', 0.7)
                            weights.append(confidence)
                        except (ValueError, TypeError):
                            continue

                if values:
                    # Weighted statistics
                    if weights:
                        weighted_mean = np.average(values, weights=weights)
                        # Simplified: use regular variance (could be weighted)
                        weighted_variance = np.var(values)
                    else:
                        weighted_mean = np.mean(values)
                        weighted_variance = np.var(values)

                    self.numeric_stats[feature][engine] = {
                        "mean": float(weighted_mean),
                        "variance": float(max(weighted_variance, 1e-6))
                    }

    def _estimate_enhanced_categorical_likelihoods(self, X: List[Dict], y: List[str]):
        """Estimate categorical likelihoods (unchanged from Phase 2)."""
        self.categorical_stats = {}

        for feature in self.features_used:
            if self._is_numeric_feature([x.get(feature) for x in X if x.get(feature) is not None]):
                continue

            self.categorical_stats[feature] = {}

            all_values = set()
            for features in X:
                if features.get(feature) is not None:
                    all_values.add(str(features[feature]))

            for value in all_values:
                self.categorical_stats[feature][value] = {}

                for engine in self.engines:
                    count = 0
                    total_for_engine = 0

                    for i, features in enumerate(X):
                        if y[i] == engine:
                            total_for_engine += 1
                            if str(features.get(feature, "")) == value:
                                count += 1

                    # Laplace smoothing
                    prob = (count + 1) / (total_for_engine + len(all_values))
                    self.categorical_stats[feature][value][engine] = float(prob)

    def _train_deep_feature_integration(self, documents: List[Dict]):
        """Train deep feature integration weights."""
        # Simplified: learn weights for different feature types
        deep_feature_performance = {}

        for doc in documents:
            if doc.get('deep_features_extracted', False):
                engine = self._extract_enhanced_engine_label(doc)
                feature_dims = doc.get('feature_dimensions', {})

                if engine not in deep_feature_performance:
                    deep_feature_performance[engine] = []

                # Use feature count as performance proxy
                total_features = sum(feature_dims.values())
                deep_feature_performance[engine].append(total_features)

        # Calculate average performance per engine
        for engine in self.engines:
            if engine in deep_feature_performance:
                avg_performance = np.mean(deep_feature_performance[engine])
                self.deep_feature_weights[engine] = float(avg_performance / 3000.0)  # Normalize
            else:
                self.deep_feature_weights[engine] = 0.5  # Default

    def predict_enhanced_routing(self, documents: List[Dict]) -> Dict[str, Any]:
        """Predict OCR routing with Phase 3 enhancements."""
        logger.info(f"Predicting enhanced OCR routing for {len(documents)} documents...")

        # Preprocess with Phase 3 features
        enhanced_documents = self.preprocess_phase3_metadata(documents)

        predictions = []

        for i, doc in enumerate(enhanced_documents):
            # Compute enhanced posteriors
            posteriors = self._compute_enhanced_posterior(doc)

            # Normalize posteriors
            total_posterior = sum(posteriors.values())
            if total_posterior > 0:
                for engine in posteriors:
                    posteriors[engine] /= total_posterior

            # Enhanced utility calculation
            utility_scores = self._compute_enhanced_utility(doc, posteriors)

            # Select optimal engine
            chosen_engine = max(utility_scores, key=utility_scores.get)

            # Enhanced fallback candidates
            fallback_candidates = self._find_enhanced_fallbacks(utility_scores, chosen_engine, doc)

            # Enhanced preprocessing recommendations
            preprocessing = self._recommend_enhanced_preprocessing(doc, chosen_engine)

            # Enhanced reasoning
            reasoning = self._generate_enhanced_reasoning(doc, chosen_engine, posteriors, utility_scores)

            prediction = {
                "document_id": doc.get('document_id', f'doc_{i}'),
                "phase3_enhanced": True,
                "gpu_accelerated": doc.get('gpu_accelerated', False),
                "deep_features_used": doc.get('deep_features_extracted', False),
                "posteriors": posteriors,
                "utility_scores": utility_scores,
                "chosen_engine": chosen_engine,
                "fallback_candidates": fallback_candidates,
                "expected_latency_sec": self._calculate_enhanced_latency(doc, chosen_engine),
                "expected_accuracy": self._calculate_expected_accuracy(doc, chosen_engine, posteriors),
                "confidence_score": posteriors[chosen_engine],
                "preprocessing_recommendations": preprocessing,
                "reasoning": reasoning,
                "performance_estimate": {
                    "speed_boost": doc.get('processing_speed_boost', 1.0),
                    "quality_improvement": doc.get('gpu_enhanced_accuracy', 0.0) - (doc.get('text_density', 0) / 1000.0)
                }
            }

            predictions.append(prediction)

        # Track routing history
        self.routing_history.extend(predictions)

        return {
            "routing_predictions": predictions,
            "phase3_metadata": {
                "gpu_accelerated_count": sum(1 for p in predictions if p['gpu_accelerated']),
                "deep_features_count": sum(1 for p in predictions if p['deep_features_used']),
                "enhanced_routing": True
            }
        }

    def _compute_enhanced_posterior(self, doc: Dict) -> Dict[str, float]:
        """Compute enhanced posterior probabilities with Phase 3 features."""
        posteriors = {}

        for engine in self.engines:
            log_posterior = math.log(self.class_priors.get(engine, 1e-6))

            # Standard feature likelihoods
            log_posterior += self._compute_standard_likelihoods(doc, engine)

            # Phase 3 feature enhancements
            log_posterior += self._compute_phase3_likelihood_boost(doc, engine)

            posteriors[engine] = math.exp(log_posterior)

        return posteriors

    def _compute_standard_likelihoods(self, doc: Dict, engine: str) -> float:
        """Compute standard Naive Bayes likelihoods."""
        log_likelihood = 0

        for feature in self.features_used:
            value = doc.get(feature)
            if value is None:
                continue

            # Numeric features
            if feature in self.numeric_stats and engine in self.numeric_stats[feature]:
                try:
                    x = float(value)
                    mean = self.numeric_stats[feature][engine]["mean"]
                    variance = self.numeric_stats[feature][engine]["variance"]
                    log_likelihood += self._gaussian_log_likelihood(x, mean, variance)
                except (ValueError, TypeError):
                    continue

            # Categorical features
            elif feature in self.categorical_stats:
                value_str = str(value)
                if value_str in self.categorical_stats[feature]:
                    if engine in self.categorical_stats[feature][value_str]:
                        prob = self.categorical_stats[feature][value_str][engine]
                        if prob > 0:
                            log_likelihood += math.log(prob)

        return log_likelihood

    def _compute_phase3_likelihood_boost(self, doc: Dict, engine: str) -> float:
        """Compute Phase 3 feature likelihood boost."""
        boost = 0

        # GPU acceleration boost
        if doc.get('gpu_accelerated', False) and self.engine_specs[engine]['gpu_accelerated']:
            boost += 0.5  # Prefer GPU engines when GPU processing available

        # Deep features boost
        if doc.get('deep_features_extracted', False):
            deep_weight = self.deep_feature_weights.get(engine, 0.5)
            boost += deep_weight * 0.3

        # Feature confidence boost
        confidence = doc.get('feature_confidence', 0.7)
        boost += (confidence - 0.7) * 0.2

        # Quality improvement boost
        quality_boost = doc.get('gpu_enhanced_accuracy', 0) - (doc.get('text_density', 0) / 1000.0)
        if quality_boost > 0:
            boost += quality_boost * 0.1

        return boost

    def _gaussian_log_likelihood(self, x: float, mean: float, variance: float) -> float:
        """Compute Gaussian log likelihood."""
        if variance <= 0:
            return 0
        return -0.5 * ((x - mean) ** 2) / variance - 0.5 * math.log(2 * math.pi * variance)

    def _compute_enhanced_utility(self, doc: Dict, posteriors: Dict[str, float]) -> Dict[str, float]:
        """Compute enhanced utility scores with Phase 3 optimizations."""
        page_count = doc.get('page_count', 1)
        utility_scores = {}

        # Get normalized latency and resource costs
        latencies = {}
        resource_costs = {}

        for engine in self.engines:
            latencies[engine] = self.engine_specs[engine]['latency_baseline'] * page_count
            resource_costs[engine] = self.engine_specs[engine]['resource_cost']

        # Normalize values
        min_lat, max_lat = min(latencies.values()), max(latencies.values())
        min_cost, max_cost = min(resource_costs.values()), max(resource_costs.values())

        normalized_latencies = {}
        normalized_costs = {}

        for engine in self.engines:
            if max_lat > min_lat:
                normalized_latencies[engine] = (latencies[engine] - min_lat) / (max_lat - min_lat)
            else:
                normalized_latencies[engine] = 0.5

            if max_cost > min_cost:
                normalized_costs[engine] = (resource_costs[engine] - min_cost) / (max_cost - min_cost)
            else:
                normalized_costs[engine] = 0.5

        # Compute enhanced utility
        for engine in self.engines:
            accuracy_component = self.alpha * posteriors[engine]
            latency_component = -self.beta * normalized_latencies[engine]
            cost_component = -self.gamma * normalized_costs[engine]

            # Phase 3 enhancements
            gpu_boost = 0
            if doc.get('gpu_accelerated', False) and self.engine_specs[engine]['gpu_accelerated']:
                gpu_boost = 0.1  # Bonus for GPU engines with GPU processing

            preprocessing_boost = self.engine_specs[engine]['preprocessing_boost'] * 0.05

            utility_scores[engine] = float(accuracy_component + latency_component + cost_component + gpu_boost + preprocessing_boost)

        return utility_scores

    def _find_enhanced_fallbacks(self, utility_scores: Dict[str, float], chosen_engine: str,
                               doc: Dict) -> List[Dict[str, Any]]:
        """Find enhanced fallback candidates with Phase 3 considerations."""
        fallback_candidates = []
        chosen_utility = utility_scores[chosen_engine]

        for engine, utility in utility_scores.items():
            if engine == chosen_engine:
                continue

            utility_gap = chosen_utility - utility
            if utility_gap <= self.delta_threshold:
                # Phase 3: Consider GPU compatibility for fallbacks
                gpu_compatible = (
                    doc.get('gpu_accelerated', False) == self.engine_specs[engine]['gpu_accelerated']
                )

                fallback_candidates.append({
                    "engine": engine,
                    "utility_gap": float(utility_gap),
                    "gpu_compatible": gpu_compatible,
                    "performance_similarity": 1.0 - utility_gap,
                    "reason": self._explain_fallback_choice(engine, doc)
                })

        return fallback_candidates

    def _explain_fallback_choice(self, engine: str, doc: Dict) -> str:
        """Explain why an engine is a good fallback."""
        reasons = []

        if doc.get('gpu_accelerated', False) and self.engine_specs[engine]['gpu_accelerated']:
            reasons.append("GPU compatible")

        engine_strengths = self.engine_specs[engine]['strengths']
        doc_features = []

        if doc.get('has_tables', False):
            doc_features.append('tables')
        if doc.get('has_forms', False):
            doc_features.append('forms')
        if doc.get('has_multilingual', False):
            doc_features.append('multilingual')

        matching_strengths = set(engine_strengths) & set(doc_features)
        if matching_strengths:
            reasons.append(f"good for {', '.join(matching_strengths)}")

        return ', '.join(reasons) if reasons else "utility threshold met"

    def _recommend_enhanced_preprocessing(self, doc: Dict, chosen_engine: str) -> List[Dict[str, Any]]:
        """Recommend enhanced preprocessing with Phase 3 insights."""
        recommendations = []

        # Basic preprocessing (from Phase 2)
        if doc.get('aspect_ratio', 1.0) < 0.5 or doc.get('aspect_ratio', 1.0) > 2.0:
            recommendations.append({
                "step": "deskew",
                "reason": "Irregular aspect ratio suggests skewed document",
                "impact": "high",
                "engine_benefit": "all"
            })

        if doc.get('text_density', 0) < 100:
            recommendations.append({
                "step": "DPI_increase",
                "reason": "Low text density indicates poor resolution",
                "impact": "high",
                "engine_benefit": chosen_engine
            })

        if doc.get('noise_level', 0) > 0.3:
            recommendations.append({
                "step": "denoise",
                "reason": "High noise level detected",
                "impact": "medium",
                "engine_benefit": "all"
            })

        # Phase 3 enhanced recommendations
        if doc.get('gpu_accelerated', False) and self.engine_specs[chosen_engine]['gpu_accelerated']:
            recommendations.append({
                "step": "gpu_optimization",
                "reason": "GPU processing available for enhanced accuracy",
                "impact": "high",
                "engine_benefit": chosen_engine
            })

        if doc.get('deep_features_extracted', False):
            recommendations.append({
                "step": "semantic_enhancement",
                "reason": "Deep features available for intelligent preprocessing",
                "impact": "medium",
                "engine_benefit": "all"
            })

        # Engine-specific recommendations
        if chosen_engine == 'donut_tabular' and doc.get('has_tables', False):
            recommendations.append({
                "step": "table_extraction",
                "reason": "Table-specific OCR engine selected",
                "impact": "high",
                "engine_benefit": "donut_tabular"
            })

        if chosen_engine == 'paddle_malayalam' and doc.get('has_multilingual', False):
            recommendations.append({
                "step": "language_detection",
                "reason": "Multi-language OCR engine selected",
                "impact": "high",
                "engine_benefit": "paddle_malayalam"
            })

        return recommendations

    def _generate_enhanced_reasoning(self, doc: Dict, chosen_engine: str,
                                   posteriors: Dict[str, float], utility_scores: Dict[str, float]) -> str:
        """Generate enhanced reasoning with Phase 3 insights."""
        reasons = []
        confidence = posteriors[chosen_engine]

        # Confidence level
        if confidence > 0.8:
            reasons.append("very high confidence")
        elif confidence > 0.6:
            reasons.append("high confidence")
        elif confidence > 0.4:
            reasons.append("moderate confidence")
        else:
            reasons.append("exploratory selection")

        # Phase 3 enhancements
        if doc.get('gpu_accelerated', False) and self.engine_specs[chosen_engine]['gpu_accelerated']:
            reasons.append("GPU processing available and utilized")

        if doc.get('deep_features_extracted', False):
            reasons.append("deep feature analysis enabled")

        # Content analysis
        engine_strengths = self.engine_specs[chosen_engine]['strengths']
        doc_characteristics = []

        if doc.get('has_tables', False):
            doc_characteristics.append('tables')
        if doc.get('has_forms', False):
            doc_characteristics.append('forms')
        if doc.get('has_multilingual', False):
            doc_characteristics.append('multilingual content')
        if doc.get('text_density', 0) > 500:
            doc_characteristics.append('high text density')

        matching = set(engine_strengths) & set(doc_characteristics)
        if matching:
            reasons.append(f"optimal for {', '.join(matching)}")

        # Performance considerations
        if utility_scores[chosen_engine] > 0.7:
            reasons.append("excellent utility score across accuracy, speed, and cost")

        return f"Selected {chosen_engine} due to {', '.join(reasons)}"

    def _calculate_enhanced_latency(self, doc: Dict, engine: str) -> float:
        """Calculate enhanced latency estimate with Phase 3 optimizations."""
        base_latency = self.engine_specs[engine]['latency_baseline']
        page_count = doc.get('page_count', 1)

        # GPU acceleration reduces latency
        if doc.get('gpu_accelerated', False) and self.engine_specs[engine]['gpu_accelerated']:
            base_latency *= 0.6  # 40% speedup with GPU

        # Deep features can optimize processing
        if doc.get('deep_features_extracted', False):
            base_latency *= 0.9  # 10% optimization

        return float(base_latency * page_count)

    def _calculate_expected_accuracy(self, doc: Dict, engine: str, posteriors: Dict[str, float]) -> float:
        """Calculate expected accuracy with Phase 3 enhancements."""
        base_accuracy = self.engine_specs[engine]['accuracy_profile'].get('general_text', 0.8)

        # GPU processing improves accuracy
        if doc.get('gpu_accelerated', False) and self.engine_specs[engine]['gpu_accelerated']:
            base_accuracy += 0.05  # 5% boost

        # Content-specific accuracy adjustments
        if doc.get('has_tables', False) and engine == 'donut_tabular':
            base_accuracy += 0.1
        elif doc.get('has_forms', False) and engine == 'tesseract_form_trained':
            base_accuracy += 0.08
        elif doc.get('has_multilingual', False) and engine in ['paddle_malayalam', 'easyocr']:
            base_accuracy += 0.07

        # Confidence weighting
        confidence_boost = (posteriors[engine] - 0.5) * 0.1
        base_accuracy += confidence_boost

        return min(1.0, max(0.0, base_accuracy))

    def _is_numeric_feature(self, values: List) -> bool:
        """Check if feature values are numeric."""
        try:
            for val in values:
                if val is not None and val != "":
                    float(val)
            return True
        except (ValueError, TypeError):
            return False

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics."""
        if not self.routing_history:
            return {}

        engine_usage = Counter()
        gpu_usage = 0
        deep_feature_usage = 0
        avg_confidence = []

        for prediction in self.routing_history:
            engine_usage[prediction['chosen_engine']] += 1
            if prediction['gpu_accelerated']:
                gpu_usage += 1
            if prediction['deep_features_used']:
                deep_feature_usage += 1
            avg_confidence.append(prediction['confidence_score'])

        return {
            "total_predictions": len(self.routing_history),
            "engine_distribution": dict(engine_usage),
            "gpu_accelerated_predictions": gpu_usage,
            "deep_features_used": deep_feature_usage,
            "average_confidence": np.mean(avg_confidence),
            "phase3_enhancement_rate": (gpu_usage + deep_feature_usage) / len(self.routing_history)
        }


def create_enhanced_ocr_routing_response(documents: List[Dict],
                                       alpha: float = 1.0,
                                       beta: float = 0.5,
                                       gamma: float = 0.5,
                                       delta_threshold: float = 0.03,
                                       use_gpu_features: bool = True) -> Dict[str, Any]:
    """Create enhanced OCR routing response with Phase 3 integration."""

    # Initialize enhanced dispatcher
    dispatcher = EnhancedOCRDispatcher(
        alpha=alpha, beta=beta, gamma=gamma,
        delta_threshold=delta_threshold,
        use_gpu_features=use_gpu_features
    )

    # Load or create training data (would use actual training data in production)
    # For demo, use the input documents as training data
    training_data = documents

    # Train enhanced model
    model_summary = dispatcher.train_enhanced_model(training_data)

    # Generate routing predictions
    routing_result = dispatcher.predict_enhanced_routing(documents)

    # Get analytics
    analytics = dispatcher.get_routing_analytics()

    # Create comprehensive response
    response = {
        "metadata": {
            "algorithm": "enhanced_naive_bayes_with_phase3_gpu",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "phase3_enabled": use_gpu_features,
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "delta_fallback_threshold": delta_threshold
        },
        "model_summary": model_summary,
        "routing_results": routing_result,
        "analytics": analytics,
        "engine_specs": dispatcher.engine_specs,
        "enhancement_summary": {
            "gpu_accelerated_engines": [e for e, s in dispatcher.engine_specs.items() if s['gpu_accelerated']],
            "phase3_feature_boost": "enabled" if use_gpu_features else "disabled",
            "deep_feature_integration": bool(dispatcher.deep_feature_weights),
            "enhanced_preprocessing": True
        }
    }

    return response