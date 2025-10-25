#!/usr/bin/env python3
"""
Naive Bayes OCR Routing Dispatcher Microservice
Routes documents to optimal OCR engines based on document features using Naive Bayes classification
"""

import json
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaiveBayesOCRRouter:
    """Naive Bayes classifier for OCR engine routing with utility function optimization"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, delta_threshold=0.03):
        self.alpha = alpha  # Weight for accuracy
        self.beta = beta    # Weight for latency
        self.gamma = gamma  # Weight for resource cost
        self.delta_threshold = delta_threshold
        
        # Engine baselines
        self.engine_latency_baseline = {
            "tesseract_standard": 0.8,
            "tesseract_form_trained": 1.0,
            "paddle_malayalam": 0.5,
            "donut_tabular": 1.5,
            "easyocr": 1.0
        }
        
        self.engine_resource_baseline = {
            "tesseract_standard": 1.0,
            "tesseract_form_trained": 1.2,
            "paddle_malayalam": 3.0,
            "donut_tabular": 4.0,
            "easyocr": 1.1
        }
        
        # Model parameters
        self.class_priors = {}
        self.numeric_stats = {}
        self.categorical_stats = {}
        self.features_used = []
        self.engines = list(self.engine_latency_baseline.keys())
        
    def preprocess_features(self, documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Preprocess and select features for training"""
        logger.info("Preprocessing document features...")
        
        # Extract all possible features from documents
        all_features = set()
        for doc in documents:
            # Flatten nested structures
            flat_doc = self._flatten_document(doc)
            all_features.update(flat_doc.keys())
        
        # Select relevant features (exclude metadata fields)
        exclude_fields = {
            'document_id', 'file_name', 'file_path', 'file_size_bytes', 
            'file_modified_time', 'processing_status', 'processing_time',
            'recommended_ocr_engine', 'font_sizes', 'language_indicators'
        }
        
        numeric_features = []
        categorical_features = []
        
        for feature in all_features:
            if feature in exclude_fields:
                continue
                
            # Check if feature is numeric or categorical
            values = []
            for doc in documents:
                flat_doc = self._flatten_document(doc)
                if feature in flat_doc:
                    values.append(flat_doc[feature])
            
            if not values:
                continue
                
            # Determine if numeric or categorical
            if self._is_numeric_feature(values):
                numeric_features.append(feature)
            else:
                categorical_features.append(feature)
        
        # Select top features based on variance and information gain
        selected_features = self._select_best_features(documents, numeric_features, categorical_features)
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features
    
    def _flatten_document(self, doc: Dict) -> Dict:
        """Flatten nested document structure"""
        flat = {}
        for key, value in doc.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}_{sub_key}"] = sub_value
            else:
                flat[key] = value
        return flat
    
    def _is_numeric_feature(self, values: List) -> bool:
        """Check if feature values are numeric"""
        try:
            for val in values:
                if val is not None and val != "":
                    float(val)
            return True
        except (ValueError, TypeError):
            return False
    
    def _select_best_features(self, documents: List[Dict], numeric_features: List[str], 
                            categorical_features: List[str]) -> List[str]:
        """Select best features based on variance and information gain"""
        selected = []
        
        # Always include key features
        key_features = ['page_count', 'total_characters', 'total_words', 'text_density', 
                       'aspect_ratio', 'has_tables', 'has_numbers', 'has_currency']
        
        for feature in key_features:
            if feature in numeric_features or feature in categorical_features:
                selected.append(feature)
        
        # Add other numeric features with high variance
        for feature in numeric_features:
            if feature not in selected and len(selected) < 15:  # Limit features
                values = []
                for doc in documents:
                    flat_doc = self._flatten_document(doc)
                    if feature in flat_doc and flat_doc[feature] is not None:
                        values.append(float(flat_doc[feature]))
                
                if values and np.var(values) > 0.01:  # Minimum variance threshold
                    selected.append(feature)
        
        # Add categorical features with good distribution
        for feature in categorical_features:
            if feature not in selected and len(selected) < 20:  # Limit features
                values = []
                for doc in documents:
                    flat_doc = self._flatten_document(doc)
                    if feature in flat_doc and flat_doc[feature] is not None:
                        values.append(flat_doc[feature])
                
                if values and len(set(values)) > 1:  # Multiple values
                    selected.append(feature)
        
        return selected
    
    def train(self, documents: List[Dict]) -> Dict[str, Any]:
        """Train the Naive Bayes model"""
        logger.info(f"Training Naive Bayes model on {len(documents)} documents...")
        
        # Preprocess and select features
        self.features_used = self.preprocess_features(documents)
        
        # Extract training data
        X, y = self._extract_training_data(documents)
        
        # Estimate class priors
        self._estimate_class_priors(y)
        
        # Estimate likelihood parameters
        self._estimate_numeric_likelihoods(X, y)
        self._estimate_categorical_likelihoods(X, y)
        
        logger.info("Model training completed")
        
        return {
            "features_used": self.features_used,
            "class_priors": self.class_priors,
            "numeric_stats": self.numeric_stats,
            "categorical_stats": self.categorical_stats
        }
    
    def _extract_training_data(self, documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Extract feature vectors and labels from documents"""
        X = []
        y = []
        
        for doc in documents:
            flat_doc = self._flatten_document(doc)
            
            # Extract features
            features = {}
            for feature in self.features_used:
                if feature in flat_doc:
                    features[feature] = flat_doc[feature]
                else:
                    features[feature] = None
            
            X.append(features)
            
            # Extract label
            if 'recommended_ocr_engine' in flat_doc:
                engine = flat_doc['recommended_ocr_engine']
                # Map to standard engine names
                if engine == 'paddleocr':
                    engine = 'paddle_malayalam'
                elif engine == 'tesseract':
                    engine = 'tesseract_standard'
                elif engine == 'easyocr':
                    engine = 'easyocr'
                y.append(engine)
            else:
                y.append('tesseract_standard')  # Default
        
        return X, y
    
    def _estimate_class_priors(self, y: List[str]):
        """Estimate class priors P(engine)"""
        engine_counts = Counter(y)
        total = len(y)
        
        for engine in self.engines:
            count = engine_counts.get(engine, 0)
            self.class_priors[engine] = (count + 1) / (total + len(self.engines))  # Laplace smoothing
    
    def _estimate_numeric_likelihoods(self, X: List[Dict], y: List[str]):
        """Estimate Gaussian parameters for numeric features"""
        self.numeric_stats = {}
        
        for feature in self.features_used:
            if not self._is_numeric_feature([x.get(feature) for x in X if x.get(feature) is not None]):
                continue
                
            self.numeric_stats[feature] = {}
            
            for engine in self.engines:
                # Get values for this engine
                values = []
                for i, features in enumerate(X):
                    if y[i] == engine and features.get(feature) is not None:
                        try:
                            values.append(float(features[feature]))
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    mean = np.mean(values)
                    variance = np.var(values)
                    # Add small variance to avoid division by zero
                    variance = max(variance, 1e-6)
                else:
                    # Default values if no data
                    mean = 0.0
                    variance = 1.0
                
                self.numeric_stats[feature][engine] = {
                    "mean": float(mean),
                    "variance": float(variance)
                }
    
    def _estimate_categorical_likelihoods(self, X: List[Dict], y: List[str]):
        """Estimate categorical likelihoods with Laplace smoothing"""
        self.categorical_stats = {}
        
        for feature in self.features_used:
            if self._is_numeric_feature([x.get(feature) for x in X if x.get(feature) is not None]):
                continue
                
            self.categorical_stats[feature] = {}
            
            # Get all possible values for this feature
            all_values = set()
            for features in X:
                if features.get(feature) is not None:
                    all_values.add(str(features[feature]))
            
            for value in all_values:
                self.categorical_stats[feature][value] = {}
                
                for engine in self.engines:
                    # Count occurrences
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
    
    def predict(self, documents: List[Dict]) -> Dict[str, Any]:
        """Predict OCR engine for documents"""
        logger.info(f"Predicting OCR engines for {len(documents)} documents...")
        
        results = []
        
        for doc in documents:
            flat_doc = self._flatten_document(doc)
            
            # Compute posteriors for each engine
            posteriors = {}
            for engine in self.engines:
                posterior = self._compute_posterior(flat_doc, engine)
                posteriors[engine] = float(posterior)
            
            # Normalize posteriors
            total_posterior = sum(posteriors.values())
            if total_posterior > 0:
                for engine in posteriors:
                    posteriors[engine] /= total_posterior
            
            # Compute utility scores
            page_count = flat_doc.get('page_count', 1)
            normalized_latency = self._normalize_latency(page_count)
            normalized_resource = self._normalize_resource_cost()
            
            utility_scores = {}
            for engine in self.engines:
                utility = (self.alpha * posteriors[engine] - 
                          self.beta * normalized_latency[engine] - 
                          self.gamma * normalized_resource[engine])
                utility_scores[engine] = float(utility)
            
            # Select best engine
            chosen_engine = max(utility_scores, key=utility_scores.get)
            
            # Find fallback candidates
            fallback_candidates = []
            max_utility = utility_scores[chosen_engine]
            for engine, utility in utility_scores.items():
                if engine != chosen_engine and max_utility - utility <= self.delta_threshold:
                    fallback_candidates.append(engine)
            
            # Compute expected latency and confidence
            expected_latency = self.engine_latency_baseline[chosen_engine] * page_count
            expected_confidence = posteriors[chosen_engine]
            
            # Recommend preprocessing
            preprocessing = self._recommend_preprocessing(flat_doc)
            
            # Generate reason
            reason = self._generate_reason(flat_doc, chosen_engine, posteriors[chosen_engine])
            
            results.append({
                "document_id": doc.get('document_id', 'unknown'),
                "posteriors": posteriors,
                "normalized_latency": normalized_latency,
                "normalized_resource_cost": normalized_resource,
                "utility_scores": utility_scores,
                "chosen_engine": chosen_engine,
                "fallback_candidates": fallback_candidates,
                "expected_latency_sec": float(expected_latency),
                "expected_confidence": float(expected_confidence),
                "preprocessing_recommendation": preprocessing,
                "reason": reason
            })
        
        return results
    
    def _compute_posterior(self, features: Dict, engine: str) -> float:
        """Compute posterior probability P(engine | features) using Naive Bayes"""
        # Start with prior
        log_posterior = math.log(self.class_priors.get(engine, 1e-6))
        
        # Add likelihood terms
        for feature in self.features_used:
            value = features.get(feature)
            if value is None:
                continue
            
            # Numeric features
            if feature in self.numeric_stats:
                if engine in self.numeric_stats[feature]:
                    mean = self.numeric_stats[feature][engine]["mean"]
                    variance = self.numeric_stats[feature][engine]["variance"]
                    try:
                        x = float(value)
                        # Gaussian likelihood
                        log_likelihood = -0.5 * ((x - mean) ** 2) / variance - 0.5 * math.log(2 * math.pi * variance)
                        log_posterior += log_likelihood
                    except (ValueError, TypeError):
                        continue
            
            # Categorical features
            elif feature in self.categorical_stats:
                value_str = str(value)
                if value_str in self.categorical_stats[feature]:
                    if engine in self.categorical_stats[feature][value_str]:
                        prob = self.categorical_stats[feature][value_str][engine]
                        if prob > 0:
                            log_posterior += math.log(prob)
        
        return math.exp(log_posterior)
    
    def _normalize_latency(self, page_count: int) -> Dict[str, float]:
        """Normalize latency values to [0,1]"""
        latencies = {}
        for engine in self.engines:
            latencies[engine] = self.engine_latency_baseline[engine] * page_count
        
        min_lat = min(latencies.values())
        max_lat = max(latencies.values())
        
        if max_lat == min_lat:
            return {engine: 0.5 for engine in self.engines}
        
        normalized = {}
        for engine in self.engines:
            normalized[engine] = (latencies[engine] - min_lat) / (max_lat - min_lat)
        
        return normalized
    
    def _normalize_resource_cost(self) -> Dict[str, float]:
        """Normalize resource costs to [0,1]"""
        costs = self.engine_resource_baseline.copy()
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        
        if max_cost == min_cost:
            return {engine: 0.5 for engine in self.engines}
        
        normalized = {}
        for engine in self.engines:
            normalized[engine] = (costs[engine] - min_cost) / (max_cost - min_cost)
        
        return normalized
    
    def _recommend_preprocessing(self, features: Dict) -> List[str]:
        """Recommend preprocessing steps based on document features"""
        recommendations = []
        
        # Check for skew (simplified heuristic)
        if features.get('aspect_ratio', 1.0) < 0.5 or features.get('aspect_ratio', 1.0) > 2.0:
            recommendations.append("deskew")
        
        # Check for low resolution
        if features.get('text_density', 0) < 100:
            recommendations.append("DPI_increase")
        
        # Check for noise
        if features.get('noise_level', 0) > 0.3:
            recommendations.append("denoise")
        
        # Check for tables
        if features.get('has_tables', False):
            recommendations.append("table_extraction")
        
        # Check for signatures
        if features.get('has_annotations', False):
            recommendations.append("signature_crop")
        
        return recommendations
    
    def _generate_reason(self, features: Dict, chosen_engine: str, confidence: float) -> str:
        """Generate explanation for the chosen engine"""
        reasons = []
        
        if confidence > 0.7:
            reasons.append("high confidence")
        elif confidence > 0.4:
            reasons.append("moderate confidence")
        else:
            reasons.append("low confidence")
        
        # Add feature-based reasons
        if features.get('has_tables', False):
            reasons.append("contains tables")
        
        if features.get('text_density', 0) > 500:
            reasons.append("high text density")
        
        if features.get('has_forms', False):
            reasons.append("contains forms")
        
        return f"Selected {chosen_engine} due to {', '.join(reasons)}"

def load_training_data(ml_dataset_file: str, metadata_file: str) -> List[Dict]:
    """Load and combine training data from both files"""
    logger.info("Loading training data...")
    
    # Load ML dataset
    with open(ml_dataset_file, 'r', encoding='utf-8') as f:
        ml_data = json.load(f)
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Combine data
    combined_data = []
    
    # Use ML dataset as primary source
    for doc in ml_data:
        combined_data.append(doc)
    
    # Add any additional documents from metadata that aren't in ML dataset
    ml_doc_ids = {doc['document_id'] for doc in ml_data}
    
    for doc in metadata.get('document_features', []):
        if doc.get('document_id') not in ml_doc_ids:
            # Convert metadata format to ML format
            converted_doc = convert_metadata_to_ml_format(doc)
            if converted_doc:
                combined_data.append(converted_doc)
    
    logger.info(f"Loaded {len(combined_data)} training documents")
    return combined_data

def convert_metadata_to_ml_format(doc: Dict) -> Dict:
    """Convert metadata format to ML dataset format"""
    try:
        converted = {
            'document_id': doc.get('document_id', ''),
            'file_name': doc.get('file_name', ''),
            'page_count': doc.get('document_structure', {}).get('page_count', 1),
            'has_metadata': doc.get('document_structure', {}).get('has_metadata', False),
            'has_forms': doc.get('document_structure', {}).get('has_forms', False),
            'has_annotations': doc.get('document_structure', {}).get('has_annotations', False),
            'total_characters': doc.get('text_content', {}).get('total_characters', 0),
            'total_words': doc.get('text_content', {}).get('total_words', 0),
            'unique_fonts': doc.get('text_content', {}).get('unique_fonts', 0),
            'has_tables': doc.get('text_content', {}).get('has_tables', False),
            'has_numbers': doc.get('text_content', {}).get('has_numbers', False),
            'has_currency': doc.get('text_content', {}).get('has_currency', False),
            'has_dates': doc.get('text_content', {}).get('has_dates', False),
            'has_emails': doc.get('text_content', {}).get('has_emails', False),
            'has_phone_numbers': doc.get('text_content', {}).get('has_phone_numbers', False),
            'aspect_ratio': doc.get('visual_features', {}).get('aspect_ratio', 1.0),
            'brightness_mean': doc.get('visual_features', {}).get('brightness_mean', 128.0),
            'contrast': doc.get('visual_features', {}).get('contrast', 50.0),
            'has_images': doc.get('layout_features', {}).get('has_images', False),
            'has_graphics': doc.get('layout_features', {}).get('has_graphics', False),
            'column_count': doc.get('layout_features', {}).get('column_count', 1),
            'text_density': doc.get('ocr_features', {}).get('text_density', 0.0),
            'font_clarity': doc.get('ocr_features', {}).get('font_clarity', 0.0),
            'noise_level': doc.get('ocr_features', {}).get('noise_level', 0.0),
            'recommended_ocr_engine': doc.get('ocr_features', {}).get('recommended_ocr_engine', 'tesseract')
        }
        return converted
    except Exception as e:
        logger.error(f"Error converting document: {e}")
        return None

def create_ocr_routing_response(training_data: List[Dict], test_documents: List[Dict], 
                              alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5,
                              delta_threshold: float = 0.03) -> Dict[str, Any]:
    """Create the complete OCR routing response"""
    
    # Initialize router
    router = NaiveBayesOCRRouter(alpha, beta, gamma, delta_threshold)
    
    # Train model
    model_summary = router.train(training_data)
    
    # Make predictions
    predictions = router.predict(test_documents)
    
    # Create response
    response = {
        "metadata": {
            "algorithm": "naive_bayes_with_utility",
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta_fallback_threshold": delta_threshold,
            "engine_latency_baseline": router.engine_latency_baseline,
            "engine_resource_baseline": router.engine_resource_baseline,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z"
        },
        "features_used": model_summary["features_used"],
        "model_summary": {
            "class_priors": model_summary["class_priors"],
            "numeric_stats": model_summary["numeric_stats"],
            "categorical_stats": model_summary["categorical_stats"]
        },
        "documents": predictions,
        "evaluation_recommendations": {
            "min_samples_per_engine": 50,
            "cross_validation": "k_fold",
            "k": 5,
            "metrics": ["accuracy", "precision", "recall", "f1", "latency_ms_p95"]
        },
        "notes": f"Model trained on {len(training_data)} documents. Using {len(model_summary['features_used'])} features. Consider collecting more samples per engine for better performance."
    }
    
    return response

def main():
    """Main function to demonstrate the OCR routing system"""
    print("=" * 80)
    print("NAIVE BAYES OCR ROUTING DISPATCHER")
    print("=" * 80)
    
    # Load training data
    training_data = load_training_data("ml_ocr_routing_dataset.json", "fixed_fast_pdf_metadata.json")
    
    # Create test documents (use first 10 from training data)
    test_documents = training_data[:10]
    
    # Create routing response
    response = create_ocr_routing_response(
        training_data=training_data,
        test_documents=test_documents,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta_threshold=0.03
    )
    
    # Save response
    with open("ocr_routing_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)
    
    print("OCR Routing Response Generated!")
    print(f"Features used: {len(response['features_used'])}")
    print(f"Documents processed: {len(response['documents'])}")
    print(f"Engines available: {list(response['metadata']['engine_latency_baseline'].keys())}")
    print("\nSample predictions:")
    for i, doc in enumerate(response['documents'][:3]):
        print(f"  {i+1}. {doc['document_id']} -> {doc['chosen_engine']} (confidence: {doc['expected_confidence']:.3f})")
    
    print(f"\nResponse saved to: ocr_routing_response.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
