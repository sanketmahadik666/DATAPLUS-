#!/usr/bin/env python3
"""
OCR Routing Microservice
FastAPI service for Naive Bayes-based OCR engine routing
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ocr_routing_dispatcher import NaiveBayesOCRRouter, load_training_data, create_ocr_routing_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Routing Microservice",
    description="Naive Bayes-based OCR engine routing with utility function optimization",
    version="1.0.0"
)

# Global router instance
router_instance = None

class DocumentFeatures(BaseModel):
    """Document features for OCR routing"""
    document_id: str
    file_name: Optional[str] = None
    page_count: int = 1
    has_metadata: bool = False
    has_forms: bool = False
    has_annotations: bool = False
    total_characters: int = 0
    total_words: int = 0
    unique_fonts: int = 0
    has_tables: bool = False
    has_numbers: bool = False
    has_currency: bool = False
    has_dates: bool = False
    has_emails: bool = False
    has_phone_numbers: bool = False
    aspect_ratio: float = 1.0
    brightness_mean: float = 128.0
    contrast: float = 50.0
    has_images: bool = False
    has_graphics: bool = False
    column_count: int = 1
    text_density: float = 0.0
    font_clarity: float = 0.0
    noise_level: float = 0.0
    recommended_ocr_engine: Optional[str] = None

class RoutingRequest(BaseModel):
    """Request model for OCR routing"""
    document_features: List[DocumentFeatures]
    alpha: float = Field(default=1.0, ge=0.0, le=2.0, description="Weight for accuracy")
    beta: float = Field(default=0.5, ge=0.0, le=2.0, description="Weight for latency")
    gamma: float = Field(default=0.5, ge=0.0, le=2.0, description="Weight for resource cost")
    delta_fallback_threshold: float = Field(default=0.03, ge=0.0, le=0.1, description="Fallback threshold")

class RoutingResponse(BaseModel):
    """Response model for OCR routing"""
    metadata: Dict[str, Any]
    features_used: List[str]
    model_summary: Dict[str, Any]
    documents: List[Dict[str, Any]]
    evaluation_recommendations: Dict[str, Any]
    notes: str

@app.on_event("startup")
async def startup_event():
    """Initialize the OCR router on startup"""
    global router_instance
    try:
        logger.info("Loading training data and initializing OCR router...")
        training_data = load_training_data("ml_ocr_routing_dataset.json", "fixed_fast_pdf_metadata.json")
        router_instance = NaiveBayesOCRRouter()
        router_instance.train(training_data)
        logger.info(f"OCR router initialized with {len(training_data)} training documents")
    except Exception as e:
        logger.error(f"Failed to initialize OCR router: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "OCR Routing Microservice",
        "version": "1.0.0"
    }

@app.get("/engines")
async def get_available_engines():
    """Get list of available OCR engines"""
    if router_instance is None:
        raise HTTPException(status_code=503, detail="OCR router not initialized")
    
    return {
        "engines": list(router_instance.engine_latency_baseline.keys()),
        "latency_baseline": router_instance.engine_latency_baseline,
        "resource_baseline": router_instance.engine_resource_baseline
    }

@app.post("/route", response_model=RoutingResponse)
async def route_documents(request: RoutingRequest):
    """Route documents to optimal OCR engines"""
    if router_instance is None:
        raise HTTPException(status_code=503, detail="OCR router not initialized")
    
    try:
        # Convert request to document format
        documents = []
        for doc_feat in request.document_features:
            doc_dict = doc_feat.dict()
            # Remove None values
            doc_dict = {k: v for k, v in doc_dict.items() if v is not None}
            documents.append(doc_dict)
        
        # Make predictions
        predictions = router_instance.predict(documents)
        
        # Create response
        response = {
            "metadata": {
                "algorithm": "naive_bayes_with_utility",
                "alpha": request.alpha,
                "beta": request.beta,
                "gamma": request.gamma,
                "delta_fallback_threshold": request.delta_fallback_threshold,
                "engine_latency_baseline": router_instance.engine_latency_baseline,
                "engine_resource_baseline": router_instance.engine_resource_baseline,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z"
            },
            "features_used": router_instance.features_used,
            "model_summary": {
                "class_priors": router_instance.class_priors,
                "numeric_stats": router_instance.numeric_stats,
                "categorical_stats": router_instance.categorical_stats
            },
            "documents": predictions,
            "evaluation_recommendations": {
                "min_samples_per_engine": 50,
                "cross_validation": "k_fold",
                "k": 5,
                "metrics": ["accuracy", "precision", "recall", "f1", "latency_ms_p95"]
            },
            "notes": f"Model trained on {len(router_instance.features_used)} features. Using utility function with α={request.alpha}, β={request.beta}, γ={request.gamma}."
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error routing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/route-batch")
async def route_documents_batch(request: RoutingRequest):
    """Route documents in batch with detailed response"""
    if router_instance is None:
        raise HTTPException(status_code=503, detail="OCR router not initialized")
    
    try:
        # Convert request to document format
        documents = []
        for doc_feat in request.document_features:
            doc_dict = doc_feat.dict()
            # Remove None values
            doc_dict = {k: v for k, v in doc_dict.items() if v is not None}
            documents.append(doc_dict)
        
        # Make predictions
        predictions = router_instance.predict(documents)
        
        # Create detailed response
        response = {
            "metadata": {
                "algorithm": "naive_bayes_with_utility",
                "alpha": request.alpha,
                "beta": request.beta,
                "gamma": request.gamma,
                "delta_fallback_threshold": request.delta_fallback_threshold,
                "engine_latency_baseline": router_instance.engine_latency_baseline,
                "engine_resource_baseline": router_instance.engine_resource_baseline,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "total_documents": len(documents),
                "processing_time_ms": 0  # Could add timing if needed
            },
            "features_used": router_instance.features_used,
            "model_summary": {
                "class_priors": router_instance.class_priors,
                "numeric_stats": router_instance.numeric_stats,
                "categorical_stats": router_instance.categorical_stats
            },
            "documents": predictions,
            "summary": {
                "engine_distribution": {},
                "average_confidence": 0.0,
                "average_latency": 0.0
            },
            "evaluation_recommendations": {
                "min_samples_per_engine": 50,
                "cross_validation": "k_fold",
                "k": 5,
                "metrics": ["accuracy", "precision", "recall", "f1", "latency_ms_p95"]
            },
            "notes": f"Batch processing of {len(documents)} documents using {len(router_instance.features_used)} features."
        }
        
        # Calculate summary statistics
        engine_counts = {}
        total_confidence = 0.0
        total_latency = 0.0
        
        for doc in predictions:
            engine = doc["chosen_engine"]
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
            total_confidence += doc["expected_confidence"]
            total_latency += doc["expected_latency_sec"]
        
        response["summary"]["engine_distribution"] = engine_counts
        response["summary"]["average_confidence"] = total_confidence / len(predictions) if predictions else 0.0
        response["summary"]["average_latency"] = total_latency / len(predictions) if predictions else 0.0
        
        return response
        
    except Exception as e:
        logger.error(f"Error routing documents batch: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if router_instance is None:
        raise HTTPException(status_code=503, detail="OCR router not initialized")
    
    return {
        "features_used": router_instance.features_used,
        "engines": list(router_instance.engine_latency_baseline.keys()),
        "class_priors": router_instance.class_priors,
        "numeric_features": list(router_instance.numeric_stats.keys()),
        "categorical_features": list(router_instance.categorical_stats.keys()),
        "total_features": len(router_instance.features_used)
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with current data"""
    global router_instance
    try:
        logger.info("Retraining OCR router...")
        training_data = load_training_data("ml_ocr_routing_dataset.json", "fixed_fast_pdf_metadata.json")
        router_instance = NaiveBayesOCRRouter()
        router_instance.train(training_data)
        logger.info(f"OCR router retrained with {len(training_data)} documents")
        
        return {
            "status": "success",
            "message": f"Model retrained with {len(training_data)} documents",
            "features_used": len(router_instance.features_used),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "ocr_routing_service:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
