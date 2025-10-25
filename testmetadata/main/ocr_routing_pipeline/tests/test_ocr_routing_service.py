#!/usr/bin/env python3
"""
Test client for OCR Routing Microservice
Demonstrates the API endpoints and functionality
"""

import json
import requests
import time
from typing import Dict, List, Any

class OCRRoutingClient:
    """Client for OCR Routing Microservice"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_engines(self) -> Dict[str, Any]:
        """Get available OCR engines"""
        response = self.session.get(f"{self.base_url}/engines")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def route_documents(self, documents: List[Dict], alpha: float = 1.0, 
                       beta: float = 0.5, gamma: float = 0.5, 
                       delta_threshold: float = 0.03) -> Dict[str, Any]:
        """Route documents to OCR engines"""
        payload = {
            "document_features": documents,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta_fallback_threshold": delta_threshold
        }
        
        response = self.session.post(f"{self.base_url}/route", json=payload)
        response.raise_for_status()
        return response.json()
    
    def route_documents_batch(self, documents: List[Dict], alpha: float = 1.0, 
                             beta: float = 0.5, gamma: float = 0.5, 
                             delta_threshold: float = 0.03) -> Dict[str, Any]:
        """Route documents in batch"""
        payload = {
            "document_features": documents,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta_fallback_threshold": delta_threshold
        }
        
        response = self.session.post(f"{self.base_url}/route-batch", json=payload)
        response.raise_for_status()
        return response.json()
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retrain the model"""
        response = self.session.post(f"{self.base_url}/retrain")
        response.raise_for_status()
        return response.json()

def create_sample_documents() -> List[Dict]:
    """Create sample documents for testing"""
    return [
        {
            "document_id": "test_invoice_001",
            "file_name": "invoice_sample_001.pdf",
            "page_count": 1,
            "has_metadata": True,
            "has_forms": True,
            "has_annotations": False,
            "total_characters": 450,
            "total_words": 65,
            "unique_fonts": 2,
            "has_tables": False,
            "has_numbers": True,
            "has_currency": True,
            "has_dates": True,
            "has_emails": False,
            "has_phone_numbers": True,
            "aspect_ratio": 0.77,
            "brightness_mean": 125.0,
            "contrast": 75.0,
            "has_images": False,
            "has_graphics": True,
            "column_count": 3,
            "text_density": 850.0,
            "font_clarity": 0.8,
            "noise_level": 0.1
        },
        {
            "document_id": "test_form_002",
            "file_name": "form_sample_002.pdf",
            "page_count": 2,
            "has_metadata": True,
            "has_forms": True,
            "has_annotations": True,
            "total_characters": 800,
            "total_words": 120,
            "unique_fonts": 3,
            "has_tables": True,
            "has_numbers": True,
            "has_currency": False,
            "has_dates": True,
            "has_emails": True,
            "has_phone_numbers": True,
            "aspect_ratio": 0.77,
            "brightness_mean": 130.0,
            "contrast": 80.0,
            "has_images": True,
            "has_graphics": False,
            "column_count": 2,
            "text_density": 600.0,
            "font_clarity": 0.9,
            "noise_level": 0.05
        },
        {
            "document_id": "test_receipt_003",
            "file_name": "receipt_sample_003.pdf",
            "page_count": 1,
            "has_metadata": False,
            "has_forms": False,
            "has_annotations": False,
            "total_characters": 200,
            "total_words": 35,
            "unique_fonts": 1,
            "has_tables": False,
            "has_numbers": True,
            "has_currency": True,
            "has_dates": True,
            "has_emails": False,
            "has_phone_numbers": False,
            "aspect_ratio": 0.6,
            "brightness_mean": 140.0,
            "contrast": 90.0,
            "has_images": False,
            "has_graphics": False,
            "column_count": 1,
            "text_density": 300.0,
            "font_clarity": 0.7,
            "noise_level": 0.2
        }
    ]

def test_service():
    """Test the OCR routing service"""
    print("=" * 80)
    print("TESTING OCR ROUTING MICROSERVICE")
    print("=" * 80)
    
    client = OCRRoutingClient()
    
    try:
        # Test health check
        print("1. Testing health check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Service: {health['service']}")
        print(f"   Version: {health['version']}")
        
        # Test available engines
        print("\n2. Testing available engines...")
        engines = client.get_engines()
        print(f"   Available engines: {engines['engines']}")
        print(f"   Latency baseline: {engines['latency_baseline']}")
        
        # Test model info
        print("\n3. Testing model information...")
        model_info = client.get_model_info()
        print(f"   Features used: {len(model_info['features_used'])}")
        print(f"   Engines: {model_info['engines']}")
        print(f"   Numeric features: {len(model_info['numeric_features'])}")
        print(f"   Categorical features: {len(model_info['categorical_features'])}")
        
        # Test document routing
        print("\n4. Testing document routing...")
        sample_docs = create_sample_documents()
        
        start_time = time.time()
        routing_result = client.route_documents(sample_docs)
        end_time = time.time()
        
        print(f"   Processing time: {(end_time - start_time)*1000:.2f}ms")
        print(f"   Documents processed: {len(routing_result['documents'])}")
        print(f"   Features used: {len(routing_result['features_used'])}")
        
        print("\n   Routing results:")
        for i, doc in enumerate(routing_result['documents']):
            print(f"     {i+1}. {doc['document_id']} -> {doc['chosen_engine']}")
            print(f"        Confidence: {doc['expected_confidence']:.3f}")
            print(f"        Expected latency: {doc['expected_latency_sec']:.2f}s")
            print(f"        Preprocessing: {doc['preprocessing_recommendation']}")
            print(f"        Reason: {doc['reason']}")
        
        # Test batch routing
        print("\n5. Testing batch routing...")
        batch_result = client.route_documents_batch(sample_docs)
        
        print(f"   Total documents: {batch_result['metadata']['total_documents']}")
        print(f"   Engine distribution: {batch_result['summary']['engine_distribution']}")
        print(f"   Average confidence: {batch_result['summary']['average_confidence']:.3f}")
        print(f"   Average latency: {batch_result['summary']['average_latency']:.2f}s")
        
        # Test different utility parameters
        print("\n6. Testing different utility parameters...")
        
        # High accuracy preference
        accuracy_result = client.route_documents(sample_docs, alpha=2.0, beta=0.1, gamma=0.1)
        print(f"   High accuracy preference:")
        for doc in accuracy_result['documents']:
            print(f"     {doc['document_id']} -> {doc['chosen_engine']} (confidence: {doc['expected_confidence']:.3f})")
        
        # High speed preference
        speed_result = client.route_documents(sample_docs, alpha=0.5, beta=2.0, gamma=0.1)
        print(f"   High speed preference:")
        for doc in speed_result['documents']:
            print(f"     {doc['document_id']} -> {doc['chosen_engine']} (latency: {doc['expected_latency_sec']:.2f}s)")
        
        # Low resource preference
        resource_result = client.route_documents(sample_docs, alpha=0.5, beta=0.1, gamma=2.0)
        print(f"   Low resource preference:")
        for doc in resource_result['documents']:
            print(f"     {doc['document_id']} -> {doc['chosen_engine']}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the OCR routing service.")
        print("Make sure the service is running on http://localhost:8002")
        print("Start the service with: python ocr_routing_service.py")
    except Exception as e:
        print(f"ERROR: {e}")

def create_test_data_file():
    """Create a test data file for the service"""
    sample_docs = create_sample_documents()
    
    test_data = {
        "document_features": sample_docs,
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.5,
        "delta_fallback_threshold": 0.03
    }
    
    with open("test_routing_request.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("Test data saved to: test_routing_request.json")

if __name__ == "__main__":
    print("OCR Routing Service Test Client")
    print("1. Test service (requires service to be running)")
    print("2. Create test data file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_service()
    elif choice == "2":
        create_test_data_file()
    else:
        print("Invalid choice")
