#!/usr/bin/env python3
"""
High-Performance Document Analyzer Microservice
Optimized for SLO targets with parallel processing, chunking, and caching
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import hashlib
import tempfile
from collections import defaultdict
import threading
from queue import Queue, Empty
import weakref

# FastAPI and async imports
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Analysis imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from service_analyzer import analyze_path, discover_inputs, convert_pdf_to_images, analyze_file
from document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SLOTargets:
    """SLO targets for the microservice"""
    p50_latency: float = 2.0      # seconds
    p95_latency: float = 8.0      # seconds
    p99_latency: float = 15.0     # seconds
    max_latency: float = 30.0     # seconds
    target_throughput: int = 100  # docs/minute
    max_error_rate: float = 0.001 # 0.1%
    min_accuracy: float = 0.95   # 95%
    min_cache_hit_rate: float = 0.8  # 80%

@dataclass
class ProcessingConfig:
    """Configuration for parallel processing"""
    max_threads: int = 64
    max_processes: int = 8
    chunk_size: int = 32
    max_queue_size: int = 1000
    cache_size_mb: int = 512
    temp_dir: str = ".processing_tmp"

class DocumentCache:
    """High-performance document cache with LRU eviction"""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _evict_lru(self):
        """Evict least recently used items"""
        with self.lock:
            if not self.cache:
                return
            
            # Find LRU item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            if lru_key in self.cache:
                item_size = len(str(self.cache[lru_key]))
                del self.cache[lru_key]
                del self.access_times[lru_key]
                self.current_size -= item_size
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached result"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Dict):
        """Cache result"""
        with self.lock:
            item_size = len(str(value))
            
            # Evict if needed
            while self.current_size + item_size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.current_size += item_size
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size = 0

class MetricsCollector:
    """Collect and track SLO metrics"""
    
    def __init__(self):
        self.latencies = []
        self.error_count = 0
        self.success_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_latency(self, latency: float):
        """Record processing latency"""
        with self.lock:
            self.latencies.append(latency)
            # Keep only last 1000 measurements
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]
    
    def record_success(self):
        """Record successful processing"""
        with self.lock:
            self.success_count += 1
    
    def record_error(self):
        """Record processing error"""
        with self.lock:
            self.error_count += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        with self.lock:
            self.cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            total_requests = self.success_count + self.error_count
            error_rate = self.error_count / total_requests if total_requests > 0 else 0
            
            cache_total = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0
            
            latencies = sorted(self.latencies)
            p50 = latencies[len(latencies)//2] if latencies else 0
            p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
            p99 = latencies[int(len(latencies)*0.99)] if latencies else 0
            
            uptime = time.time() - self.start_time
            
            return {
                "uptime_seconds": uptime,
                "total_requests": total_requests,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "cache_hit_rate": cache_hit_rate,
                "latency_p50": p50,
                "latency_p95": p95,
                "latency_p99": p99,
                "throughput_per_minute": (total_requests / uptime) * 60 if uptime > 0 else 0
            }

class OptimizedDocumentAnalyzer:
    """High-performance document analyzer with parallel processing"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cache = DocumentCache(config.cache_size_mb)
        self.metrics = MetricsCollector()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_processes)
        self.analyzer = DocumentAnalyzer()
        
        # Create temp directory
        Path(config.temp_dir).mkdir(exist_ok=True)
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file"""
        stat = file_path.stat()
        return hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    
    async def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file with caching and metrics"""
        start_time = time.time()
        cache_key = self._get_cache_key(file_path)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.metrics.record_cache_hit()
            return cached_result
        
        self.metrics.record_cache_miss()
        
        try:
            # Process file
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, analyze_file, self.analyzer, file_path
            )
            
            # Cache result
            self.cache.put(cache_key, result)
            self.metrics.record_success()
            
            return result
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "document_id": file_path.stem,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "processing_status": "error",
                "error_message": str(e)
            }
        finally:
            latency = time.time() - start_time
            self.metrics.record_latency(latency)
    
    async def _process_chunk(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process a chunk of files in parallel"""
        tasks = [self._process_single_file(f) for f in files]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def analyze_path_optimized(self, input_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Optimized path analysis with streaming results"""
        target = Path(input_path)
        files = discover_inputs(target)
        
        if not files:
            yield {
                "summary": {"total_inputs": 0, "successful": 0, "failed": 0},
                "results": [],
                "metadata": {"input_path": str(target.resolve())}
            }
            return
        
        # Process in chunks
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        for i in range(0, len(files), self.config.chunk_size):
            chunk = files[i:i + self.config.chunk_size]
            
            # Process chunk
            chunk_results = await self._process_chunk(chunk)
            
            # Filter out exceptions
            valid_results = []
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Chunk processing error: {result}")
                    continue
                valid_results.append(result)
            
            # Update counters
            for result in valid_results:
                total_processed += 1
                if result.get('processing_status') == 'success':
                    total_successful += 1
                else:
                    total_failed += 1
            
            # Yield partial results
            yield {
                "chunk_index": i // self.config.chunk_size,
                "chunk_size": len(chunk),
                "total_processed": total_processed,
                "total_files": len(files),
                "results": valid_results,
                "metrics": self.metrics.get_metrics()
            }
        
        # Final summary
        yield {
            "summary": {
                "total_inputs": len(files),
                "successful": total_successful,
                "failed": total_failed,
                "processed_at": time.time()
            },
            "metadata": {
                "input_path": str(target.resolve()),
                "processing_config": {
                    "chunk_size": self.config.chunk_size,
                    "max_threads": self.config.max_threads,
                    "max_processes": self.config.max_processes
                }
            },
            "metrics": self.metrics.get_metrics()
        }

# FastAPI Application
app = FastAPI(
    title="Optimized Document Analyzer Service",
    version="2.0.0",
    description="High-performance document analysis with SLO targets"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = OptimizedDocumentAnalyzer(ProcessingConfig())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    metrics = analyzer.metrics.get_metrics()
    return {
        "status": "healthy",
        "uptime": metrics["uptime_seconds"],
        "metrics": metrics
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics"""
    return analyzer.metrics.get_metrics()

@app.get("/slo-status")
async def slo_status():
    """Check SLO compliance"""
    metrics = analyzer.metrics.get_metrics()
    slo = SLOTargets()
    
    status = {
        "latency_p50": {
            "current": metrics["latency_p50"],
            "target": slo.p50_latency,
            "compliant": metrics["latency_p50"] <= slo.p50_latency
        },
        "latency_p95": {
            "current": metrics["latency_p95"],
            "target": slo.p95_latency,
            "compliant": metrics["latency_p95"] <= slo.p95_latency
        },
        "error_rate": {
            "current": metrics["error_rate"],
            "target": slo.max_error_rate,
            "compliant": metrics["error_rate"] <= slo.max_error_rate
        },
        "cache_hit_rate": {
            "current": metrics["cache_hit_rate"],
            "target": slo.min_cache_hit_rate,
            "compliant": metrics["cache_hit_rate"] >= slo.min_cache_hit_rate
        },
        "overall_compliant": (
            metrics["latency_p50"] <= slo.p50_latency and
            metrics["latency_p95"] <= slo.p95_latency and
            metrics["error_rate"] <= slo.max_error_rate and
            metrics["cache_hit_rate"] >= slo.min_cache_hit_rate
        )
    }
    
    return status

@app.post("/analyze-path-stream")
async def analyze_path_streaming(path: str):
    """Streaming analysis endpoint with real-time results"""
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    
    async def generate():
        async for result in analyzer.analyze_path_optimized(path):
            yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/analyze-path")
async def analyze_path_batch(path: str):
    """Batch analysis endpoint - returns complete results"""
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    
    results = []
    async for result in analyzer.analyze_path_optimized(path):
        if "results" in result:
            results.extend(result["results"])
        if "summary" in result:
            return {
                "summary": result["summary"],
                "results": results,
                "metrics": analyzer.metrics.get_metrics()
            }
    
    return {"error": "No results generated"}

@app.post("/clear-cache")
async def clear_cache():
    """Clear the document cache"""
    analyzer.cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/cache-stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size_mb": analyzer.cache.current_size / (1024 * 1024),
        "max_size_mb": analyzer.cache.max_size_bytes / (1024 * 1024),
        "hit_rate": analyzer.metrics.get_metrics()["cache_hit_rate"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "optimized_service:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use single worker with async for better control
        loop="asyncio"
    )
