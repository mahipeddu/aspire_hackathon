#!/usr/bin/env python3
"""
Metrics Collection Module for Prometheus
Handles application metrics and monitoring
"""

import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter(
    'llama_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code', 'username']
)

request_duration = Histogram(
    'llama_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint', 'username'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)
)

generation_duration = Histogram(
    'llama_generation_duration_seconds',
    'Text generation duration in seconds',
    ['model_type'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
)

generation_tokens = Counter(
    'llama_generation_tokens_total',
    'Total number of tokens generated',
    ['model_type']
)

generation_speed = Gauge(
    'llama_generation_tokens_per_second',
    'Tokens generated per second',
    ['model_type']
)

memory_usage = Gauge(
    'llama_memory_usage_bytes',
    'Memory usage in bytes',
    ['type']  # 'ram' or 'vram'
)

model_memory = Gauge(
    'llama_model_memory_usage_bytes',
    'Model memory usage in bytes',
    ['model_type']
)

active_requests = Gauge(
    'llama_active_requests',
    'Number of currently active requests'
)

error_count = Counter(
    'llama_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

authentication_attempts = Counter(
    'llama_auth_attempts_total',
    'Total number of authentication attempts',
    ['status']  # 'success' or 'failure'
)

rate_limit_hits = Counter(
    'llama_rate_limit_hits_total',
    'Total number of rate limit hits',
    ['username', 'endpoint']
)

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.model_info = {}
    
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float, username: Optional[str] = None):
        """Record an API request"""
        username = username or "anonymous"
        
        request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            username=username
        ).inc()
        
        request_duration.labels(
            method=method,
            endpoint=endpoint,
            username=username
        ).observe(duration)
    
    def record_generation(self, model_type: str, duration: float, 
                         token_count: int):
        """Record text generation metrics"""
        generation_duration.labels(model_type=model_type).observe(duration)
        generation_tokens.labels(model_type=model_type).inc(token_count)
        
        if duration > 0:
            tokens_per_second = token_count / duration
            generation_speed.labels(model_type=model_type).set(tokens_per_second)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record an error occurrence"""
        error_count.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def record_auth_attempt(self, success: bool):
        """Record authentication attempt"""
        status = "success" if success else "failure"
        authentication_attempts.labels(status=status).inc()
    
    def record_rate_limit_hit(self, username: str, endpoint: str):
        """Record rate limit hit"""
        rate_limit_hits.labels(username=username, endpoint=endpoint).inc()
    
    def update_memory_metrics(self):
        """Update memory usage metrics"""
        try:
            # RAM usage
            ram_bytes = psutil.virtual_memory().used
            memory_usage.labels(type="ram").set(ram_bytes)
            
            # GPU memory usage
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.memory_allocated()
                memory_usage.labels(type="vram").set(vram_bytes)
        except Exception as e:
            logger.warning(f"Failed to update memory metrics: {e}")
    
    def set_model_memory(self, model_type: str, memory_bytes: int):
        """Set model memory usage"""
        model_memory.labels(model_type=model_type).set(memory_bytes)
    
    def increment_active_requests(self):
        """Increment active request counter"""
        active_requests.inc()
    
    def decrement_active_requests(self):
        """Decrement active request counter"""
        active_requests.dec()
    
    def get_metrics_summary(self) -> dict:
        """Get a summary of current metrics"""
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": round(uptime, 2),
            "memory": {},
            "requests": {},
            "generation": {}
        }
        
        try:
            # Memory info
            ram_gb = psutil.virtual_memory().used / (1024**3)
            summary["memory"]["ram_gb"] = round(ram_gb, 2)
            
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                summary["memory"]["vram_gb"] = round(vram_gb, 2)
                summary["memory"]["gpu_available"] = True
            else:
                summary["memory"]["gpu_available"] = False
            
            # System info
            summary["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "cpu_count": psutil.cpu_count(),
                "memory_percent": psutil.virtual_memory().percent
            }
            
        except Exception as e:
            logger.warning(f"Failed to get metrics summary: {e}")
        
        return summary

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_prometheus_metrics():
    """Generate Prometheus metrics in text format"""
    # Update memory metrics before generating
    metrics_collector.update_memory_metrics()
    return generate_latest()

class MetricsMiddleware:
    """Middleware to automatically collect request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        metrics_collector.increment_active_requests()
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        
        try:
            await self.app(scope, receive, send)
        finally:
            duration = time.time() - start_time
            metrics_collector.decrement_active_requests()
            
            # Note: We can't easily get status code here without more complex middleware
            # This is a simplified version
            metrics_collector.record_request(method, path, 200, duration)
