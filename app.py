#!/usr/bin/env python3
"""
Production-ready FastAPI Web Service for Llama 2 7B
Includes security, scalability, and observability features
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from auth import (
    authenticate_user, create_access_token, SecurityValidator,
    check_rate_limit, get_auth_user, LoginRequest, TokenResponse
)
from metrics import metrics_collector, get_prometheus_metrics, CONTENT_TYPE_LATEST
from utils import (
    setup_logging, log_request, log_response, log_generation,
    get_system_info, ModelLoader, create_error_response, validate_environment
)

# Setup logging
setup_logging("INFO")
logger = __import__('logging').getLogger(__name__)

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt to generate from")
    max_tokens: int = Field(default=100, ge=1, le=500, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    model_type: str = Field(default="8bit", pattern="^(original|8bit|4bit)$", description="Model quantization type")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    model_type: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    request_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: float
    system_info: Dict[str, Any]
    available_models: List[str]

class ModelInfoResponse(BaseModel):
    models: Dict[str, Any]
    system_info: Dict[str, Any]

# Global model loader
model_loader = ModelLoader()

# Security dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
) -> Dict[str, Any]:
    """Get current authenticated user from Bearer token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from Bearer scheme
    token = credentials.credentials
    
    # Validate token
    from auth import verify_token
    user = verify_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Llama 2 7B API Service")
    
    # Validate environment
    issues = validate_environment()
    if issues:
        logger.warning(f"Environment issues detected: {issues}")
    
    # Pre-load default model
    try:
        available_models = model_loader.list_available_models()
        logger.info(f"Available models: {available_models}")
        
        if available_models:
            # Try to load 8bit model by default
            default_model = None
            for model_name in ["llama2-7b-8bit", "llama2-7b-4bit", "llama2-7b-original"]:
                if model_name in available_models:
                    default_model = model_name
                    break
            
            if default_model:
                logger.info(f"Pre-loading default model: {default_model}")
                model_loader.load_model(default_model)
        else:
            logger.warning("No models found. Run model_quantization.py first.")
    
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
    
    yield
    
    logger.info("Shutting down Llama 2 7B API Service")

# Create FastAPI app
app = FastAPI(
    title="Llama 2 7B Production API",
    description="Production-ready API for Llama 2 7B with quantization support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware for request/response logging and metrics"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Get user info for logging
    user_info = get_auth_user(request)
    username = user_info.get("username") if user_info else None
    
    # Check rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, username):
        metrics_collector.record_rate_limit_hit(username or client_ip, str(request.url.path))
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Rate limit exceeded"}
        )
    
    # Log request
    log_request(
        request_id=request_id,
        method=request.method,
        path=str(request.url.path),
        user=username,
        client_ip=client_ip
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    log_response(
        request_id=request_id,
        status_code=response.status_code,
        duration=duration,
        user=username
    )
    
    # Record metrics
    metrics_collector.record_request(
        method=request.method,
        endpoint=str(request.url.path),
        status_code=response.status_code,
        duration=duration,
        username=username
    )
    
    return response

@app.post("/auth/login", response_model=TokenResponse)
async def login(login_request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        user = authenticate_user(login_request.username, login_request.password)
        if not user:
            metrics_collector.record_auth_attempt(False)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        access_token = create_access_token(data={"sub": user["username"]})
        metrics_collector.record_auth_attempt(True)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate text using the quantized Llama model"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Sanitize input
        sanitized_prompt = SecurityValidator.sanitize_input(request.prompt)
        validated_params = SecurityValidator.validate_generation_params(request.dict())
        
        # Determine model name
        model_map = {
            "original": "llama3.1-8b-original",
            "8bit": "llama3.1-8b-8bit", 
            "4bit": "llama3.1-8b-4bit"
        }
        
        model_name = model_map.get(request.model_type)
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {request.model_type}"
            )
        
        # Load model
        try:
            model, tokenizer = model_loader.load_model(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        
        # Generate text
        generation_start = time.time()
        
        inputs = tokenizer(
            sanitized_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=validated_params["max_tokens"],
                temperature=validated_params["temperature"],
                top_p=validated_params["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(sanitized_prompt):].strip()
        
        generation_time = time.time() - generation_start
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Log generation
        log_generation(
            request_id=request_id,
            prompt=sanitized_prompt,
            generated_text=generated_text,
            model_type=request.model_type,
            duration=generation_time,
            token_count=tokens_generated,
            user=current_user.get("username")
        )
        
        # Record metrics
        metrics_collector.record_generation(
            model_type=request.model_type,
            duration=generation_time,
            token_count=tokens_generated
        )
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=sanitized_prompt,
            model_type=request.model_type,
            tokens_generated=tokens_generated,
            generation_time=round(generation_time, 3),
            tokens_per_second=round(tokens_per_second, 2),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        metrics_collector.record_error("generation_error", "/generate")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text generation failed"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - (app.state.start_time if hasattr(app.state, 'start_time') else time.time())
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime=round(uptime, 2),
        system_info=get_system_info(),
        available_models=model_loader.list_available_models()
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=get_prometheus_metrics(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get information about the loaded quantized model"""
    return ModelInfoResponse(
        models={
            "loaded": "meta-llama/Meta-Llama-3.1-8B-Instruct (8-bit)",
            "available": model_loader.list_available_models()
        },
        system_info={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "memory_usage": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A",
            "quantization": "8-bit"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Llama 3.1 8B Production API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    # Set start time for uptime calculation
    app.state.start_time = time.time()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
