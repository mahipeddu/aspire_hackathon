#!/usr/bin/env python3
"""
Utility functions and helpers
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """Setup structured logging"""
    log_format = {
        "timestamp": "%(asctime)s",
        "level": "%(levelname)s",
        "module": "%(name)s",
        "message": "%(message)s"
    }
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "module": record.name,
                "message": record.getMessage(),
                "line": record.lineno,
                "function": record.funcName
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'exc_info', 
                              'exc_text', 'stack_info']:
                    log_entry[key] = value
            
            return json.dumps(log_entry)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(JSONFormatter())

def log_request(request_id: str, method: str, path: str, 
                user: Optional[str] = None, **kwargs):
    """Log API request"""
    log_data = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "user": user,
        "type": "request"
    }
    log_data.update(kwargs)
    
    logger.info("API Request", extra=log_data)

def log_response(request_id: str, status_code: int, duration: float, 
                 user: Optional[str] = None, **kwargs):
    """Log API response"""
    log_data = {
        "request_id": request_id,
        "status_code": status_code,
        "duration": duration,
        "user": user,
        "type": "response"
    }
    log_data.update(kwargs)
    
    logger.info("API Response", extra=log_data)

def log_generation(request_id: str, prompt: str, generated_text: str,
                  model_type: str, duration: float, token_count: int,
                  user: Optional[str] = None):
    """Log text generation"""
    log_data = {
        "request_id": request_id,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "generated_text": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
        "model_type": model_type,
        "duration": duration,
        "token_count": token_count,
        "tokens_per_second": token_count / duration if duration > 0 else 0,
        "user": user,
        "type": "generation"
    }
    
    logger.info("Text Generation", extra=log_data)

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": sys.platform
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_memory": {
                f"gpu_{i}": {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i)
                }
                for i in range(torch.cuda.device_count())
            }
        })
    
    return info

def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load model configuration"""
    config_file = Path(model_path) / "config.json"
    
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    
    return {}

def estimate_memory_requirements(model_config: Dict[str, Any], 
                                precision: str = "float16") -> Dict[str, float]:
    """Estimate memory requirements for model"""
    
    # Get model parameters
    vocab_size = model_config.get("vocab_size", 32000)
    hidden_size = model_config.get("hidden_size", 4096)
    num_layers = model_config.get("num_hidden_layers", 32)
    intermediate_size = model_config.get("intermediate_size", 11008)
    
    # Calculate parameter counts (rough estimates)
    embedding_params = vocab_size * hidden_size
    attention_params = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O projections
    mlp_params = num_layers * (2 * hidden_size * intermediate_size)  # Gate, up, down
    layer_norm_params = num_layers * 2 * hidden_size  # Input and post-attention
    
    total_params = embedding_params + attention_params + mlp_params + layer_norm_params
    
    # Bytes per parameter based on precision
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "int8": 1,
        "int4": 0.5
    }.get(precision, 2)
    
    model_memory = total_params * bytes_per_param
    
    # Add overhead for activations (rough estimate)
    activation_memory = model_memory * 0.2  # 20% overhead
    
    return {
        "total_parameters": total_params,
        "model_memory_bytes": model_memory,
        "activation_memory_bytes": activation_memory,
        "total_memory_bytes": model_memory + activation_memory,
        "model_memory_gb": model_memory / (1024**3),
        "total_memory_gb": (model_memory + activation_memory) / (1024**3)
    }

class ModelLoader:
    """Utility class for loading models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
    
    def list_available_models(self) -> List[str]:
        """List available model directories"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item.name)
        
        return models
    
    def load_model(self, model_name: str, device: Optional[str] = None, unload_previous: bool = True) -> tuple:
        """Load model and tokenizer"""
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        # Unload all previous models to free memory if requested
        if unload_previous:
            self.unload_all_models()
        
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate configuration for quantized models
            if "8bit" in model_name.lower():
                # 8-bit quantized model
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_memory={0: "14GB", "cpu": "30GB"},
                    offload_folder="./offload"
                )
            elif "4bit" in model_name.lower():
                # 4-bit quantized model
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_memory={0: "14GB", "cpu": "30GB"},
                    offload_folder="./offload"
                )
            else:
                # Original model
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else device
                )
            
            model.eval()
            
            # Cache the loaded model
            self.loaded_models[model_name] = (model, tokenizer)
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self.loaded_models:
            model, tokenizer = self.loaded_models[model_name]
            
            # For quantized models with device_map="auto", we need special handling
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Model {model_name} uses device_map, clearing device mapping...")
                try:
                    # Clear the device map
                    if hasattr(model, 'hf_device_map'):
                        del model.hf_device_map
                except Exception as e:
                    logger.warning(f"Could not clear device map: {e}")
            
            # Move model components to CPU before deletion to free GPU memory
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except Exception as e:
                    logger.warning(f"Could not move model to CPU: {e}")
            
            # For models with tied weights, clear them
            if hasattr(model, '_tied_weights_keys'):
                try:
                    model._tied_weights_keys = []
                except Exception as e:
                    logger.warning(f"Could not clear tied weights: {e}")
            
            # Delete the model and tokenizer references
            del model
            del tokenizer
            del self.loaded_models[model_name]
            
            # Force multiple rounds of garbage collection
            import gc
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                # Clear all GPU cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                torch.cuda.empty_cache()  # Second clear after sync
                
                # Reset memory stats and clear reserved memory
                try:
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_max_memory_cached()
                    # Force release of cached memory
                    torch.cuda.set_per_process_memory_fraction(1.0)
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Could not reset CUDA memory stats: {e}")
            
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False
    
    def unload_all_models(self):
        """Unload all models from memory"""
        if self.loaded_models:
            logger.info(f"Unloading {len(self.loaded_models)} models from memory...")
            model_names = list(self.loaded_models.keys())
            successful_unloads = 0
            
            for model_name in model_names:
                if self.unload_model(model_name):
                    successful_unloads += 1
            
            # Final aggressive cleanup
            import gc
            for _ in range(5):
                gc.collect()
                
            if torch.cuda.is_available():
                # Multiple rounds of cache clearing
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                try:
                    # Reset memory statistics
                    torch.cuda.reset_max_memory_allocated()
                    torch.cuda.reset_max_memory_cached()
                    
                    # Try to force release of all cached memory
                    torch.cuda.set_per_process_memory_fraction(0.95)
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(1.0)
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Could not reset CUDA memory stats: {e}")
            
            logger.info(f"Successfully unloaded {successful_unloads}/{len(model_names)} models from memory")
            return successful_unloads
        return 0
    
    def get_current_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        memory_info = {
            'loaded_models': list(self.loaded_models.keys()),
            'loaded_models_count': len(self.loaded_models)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated_bytes = torch.cuda.memory_allocated(i)
                reserved_bytes = torch.cuda.memory_reserved(i)
                total_bytes = torch.cuda.get_device_properties(i).total_memory
                
                memory_info[f'gpu_{i}'] = {
                    'allocated_gb': allocated_bytes / (1024**3),
                    'reserved_gb': reserved_bytes / (1024**3),
                    'total_gb': total_bytes / (1024**3),
                    'available_gb': (total_bytes - reserved_bytes) / (1024**3),
                    'utilization_percent': (reserved_bytes / total_bytes) * 100
                }
        else:
            memory_info['cuda_available'] = False
            
        return memory_info
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        config = load_model_config(str(model_path))
        memory_reqs = estimate_memory_requirements(config)
        
        info = {
            "name": model_name,
            "path": str(model_path),
            "config": config,
            "memory_requirements": memory_reqs,
            "loaded": model_name in self.loaded_models
        }
        
        return info

def create_error_response(error_type: str, message: str, 
                         status_code: int = 500) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": {
            "type": error_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code
        }
    }

def validate_environment():
    """Validate environment and dependencies"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check PyTorch installation
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - will use CPU (slower)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check transformers
    try:
        import transformers
    except ImportError:
        issues.append("transformers not installed")
    
    # Check models directory
    models_dir = Path("models")
    if not models_dir.exists():
        issues.append("Models directory not found")
    elif not any(models_dir.iterdir()):
        issues.append("No models found in models directory")
    
    return issues
