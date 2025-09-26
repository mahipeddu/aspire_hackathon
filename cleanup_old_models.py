#!/usr/bin/env python3
"""
Script to clean up old Llama 3.1 model files and prepare for Llama 2
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_models():
    """Remove old Llama 3.1 model directories"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        logger.info("No models directory found, nothing to clean")
        return
    
    # Old model directories to remove
    old_models = [
        "llama3.1-8b-original",
        "llama3.1-8b-8bit", 
        "llama3.1-8b-4bit"
    ]
    
    logger.info("Starting cleanup of old Llama 3.1 models...")
    
    for model_name in old_models:
        model_path = models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Removing {model_path}...")
            try:
                shutil.rmtree(model_path)
                logger.info(f"✅ Successfully removed {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {model_path}: {e}")
        else:
            logger.info(f"Directory {model_path} does not exist, skipping")
    
    # Also remove old benchmark results if they exist
    benchmark_file = models_dir / "benchmark_results.json"
    if benchmark_file.exists():
        logger.info("Removing old benchmark results...")
        try:
            benchmark_file.unlink()
            logger.info("✅ Successfully removed old benchmark results")
        except Exception as e:
            logger.error(f"❌ Failed to remove benchmark results: {e}")
    
    logger.info("Cleanup completed!")
    logger.info("Run 'python model_quantization.py' to download and quantize Llama 2 8B models")

if __name__ == "__main__":
    cleanup_old_models()
