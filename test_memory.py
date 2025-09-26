#!/usr/bin/env python3
"""
Test script to verify GPU memory management when switching models
"""

import torch
from utils import ModelLoader
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def test_memory_management():
    """Test model loading and unloading"""
    
    print("=== GPU Memory Management Test ===")
    
    # Initial memory
    allocated, reserved = get_gpu_memory()
    print(f"Initial memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # Create model loader
    model_loader = ModelLoader()
    
    # Test sequence: load models one by one
    models_to_test = ["llama2-7b-8bit", "llama2-7b-4bit", "llama2-7b-original"]
    
    for i, model_name in enumerate(models_to_test):
        print(f"\n--- Test {i+1}: Loading {model_name} ---")
        
        try:
            # Load model
            print(f"Loading {model_name}...")
            model, tokenizer = model_loader.load_model(model_name, unload_previous=True)
            
            # Check memory after loading
            allocated, reserved = get_gpu_memory()
            print(f"After loading {model_name} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # Test a simple generation to make sure model works
            if model and tokenizer:
                print("Testing model generation...")
                inputs = tokenizer("Hello, how are you?", return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'], 
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Model response: {response}")
            
            print(f"Successfully tested {model_name}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    print(f"\n--- Final cleanup test ---")
    
    # Clear all models
    unloaded_count = model_loader.unload_all_models()
    print(f"Unloaded {unloaded_count} models")
    
    # Force cleanup
    import gc
    for _ in range(5):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Final memory check
    allocated, reserved = get_gpu_memory()
    print(f"After cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print("\n=== Memory Management Test Complete ===")

if __name__ == "__main__":
    test_memory_management()
