#!/usr/bin/env python3
"""
Model Quantization Script for Llama 2 7B
Implements multiple quantization techniques and benchmarking
"""

import os
import time
import json
import torch
import psutil
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from accelerate import Accelerator
from datasets import load_dataset
import evaluate
import gc
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Handles model downloading, quantization, and benchmarking"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.results = {}
        
        # Create accelerator for distributed training/inference
        self.accelerator = Accelerator()
        
        # Note: Llama 2 requires HuggingFace authentication
        # Make sure you're logged in: huggingface-cli login
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current RAM and VRAM usage"""
        # RAM usage
        ram_usage = psutil.virtual_memory().used / (1024**3)  # GB
        
        # VRAM usage
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            
        return ram_usage, vram_usage
    
    def get_model_size(self, model_path: str) -> float:
        """Calculate model size on disk in GB"""
        if os.path.isdir(model_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**3)  # Convert to GB
        return 0
    
    def benchmark_inference_speed(self, model, tokenizer, num_samples: int = 10) -> float:
        """Benchmark inference speed in tokens per second"""
        test_prompts = [
            "The future of artificial intelligence is",
            "Climate change affects",
            "Technology has transformed",
            "Space exploration reveals",
            "Medical research shows",
        ] * (num_samples // 5 + 1)
        
        total_tokens = 0
        start_time = time.time()
        
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts[:num_samples]):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                total_tokens += outputs.shape[1] - inputs["input_ids"].shape[1]
                
                if i % 5 == 0:
                    logger.info(f"Processed {i+1}/{num_samples} samples")
        
        end_time = time.time()
        tokens_per_second = total_tokens / (end_time - start_time)
        
        return tokens_per_second
    
    def evaluate_perplexity(self, model, tokenizer, num_samples: int = 100) -> float:
        """Evaluate model perplexity on a standard dataset"""
        try:
            # Use a small subset of WikiText for evaluation
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            
            # Take only a subset for faster evaluation
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50][:num_samples]
            
            total_loss = 0
            total_tokens = 0
            
            model.eval()
            with torch.no_grad():
                for text in texts:
                    if not text.strip():
                        continue
                        
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                    total_tokens += inputs["input_ids"].shape[1]
                    
            average_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(average_loss)).item()
            
            return perplexity
        except Exception as e:
            logger.warning(f"Could not evaluate perplexity: {e}")
            return -1
    
    def qualitative_evaluation(self, model, tokenizer) -> float:
        """Simple qualitative evaluation with test prompts"""
        test_cases = [
            {
                "prompt": "Explain quantum computing in simple terms:",
                "expected_keywords": ["quantum", "computing", "bits", "information"]
            },
            {
                "prompt": "What are the benefits of renewable energy?",
                "expected_keywords": ["renewable", "energy", "environment", "sustainable"]
            },
            {
                "prompt": "Describe the process of photosynthesis:",
                "expected_keywords": ["photosynthesis", "plants", "sunlight", "oxygen"]
            }
        ]
        
        total_score = 0
        
        model.eval()
        with torch.no_grad():
            for case in test_cases:
                inputs = tokenizer(case["prompt"], return_tensors="pt", truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(case["prompt"]):].lower()
                
                # Simple keyword matching score
                keyword_matches = sum(1 for keyword in case["expected_keywords"] 
                                    if keyword.lower() in generated_text)
                case_score = keyword_matches / len(case["expected_keywords"])
                total_score += case_score
                
        average_score = (total_score / len(test_cases)) * 10  # Scale to 0-10
        return average_score
    
    def download_and_save_original_model(self):
        """Download and save the original FP16 model"""
        logger.info("Downloading original Llama 2 7B model...")
        
        model_path = self.models_dir / "llama2-7b-original"
        
        if model_path.exists():
            logger.info("Original model already exists, loading from disk...")
        else:
            logger.info("Downloading model from HuggingFace...")
            
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(model_path),
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(model_path),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # Save the model and tokenizer
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        logger.info(f"Original model saved to {model_path}")
        return model, tokenizer, str(model_path)
    
    def create_8bit_quantized_model(self):
        """Create 8-bit quantized model using BitsAndBytesConfig"""
        logger.info("Creating 8-bit quantized model...")
        
        model_path = self.models_dir / "llama2-7b-8bit"
        
        # Check if already exists
        if model_path.exists() and (model_path / "config.json").exists():
            logger.info("8-bit model already exists, loading from disk...")
            
            # Configure 8-bit quantization for loading saved model
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )
            
            # Set up device map with proper memory allocation
            max_memory = {0: "14GB", "cpu": "30GB"}  # Allocate 14GB to GPU, rest to CPU
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
            return model, tokenizer, str(model_path)
        
        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
        )
        
        # Set up device map with proper memory allocation
        max_memory = {0: "14GB", "cpu": "30GB"}  # Allocate 14GB to GPU, rest to CPU
        
        # Load from original model path to avoid re-downloading
        original_model_path = self.models_dir / "llama2-7b-original"
        if original_model_path.exists():
            logger.info("Loading from local original model for 8-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(str(original_model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(original_model_path),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
        else:
            logger.info("Loading from HuggingFace for 8-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
        
        # Save the quantized model
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        logger.info(f"8-bit model saved to {model_path}")
        return model, tokenizer, str(model_path)
    
    def create_4bit_quantized_model(self):
        """Create 4-bit quantized model using BitsAndBytesConfig"""
        logger.info("Creating 4-bit quantized model...")
        
        model_path = self.models_dir / "llama2-7b-4bit"
        
        # Check if already exists
        if model_path.exists() and (model_path / "config.json").exists():
            logger.info("4-bit model already exists, loading from disk...")
            
            # Configure 4-bit quantization for loading saved model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )
            
            # Set up device map with proper memory allocation
            max_memory = {0: "14GB", "cpu": "30GB"}  # Allocate 14GB to GPU, rest to CPU
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
            return model, tokenizer, str(model_path)
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
            llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
        )
        
        # Set up device map with proper memory allocation
        max_memory = {0: "14GB", "cpu": "30GB"}  # Allocate 14GB to GPU, rest to CPU
        
        # Load from original model path to avoid re-downloading
        original_model_path = self.models_dir / "llama2-7b-original"
        if original_model_path.exists():
            logger.info("Loading from local original model for 4-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(str(original_model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(original_model_path),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
        else:
            logger.info("Loading from HuggingFace for 4-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="./offload"
            )
        
        # Save the quantized model
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        logger.info(f"4-bit model saved to {model_path}")
        return model, tokenizer, str(model_path)
    
    def benchmark_model(self, model, tokenizer, model_path: str, model_name: str):
        """Comprehensive benchmarking of a model"""
        logger.info(f"Benchmarking {model_name} model...")
        
        # Clear GPU cache before benchmarking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Get baseline memory usage (before model operations)
        ram_baseline = psutil.virtual_memory().used / (1024**3)
        vram_baseline = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Model size on disk
        model_size = self.get_model_size(model_path)
        
        # Perform a warmup inference to ensure model is fully loaded
        logger.info("Warming up model...")
        try:
            warmup_input = tokenizer("Hello world", return_tensors="pt", truncation=True)
            if torch.cuda.is_available():
                warmup_input = {k: v.cuda() for k, v in warmup_input.items()}
            
            with torch.no_grad():
                _ = model.generate(**warmup_input, max_new_tokens=5, do_sample=False)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Get memory usage after model is fully loaded and warmed up
        ram_after_warmup = psutil.virtual_memory().used / (1024**3)
        vram_after_warmup = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Inference speed
        logger.info("Measuring inference speed...")
        inference_speed = self.benchmark_inference_speed(model, tokenizer, num_samples=5)
        
        # Memory usage during/after inference
        ram_after_inference = psutil.virtual_memory().used / (1024**3)
        vram_after_inference = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Calculate actual memory usage (take the maximum observed)
        ram_usage = max(ram_after_warmup - ram_baseline, ram_after_inference - ram_baseline, 0)
        vram_usage = max(vram_after_warmup - vram_baseline, vram_after_inference - vram_baseline, 0)
        
        # If still getting low values, use peak memory
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            if peak_vram > vram_usage:
                vram_usage = peak_vram
        
        # Perplexity evaluation
        logger.info("Evaluating perplexity...")
        perplexity = self.evaluate_perplexity(model, tokenizer, num_samples=20)
        
        # Qualitative evaluation
        logger.info("Performing qualitative evaluation...")
        quality_score = self.qualitative_evaluation(model, tokenizer)
        
        results = {
            "model_name": model_name,
            "model_size_gb": round(model_size, 2),
            "ram_usage_gb": round(ram_usage, 2),
            "vram_usage_gb": round(vram_usage, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0,
            "inference_speed_tokens_per_sec": round(inference_speed, 1),
            "perplexity": round(perplexity, 2) if perplexity > 0 else "N/A",
            "quality_score": round(quality_score, 1)
        }
        
        self.results[model_name] = results
        logger.info(f"Benchmarking complete for {model_name}")
        logger.info(f"Results: {results}")
        
        return results
    
    def run_complete_quantization_pipeline(self):
        """Run the complete quantization and benchmarking pipeline"""
        logger.info("Starting complete quantization pipeline...")
        
        # Get initial system memory baseline
        initial_ram = psutil.virtual_memory().used / (1024**3)
        initial_vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        logger.info(f"Initial system state - RAM: {initial_ram:.2f}GB, VRAM: {initial_vram:.2f}GB")
        
        try:
            # 1. Download and benchmark original model
            logger.info("=" * 50)
            logger.info("STEP 1: Original Model")
            logger.info("=" * 50)
            
            model_orig, tokenizer_orig, path_orig = self.download_and_save_original_model()
            self.benchmark_model(model_orig, tokenizer_orig, path_orig, "Original FP16")
            
            # Clear memory
            del model_orig, tokenizer_orig
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            
            # 2. Create and benchmark 8-bit model
            logger.info("=" * 50)
            logger.info("STEP 2: 8-bit Quantized Model")
            logger.info("=" * 50)
            
            model_8bit, tokenizer_8bit, path_8bit = self.create_8bit_quantized_model()
            self.benchmark_model(model_8bit, tokenizer_8bit, path_8bit, "8-bit INT8")
            
            # Clear memory
            del model_8bit, tokenizer_8bit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            
            # 3. Create and benchmark 4-bit model
            logger.info("=" * 50)
            logger.info("STEP 3: 4-bit Quantized Model")
            logger.info("=" * 50)
            
            model_4bit, tokenizer_4bit, path_4bit = self.create_4bit_quantized_model()
            self.benchmark_model(model_4bit, tokenizer_4bit, path_4bit, "4-bit INT4")
            
            # Clear memory
            del model_4bit, tokenizer_4bit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 4. Save results
            self.save_benchmark_results()
            self.print_benchmark_comparison()
            
        except Exception as e:
            logger.error(f"Error in quantization pipeline: {e}")
            raise
    
    def save_benchmark_results(self):
        """Save benchmark results to JSON file"""
        results_file = self.models_dir / "benchmark_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def print_benchmark_comparison(self):
        """Print a formatted comparison table"""
        logger.info("=" * 80)
        logger.info("BENCHMARK RESULTS COMPARISON")
        logger.info("=" * 80)
        
        # Print header
        print(f"{'Model':<15} {'Size (GB)':<10} {'VRAM (GB)':<10} {'Peak VRAM':<10} {'Speed (tok/s)':<12} {'Perplexity':<12} {'Quality':<10}")
        print("-" * 95)
        
        # Print results for each model
        for model_name, results in self.results.items():
            peak_vram = results.get('peak_vram_gb', results.get('vram_usage_gb', 0))
            print(f"{model_name:<15} "
                  f"{results['model_size_gb']:<10} "
                  f"{results['vram_usage_gb']:<10} "
                  f"{peak_vram:<10} "
                  f"{results['inference_speed_tokens_per_sec']:<12} "
                  f"{results['perplexity']:<12} "
                  f"{results['quality_score']}/10")
        
        print("-" * 95)
        
        # Print recommendations
        print("\nRECOMMENDATIONS:")
        print("- For best quality: Original FP16")
        print("- For balanced performance: 8-bit INT8") 
        print("- For maximum efficiency: 4-bit INT4")

def main():
    """Main function to run the quantization pipeline"""
    logger.info("Starting Llama 2 7B Quantization Pipeline")
    
    # Initialize quantizer
    quantizer = ModelQuantizer()
    
    # Run the complete pipeline
    quantizer.run_complete_quantization_pipeline()
    
    logger.info("Quantization pipeline completed successfully!")

if __name__ == "__main__":
    main()
