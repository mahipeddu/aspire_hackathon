#!/usr/bin/env python3
"""
Test script to verify text generation length and quality
"""

from utils import ModelLoader
import torch

def test_text_generation():
    """Test text generation with different models"""
    
    print("=== Text Generation Length Test ===")
    
    # Create model loader
    model_loader = ModelLoader()
    
    # Test prompt that should generate a longer response
    test_prompt = "Apple or Samsung which is better"
    
    models_to_test = ["llama2-7b-4bit", "llama2-7b-8bit"]
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            # Load model
            model, tokenizer = model_loader.load_model(model_name, unload_previous=True)
            
            # Prepare input
            inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            print(f"Input prompt: '{test_prompt}'")
            print(f"Input tokens: {len(inputs['input_ids'][0])}")
            
            # Generate with the same settings as web app
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                    num_beams=1,
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_response = generated_text.replace(test_prompt, "").strip()
            
            print(f"Generated tokens: {len(outputs[0]) - len(inputs['input_ids'][0])}")
            print(f"Response length: {len(ai_response)} characters")
            print(f"Response: {ai_response}")
            print(f"Ends with complete sentence: {ai_response.endswith(('.', '!', '?'))}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Cleanup
    model_loader.unload_all_models()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_text_generation()
