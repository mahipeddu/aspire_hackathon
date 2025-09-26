#!/usr/bin/env python3
"""
Test the Llama API endpoints
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"

def test_authentication():
    """Test user authentication"""
    print("Testing authentication...")
    
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    
    if response.status_code == 200:
        token_data = response.json()
        print(f"✅ Authentication successful: {token_data['token_type']}")
        return token_data['access_token']
    else:
        print(f"❌ Authentication failed: {response.status_code} - {response.text}")
        return None

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        health_data = response.json()
        print(f"✅ Health check passed - Status: {health_data['status']}")
        print(f"   Available models: {health_data['available_models']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("Testing metrics endpoint...")
    
    response = requests.get(f"{API_BASE_URL}/metrics")
    
    if response.status_code == 200:
        print("✅ Metrics endpoint accessible")
        print(f"   Response length: {len(response.text)} characters")
        return True
    else:
        print(f"❌ Metrics endpoint failed: {response.status_code}")
        return False

def test_text_generation(token):
    """Test text generation with different models"""
    if not token:
        print("❌ Skipping text generation - no auth token")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    test_cases = [
        {
            "model_type": "8bit",
            "prompt": "The future of artificial intelligence is",
            "max_tokens": 50
        },
        {
            "model_type": "4bit", 
            "prompt": "Explain quantum computing briefly:",
            "max_tokens": 75
        },
        {
            "model_type": "original",
            "prompt": "What are the benefits of renewable energy?",
            "max_tokens": 100
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Testing generation {i}/3 - {test_case['model_type']} model...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/generate",
            headers=headers,
            json=test_case
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful:")
            print(f"   Model: {result['model_type']}")
            print(f"   Tokens: {result['tokens_generated']}")
            print(f"   Speed: {result['tokens_per_second']:.1f} tok/s")
            print(f"   API Latency: {end_time - start_time:.2f}s")
            print(f"   Generated: {result['generated_text'][:100]}...")
        else:
            print(f"❌ Generation failed: {response.status_code} - {response.text}")
        
        print()

def test_model_info(token):
    """Test model info endpoint"""
    if not token:
        print("❌ Skipping model info - no auth token")
        return False
        
    print("Testing model info endpoint...")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE_URL}/model/info", headers=headers)
    
    if response.status_code == 200:
        model_info = response.json()
        print("✅ Model info retrieved successfully")
        print(f"   Available models: {list(model_info['models'].keys())}")
        return True
    else:
        print(f"❌ Model info failed: {response.status_code}")
        return False

def test_rate_limiting():
    """Test rate limiting"""
    print("Testing rate limiting (making rapid requests)...")
    
    # Make many requests quickly to test rate limiting
    success_count = 0
    rate_limited_count = 0
    
    for i in range(10):
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            success_count += 1
        elif response.status_code == 429:
            rate_limited_count += 1
    
    print(f"   Successful requests: {success_count}")
    print(f"   Rate limited requests: {rate_limited_count}")
    
    if rate_limited_count > 0:
        print("✅ Rate limiting is working")
    else:
        print("ℹ️  Rate limiting not triggered (may need more requests)")

def main():
    """Run all tests"""
    print("🚀 Starting Llama API Tests")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ API is not healthy, stopping tests")
        return
    
    print()
    
    # Test authentication
    token = test_authentication()
    print()
    
    # Test metrics
    test_metrics()
    print()
    
    # Test model info
    test_model_info(token)
    print()
    
    # Test text generation
    test_text_generation(token)
    
    # Test rate limiting
    test_rate_limiting()
    print()
    
    print("🎉 API testing completed!")

if __name__ == "__main__":
    main()
