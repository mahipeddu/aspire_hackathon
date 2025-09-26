#!/usr/bin/env python3
"""
Simple test script for the web chat interface
"""

import requests
import time

def test_web_app():
    """Test the web app endpoints"""
    base_url = "http://localhost:8080"
    
    print("🧪 Testing Web Chat Interface...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Available models: {data.get('available_models', [])}")
            print(f"   Current model: {data.get('current_model', 'None')}")
            print(f"   Active connections: {data.get('active_connections', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test main page
        print("\n2. Testing main page...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Main page accessible")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
        
        # Test API models endpoint
        print("\n3. Testing models API...")
        response = requests.get(f"{base_url}/api/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models API working")
            print(f"   Available: {data.get('models', [])}")
            print(f"   Default: {data.get('default', 'None')}")
        else:
            print(f"❌ Models API failed: {response.status_code}")
        
        print("\n🎉 Web app is running successfully!")
        print("📱 Open your browser to: http://localhost:8080")
        print("💬 Try chatting with the AI models!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web app. Is it running on localhost:8080?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Web app may be slow to respond.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_web_app()
