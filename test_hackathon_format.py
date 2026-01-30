"""
Hackathon Endpoint Tester - Exact Format Match
Tests your API with the EXACT format the hackathon evaluation system uses
"""

import requests
import base64
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

# Your deployed API endpoint
API_ENDPOINT = "https://your-app-name.onrender.com/api/detect"

# Your API key
API_KEY = "hackathon_2026_voice_detection_secure_key"

# Test audio file path (optional)
TEST_AUDIO_FILE = "sample.mp3"  # Change this to your audio file

# ============================================================================


def test_with_hackathon_format(
    endpoint_url: str,
    api_key: str,
    language: str = "English",
    audio_format: str = "mp3",
    audio_base64: str = None,
    audio_file_path: str = None
):
    """
    Test API using EXACT hackathon format
    
    This mimics what the hackathon evaluation system will send
    """
    
    print("\n" + "="*70)
    print("🎯 HACKATHON ENDPOINT TESTER")
    print("="*70)
    
    # === Prepare Audio ===
    if audio_base64 is None:
        if audio_file_path and Path(audio_file_path).exists():
            print(f"📁 Loading audio file: {audio_file_path}")
            with open(audio_file_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode()
            print(f"✅ Audio loaded: {len(audio_base64)} characters")
        else:
            print("⚠️  No audio file provided, using dummy data")
            audio_base64 = base64.b64encode(b"dummy audio" * 100).decode()
    
    # === Prepare Headers (EXACT format) ===
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # === Prepare Request Body (EXACT format) ===
    request_body = {
        "language": language,
        "audio_format": audio_format,
        "audio_base64_format": audio_base64
    }
    
    print(f"\n📤 Sending Request to: {endpoint_url}")
    print(f"🔑 Using API Key: {api_key[:20]}...")
    print(f"🌍 Language: {language}")
    print(f"🎵 Audio Format: {audio_format}")
    print(f"📊 Audio Data Size: {len(audio_base64)} characters")
    
    # === Send Request ===
    try:
        print("\n⏳ Sending request...")
        import time
        start_time = time.time()
        
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'─'*70}")
        print(f"📬 RESPONSE RECEIVED")
        print(f"{'─'*70}")
        print(f"⏱️  Response Time: {elapsed_time:.2f}s")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS!")
            result = response.json()
            
            print(f"\n{'─'*70}")
            print(f"📋 DETECTION RESULTS")
            print(f"{'─'*70}")
            print(f"🤖 Classification: {result.get('classification', 'N/A')}")
            print(f"📈 Confidence: {result.get('confidence', 0):.2%}")
            print(f"🌍 Language: {result.get('language_detected', 'N/A')}")
            print(f"⏱️  Processing Time: {result.get('processing_time_ms', 0)}ms")
            
            print(f"\n💬 Explanation:")
            print(f"   {result.get('explanation', 'N/A')}")
            
            if 'features_analyzed' in result:
                print(f"\n🔍 Features Analyzed:")
                for key, value in result['features_analyzed'].items():
                    print(f"   • {key}: {value}")
            
            print(f"\n{'─'*70}")
            print(f"✅ API TEST PASSED!")
            print(f"{'─'*70}")
            
            # Validate response format
            validate_response(result)
            
            return True, result
            
        elif response.status_code == 401:
            print(f"❌ AUTHENTICATION FAILED")
            print(f"   Missing or invalid x-api-key header")
            print(f"   Response: {response.text}")
            return False, None
            
        elif response.status_code == 403:
            print(f"❌ AUTHORIZATION FAILED")
            print(f"   Invalid API key")
            print(f"   Response: {response.text}")
            return False, None
            
        elif response.status_code == 400:
            print(f"❌ BAD REQUEST")
            print(f"   Invalid request format or data")
            print(f"   Response: {response.text}")
            return False, None
            
        else:
            print(f"❌ UNEXPECTED STATUS CODE: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print(f"❌ REQUEST TIMEOUT")
        print(f"   API did not respond within 30 seconds")
        return False, None
        
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR")
        print(f"   Cannot reach the API endpoint")
        print(f"   Check if URL is correct and API is running")
        return False, None
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, None


def validate_response(result: dict):
    """Validate that response matches required format"""
    print(f"\n{'─'*70}")
    print(f"🔍 RESPONSE VALIDATION")
    print(f"{'─'*70}")
    
    required_fields = [
        "classification",
        "confidence",
        "explanation",
        "language_detected",
        "processing_time_ms"
    ]
    
    all_valid = True
    
    for field in required_fields:
        if field in result:
            print(f"✅ {field}: Present")
        else:
            print(f"❌ {field}: MISSING!")
            all_valid = False
    
    # Validate classification value
    if result.get('classification') in ['AI_GENERATED', 'HUMAN']:
        print(f"✅ classification value: Valid")
    else:
        print(f"❌ classification value: Must be 'AI_GENERATED' or 'HUMAN'")
        all_valid = False
    
    # Validate confidence range
    confidence = result.get('confidence', -1)
    if 0.0 <= confidence <= 1.0:
        print(f"✅ confidence range: Valid (0.0 to 1.0)")
    else:
        print(f"❌ confidence range: Must be between 0.0 and 1.0")
        all_valid = False
    
    if all_valid:
        print(f"\n✅ ALL VALIDATIONS PASSED!")
    else:
        print(f"\n⚠️  SOME VALIDATIONS FAILED!")
    
    return all_valid


def test_all_languages(endpoint_url: str, api_key: str, audio_file_path: str = None):
    """Test API with all 5 supported languages"""
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    print("\n" + "="*70)
    print("🌍 TESTING ALL 5 LANGUAGES")
    print("="*70)
    
    results = {}
    
    for lang in languages:
        print(f"\n\n{'#'*70}")
        print(f"# Testing Language: {lang}")
        print(f"{'#'*70}")
        
        success, result = test_with_hackathon_format(
            endpoint_url=endpoint_url,
            api_key=api_key,
            language=lang,
            audio_file_path=audio_file_path
        )
        
        results[lang] = {
            "success": success,
            "result": result
        }
        
        import time
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n\n" + "="*70)
    print("📊 SUMMARY - ALL LANGUAGES")
    print("="*70)
    
    for lang, data in results.items():
        status = "✅ PASS" if data["success"] else "❌ FAIL"
        print(f"{status} - {lang}")
    
    passed = sum(1 for d in results.values() if d["success"])
    print(f"\nTotal: {passed}/{len(languages)} languages passed")


def main():
    """Main test runner"""
    
    print("\n" + "="*70)
    print("🎯 AI VOICE DETECTION - HACKATHON ENDPOINT TESTER")
    print("="*70)
    
    print("\nThis tester uses the EXACT format the hackathon evaluation system uses.")
    print("Make sure your API is deployed and running!")
    
    # Get configuration
    print(f"\n📋 Current Configuration:")
    print(f"   Endpoint: {API_ENDPOINT}")
    print(f"   API Key: {API_KEY[:20]}...")
    
    change = input("\nDo you want to change these settings? (y/n): ").strip().lower()
    
    if change == 'y':
        endpoint = input("Enter API Endpoint URL: ").strip()
        if endpoint:
            global API_ENDPOINT
            API_ENDPOINT = endpoint
        
        api_key = input("Enter API Key: ").strip()
        if api_key:
            global API_KEY
            API_KEY = api_key
    
    # Get audio file
    audio_file = input(f"\nPath to audio file (press Enter to use dummy data): ").strip()
    if not audio_file:
        audio_file = None
    
    # Test options
    print("\n" + "="*70)
    print("TEST OPTIONS")
    print("="*70)
    print("1. Test with English (quick test)")
    print("2. Test all 5 languages (comprehensive)")
    print("3. Custom language test")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_with_hackathon_format(
            endpoint_url=API_ENDPOINT,
            api_key=API_KEY,
            language="English",
            audio_file_path=audio_file
        )
    
    elif choice == "2":
        test_all_languages(
            endpoint_url=API_ENDPOINT,
            api_key=API_KEY,
            audio_file_path=audio_file
        )
    
    elif choice == "3":
        lang = input("Enter language (Tamil/English/Hindi/Malayalam/Telugu): ").strip()
        test_with_hackathon_format(
            endpoint_url=API_ENDPOINT,
            api_key=API_KEY,
            language=lang,
            audio_file_path=audio_file
        )
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("1. If all tests pass, your API is ready!")
    print("2. Submit your endpoint URL and API key to the hackathon")
    print("3. Use their official endpoint tester to verify")
    print("4. Wait for evaluation results")
    print("\n🏆 Good luck getting shortlisted for Delhi! 🚀")


if __name__ == "__main__":
    main()
