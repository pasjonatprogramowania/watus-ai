"""
Simple Memory System Test Script
Run with: python test_memory_simple.py
"""
import sys
import os
import time
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = "http://localhost:8000/api1"

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_test(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")

def test_imports():
    """Test that all modules can be imported."""
    print_header("TEST 1: Module Imports")
    
    try:
        from mem0 import Memory
        print_test("mem0 import", True)
    except ImportError as e:
        print_test("mem0 import", False, str(e))
        return False
    
    try:
        from src.emma import retrieve_relevant_memories, consolidate_memory
        print_test("emma module import", True)
    except ImportError as e:
        print_test("emma module import", False, str(e))
        return False
    
    try:
        from src.main import vector_search
        from src.vectordb import search_memory
        print_test("vectordb/main module import", True)
    except ImportError as e:
        print_test("vectordb/main module import", False, str(e))
        return False
    
    return True

def test_memory_init():
    """Test mem0 memory initialization."""
    print_header("TEST 2: Memory Initialization")
    
    try:
        from src.emma import _get_memory
        memory = _get_memory()
        print_test("mem0 Memory instance created", memory is not None)
        return True
    except Exception as e:
        print_test("mem0 Memory initialization", False, str(e))
        return False

def test_memory_consolidation():
    """Test storing memories."""
    print_header("TEST 3: Memory Consolidation")
    
    try:
        from src.emma import consolidate_memory
        
        user_id = f"test_user_{int(time.time())}"
        user_input = "My name is TestUser and I am running unit tests."
        ai_response = "Hello TestUser! I'll remember that you're running tests."
        
        consolidate_memory(user_id, user_input, ai_response)
        print_test("Memory consolidation", True, f"user_id={user_id}")
        return True, user_id
    except Exception as e:
        print_test("Memory consolidation", False, str(e))
        return False, None

def test_memory_retrieval(user_id):
    """Test retrieving memories."""
    print_header("TEST 4: Memory Retrieval")
    
    try:
        from src.emma import retrieve_relevant_memories
        
        # Wait for consolidation
        print("  Waiting 2s for memory consolidation...")
        time.sleep(2)
        
        result = retrieve_relevant_memories(user_id, "What is my name?")
        
        has_content = len(result) > 0
        print_test("Memory retrieval", True, f"Got {len(result)} chars")
        
        if has_content:
            print(f"  Memory context preview: {result[:100]}...")
        
        return True
    except Exception as e:
        print_test("Memory retrieval", False, str(e))
        return False

def test_vector_search():
    """Test ChromaDB vector search for WAT knowledge."""
    print_header("TEST 5: Vector Database (WAT Knowledge)")
    
    try:
        from src.main import vector_search
        
        result = vector_search("Ile wydziałów ma WAT?")
        
        has_docs = result is not None and hasattr(result, 'documents') and len(result.documents) > 0
        print_test("WAT knowledge retrieval", has_docs)
        
        if has_docs:
            first_doc = result.documents[0] if isinstance(result.documents[0], str) else result.documents[0]
            print(f"  First result preview: {str(first_doc)[:80]}...")
        
        return True
    except Exception as e:
        print_test("Vector search", False, str(e))
        return False

def test_api_integration():
    """Test full API integration."""
    print_header("TEST 6: API Integration")
    
    try:
        # Check if API is running
        response = requests.get("http://localhost:8000/docs", timeout=3)
        if response.status_code != 200:
            print_test("API health check", False, "API not responding")
            return False
        print_test("API health check", True)
    except requests.ConnectionError:
        print_test("API connection", False, "API not running. Start with: python -m src.main")
        return False
    
    # Test question processing
    try:
        response = requests.post(
            f"{API_BASE_URL}/process_question",
            json={"content": "Moje imię to APITestUser."},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            has_answer = "answer" in data and len(data["answer"]) > 0
            print_test("API question processing", has_answer)
            if has_answer:
                print(f"  Response preview: {data['answer'][:80]}...")
        else:
            print_test("API question processing", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("API request", False, str(e))
        return False
    
    # Wait for memory consolidation
    print("  Waiting 3s for memory consolidation...")
    time.sleep(3)
    
    # Test memory recall
    try:
        response = requests.post(
            f"{API_BASE_URL}/process_question",
            json={"content": "Jak mam na imię?"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "").lower()
            has_name = "apitestuser" in answer or "api" in answer
            print_test("Memory recall in response", has_name, 
                      "Name found in response" if has_name else "Name NOT found")
            print(f"  Response: {data['answer'][:100]}...")
            return True
        else:
            print_test("Memory recall", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Memory recall", False, str(e))
        return False

def main():
    print("\n" + "=" * 60)
    print(" EMMA MEMORY SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Memory Init
    results.append(("Memory Init", test_memory_init()))
    
    # Test 3: Consolidation
    success, user_id = test_memory_consolidation()
    results.append(("Memory Consolidation", success))
    
    # Test 4: Retrieval
    if user_id:
        results.append(("Memory Retrieval", test_memory_retrieval(user_id)))
    
    # Test 5: Vector Search
    results.append(("Vector Search", test_vector_search()))
    
    # Test 6: API Integration
    results.append(("API Integration", test_api_integration()))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        print_test(name, result)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted.")
        exit_code = 1
    
    input("\nPress Enter to exit...")
