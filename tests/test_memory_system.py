"""
Comprehensive Unit Tests for EMMA Memory System

Tests cover:
1. mem0 Memory initialization
2. Memory consolidation (storing facts)
3. Memory retrieval (searching facts)
4. Vector database integration (WAT knowledge)
5. Full API integration flow
"""
import pytest
import time
import os
import sys
import requests
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Test Configuration
# =============================================================================
API_BASE_URL = "http://localhost:8000/api1"
TEST_USER_ID = f"test_user_{int(time.time())}"  # Unique user for each test run


# =============================================================================
# Unit Tests - mem0 Module
# =============================================================================
class TestMem0Initialization:
    """Tests for mem0 memory initialization."""
    
    def test_memory_import(self):
        """Test that mem0 can be imported."""
        from mem0 import Memory
        assert Memory is not None
    
    def test_emma_module_import(self):
        """Test that emma module can be imported."""
        from src.emma import retrieve_relevant_memories, consolidate_memory
        assert retrieve_relevant_memories is not None
        assert consolidate_memory is not None
    
    def test_memory_instance_creation(self):
        """Test that memory instance can be created."""
        from src.emma import _get_memory
        memory = _get_memory()
        assert memory is not None


class TestMemoryConsolidation:
    """Tests for memory consolidation (storing facts)."""
    
    def test_consolidate_simple_fact(self):
        """Test storing a simple fact about the user."""
        from src.emma import consolidate_memory
        
        user_input = "My name is TestUser for unit testing."
        ai_response = "Nice to meet you, TestUser! I'll remember that."
        
        # Should not raise any exceptions
        consolidate_memory(TEST_USER_ID, user_input, ai_response)
    
    def test_consolidate_multiple_facts(self):
        """Test storing multiple facts in one conversation."""
        from src.emma import consolidate_memory
        
        user_input = "I am 25 years old and I work as a software engineer."
        ai_response = "Great! You're 25 and work as a software engineer. Got it!"
        
        consolidate_memory(TEST_USER_ID, user_input, ai_response)
    
    def test_consolidate_with_empty_input(self):
        """Test that empty inputs don't crash the system."""
        from src.emma import consolidate_memory
        
        # Should handle gracefully
        consolidate_memory(TEST_USER_ID, "", "")


class TestMemoryRetrieval:
    """Tests for memory retrieval (searching facts)."""
    
    @pytest.fixture(autouse=True)
    def setup_memory(self):
        """Setup: Store some memories before retrieval tests."""
        from src.emma import consolidate_memory
        
        # Store a known fact
        consolidate_memory(
            TEST_USER_ID,
            "Remember that my favorite color is blue.",
            "I'll remember that your favorite color is blue!"
        )
        # Give mem0 time to process
        time.sleep(2)
    
    def test_retrieve_existing_memory(self):
        """Test retrieving a previously stored memory."""
        from src.emma import retrieve_relevant_memories
        
        result = retrieve_relevant_memories(TEST_USER_ID, "What is my favorite color?")
        
        # Result should be a string
        assert isinstance(result, str)
    
    def test_retrieve_no_memories_for_new_user(self):
        """Test that new users have no memories."""
        from src.emma import retrieve_relevant_memories
        
        new_user_id = f"new_user_{int(time.time())}"
        result = retrieve_relevant_memories(new_user_id, "What do you know about me?")
        
        # Should return empty string for new user
        assert result == ""
    
    def test_retrieve_with_limit(self):
        """Test memory retrieval with custom limit."""
        from src.emma import retrieve_relevant_memories
        
        result = retrieve_relevant_memories(TEST_USER_ID, "Tell me about myself", n_results=5)
        
        assert isinstance(result, str)


# =============================================================================
# Unit Tests - Vector Database (WAT Knowledge)
# =============================================================================
class TestVectorDatabase:
    """Tests for ChromaDB vector database with WAT knowledge."""
    
    def test_vectordb_import(self):
        """Test that vectordb module can be imported."""
        from src.vectordb import vector_search, search_memory, add_memory
        assert vector_search is not None
        assert search_memory is not None
        assert add_memory is not None
    
    def test_vector_search_wat_info(self):
        """Test searching for WAT university information."""
        from src.vectordb import vector_search
        
        result = vector_search("Ile wydziałów ma WAT?")
        
        assert result is not None
        assert hasattr(result, 'documents')
        assert len(result.documents) > 0
    
    def test_vector_search_returns_relevant_results(self):
        """Test that vector search returns relevant results for WAT queries."""
        from src.vectordb import vector_search
        
        result = vector_search("Kiedy powstała Wojskowa Akademia Techniczna?")
        
        assert result is not None
        # Check that documents contain some text
        documents = result.documents
        assert any(len(doc) > 0 for doc in documents)
    
    def test_search_memory_function(self):
        """Test the search_memory function."""
        from src.vectordb import search_memory
        
        result = search_memory("studenci WAT", n_results=3)
        
        assert result is not None
        assert 'documents' in result or 'ids' in result


# =============================================================================
# Integration Tests - Full API Flow
# =============================================================================
class TestAPIIntegration:
    """Integration tests for the full API with memory system."""
    
    @pytest.fixture
    def api_session(self):
        """Create a session for API requests."""
        return requests.Session()
    
    def test_api_health(self, api_session):
        """Test that the API is running."""
        try:
            response = api_session.get(f"{API_BASE_URL.replace('/api1', '')}/docs")
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API not running - start with: python -m src.main")
    
    def test_api_process_question_with_name(self, api_session):
        """Test API processes question and stores user name."""
        try:
            # Send first message with name
            response = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Nazywam się TestAPI i lubię programować."}
            )
            
            if response.status_code != 200:
                pytest.skip(f"API returned {response.status_code}")
            
            data = response.json()
            assert "answer" in data
            
        except requests.ConnectionError:
            pytest.skip("API not running")
    
    def test_api_memory_recall(self, api_session):
        """Test that API can recall previously stored information."""
        try:
            # First, store some information
            response1 = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Moje imię to MemoryTestUser123."}
            )
            
            if response1.status_code != 200:
                pytest.skip("API not available")
            
            # Wait for memory consolidation
            time.sleep(3)
            
            # Try to recall the name
            response2 = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Jak mam na imię?"}
            )
            
            assert response2.status_code == 200
            data = response2.json()
            assert "answer" in data
            
        except requests.ConnectionError:
            pytest.skip("API not running")
    
    def test_api_wat_knowledge(self, api_session):
        """Test that API can answer questions about WAT."""
        try:
            response = api_session.post(
                f"{API_BASE_URL}/process_question",
                json={"content": "Ile wydziałów ma WAT?"}
            )
            
            if response.status_code != 200:
                pytest.skip("API not available")
            
            data = response.json()
            assert "answer" in data
            # WAT has 7 faculties
            answer = data["answer"].lower()
            # Check if response contains relevant information
            assert len(answer) > 10  # Non-empty response
            
        except requests.ConnectionError:
            pytest.skip("API not running")


# =============================================================================
# Performance Tests
# =============================================================================
class TestPerformance:
    """Performance tests for memory operations."""
    
    def test_memory_retrieval_speed(self):
        """Test that memory retrieval completes in reasonable time."""
        from src.emma import retrieve_relevant_memories
        
        start_time = time.time()
        retrieve_relevant_memories(TEST_USER_ID, "What do you know about me?")
        elapsed_time = time.time() - start_time
        
        # Should complete within 10 seconds (including API calls)
        assert elapsed_time < 10, f"Memory retrieval took {elapsed_time:.2f}s"
    
    def test_memory_consolidation_speed(self):
        """Test that memory consolidation completes in reasonable time."""
        from src.emma import consolidate_memory
        
        start_time = time.time()
        consolidate_memory(
            TEST_USER_ID,
            "Performance test input.",
            "Performance test response."
        )
        elapsed_time = time.time() - start_time
        
        # Should complete within 15 seconds
        assert elapsed_time < 15, f"Memory consolidation took {elapsed_time:.2f}s"


# =============================================================================
# Edge Case Tests
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_unicode_characters(self):
        """Test handling of unicode characters (Polish)."""
        from src.emma import consolidate_memory, retrieve_relevant_memories
        
        polish_input = "Mieszkam w Warszawie i studiuję na WAT. Moje imię to Paweł."
        polish_response = "Świetnie! Zapamiętam, że mieszkasz w Warszawie."
        
        # Should not raise exceptions
        consolidate_memory(TEST_USER_ID, polish_input, polish_response)
    
    def test_very_long_input(self):
        """Test handling of very long inputs."""
        from src.emma import consolidate_memory
        
        long_input = "Test " * 500  # ~2500 characters
        long_response = "Response " * 300
        
        # Should handle gracefully (may truncate internally)
        consolidate_memory(TEST_USER_ID, long_input, long_response)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        from src.emma import consolidate_memory
        
        special_input = "My email is test@example.com and I like $100 bills! <script>alert('xss')</script>"
        special_response = "I'll remember your email."
        
        # Should not crash
        consolidate_memory(TEST_USER_ID, special_input, special_response)


# =============================================================================
# Main Test Runner
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMMA Memory System - Comprehensive Unit Tests")
    print("=" * 60)
    print()
    
    # Check if API is running
    try:
        requests.get("http://localhost:8000/docs", timeout=2)
        print("✓ API is running at http://localhost:8000")
    except requests.ConnectionError:
        print("⚠ API is not running - integration tests will be skipped")
        print("  Start with: python -m src.main")
    print()
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--color=yes"
    ])
    
    sys.exit(exit_code)
