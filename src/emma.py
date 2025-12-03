from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src import CURRENT_MODEL
from src.vectordb import add_memory, search_memory, get_memory_by_id, DOCUMENTS, METADATAS
import logging
import json

logger = logging.getLogger(__name__)

# --- Models ---

class MemoryFact(BaseModel):
    fact: str = Field(description="A concise, atomic fact extracted from the conversation.")

class MemoryExtractionResult(BaseModel):
    facts: List[MemoryFact]

class LinkDecision(BaseModel):
    is_related: bool = Field(description="Whether the new fact is related to the existing memory.")
    relation_type: str = Field(description="The type of relationship (e.g., 'expands', 'contradicts', 'related_to').", default="related_to")
    explanation: str = Field(description="Brief explanation of the relationship.")

# --- Prompts ---

EXTRACTION_SYSTEM_PROMPT = """
You are an Egocentric Memory Agent. Your goal is to extract key facts from the conversation from the perspective of the AI assistant or about the User.
Focus on:
1. User preferences, personal details, and history.
2. Important actions or decisions made by the AI.
3. Specific topics discussed.
Ignore trivial chit-chat. Return a list of atomic facts.
"""

LINKING_SYSTEM_PROMPT = """
You are a Memory Linking Agent. You are given a New Fact and an Existing Memory.
Determine if they are semantically related or if the New Fact updates/expands the Existing Memory.
"""

# --- Functions ---

def retrieve_relevant_memories(user_id: str, query: str, n_results: int = 3) -> str:
    """
    Retrieves memories relevant to the query, including linked memories (EMMA expansion).
    """
    print(f"DEBUG: EMMA: Retrieving memories for user {user_id} with query: {query}")
    logger.info(f"EMMA: Retrieving memories for user {user_id} with query: {query}")
    # 1. Basic Vector Search
    search_results = search_memory(query, n_results=n_results, where={"user_id": user_id})
    print(f"DEBUG: EMMA: Search results: {search_results}")
    logger.info(f"EMMA: Search results: {search_results}")
    
    if not search_results or not search_results.get(DOCUMENTS):
        print("DEBUG: EMMA: No memories found.")
        logger.info("EMMA: No memories found.")
        return ""

    hits = search_results[DOCUMENTS][0]
    metadatas = search_results[METADATAS][0]
    ids = search_results['ids'][0]

    context_memories = []
    seen_ids = set()

    # 2. Process Hits and Expand Links
    for i, content in enumerate(hits):
        mem_id = ids[i]
        if mem_id in seen_ids:
            continue
        
        seen_ids.add(mem_id)
        context_memories.append(f"- {content}")
        
        # Check for links in metadata
        meta = metadatas[i]
        if meta and "links" in meta:
            try:
                links = json.loads(meta["links"])
                for link_id in links:
                    if link_id not in seen_ids:
                        linked_mem = get_memory_by_id(link_id)
                        if linked_mem and linked_mem['documents']:
                            linked_content = linked_mem['documents'][0]
                            context_memories.append(f"  (Linked): {linked_content}")
                            seen_ids.add(link_id)
            except json.JSONDecodeError:
                pass

    if not context_memories:
        return ""

    return "MEMORY CONTEXT:\n" + "\n".join(context_memories) + "\n"

def consolidate_memory(user_id: str, user_input: str, ai_response: str):
    """
    Extracts facts from the turn and links them to existing memories.
    This should ideally run in the background.
    """
    conversation_text = f"User: {user_input}\nAI: {ai_response}"
    
    # 1. Extract Facts
    print(f"DEBUG: EMMA: Consolidating memory for user {user_id}")
    logger.info(f"EMMA: Consolidating memory for user {user_id}")
    extraction_agent = Agent(model=CURRENT_MODEL, result_type=MemoryExtractionResult, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    try:
        result = extraction_agent.run_sync(conversation_text)
        facts = result.data.facts
        print(f"DEBUG: EMMA: Extracted facts: {facts}")
        logger.info(f"EMMA: Extracted facts: {facts}")
    except Exception as e:
        print(f"DEBUG: Memory extraction failed: {e}")
        logger.error(f"Memory extraction failed: {e}")
        return

    linking_agent = Agent(model=CURRENT_MODEL, result_type=LinkDecision, system_prompt=LINKING_SYSTEM_PROMPT)

    for fact_obj in facts:
        fact = fact_obj.fact
        
        # 2. Find Candidates for Linking
        candidates = search_memory(fact, n_results=3, where={"user_id": user_id})
        links = []
        
        if candidates and candidates.get(DOCUMENTS):
            cand_docs = candidates[DOCUMENTS][0]
            cand_ids = candidates['ids'][0]
            
            for i, cand_doc in enumerate(cand_docs):
                cand_id = cand_ids[i]
                
                # Ask LLM if related
                prompt = f"New Fact: {fact}\nExisting Memory: {cand_doc}"
                try:
                    decision_result = linking_agent.run_sync(prompt)
                    decision = decision_result.data
                    if decision.is_related:
                        links.append(cand_id)
                        # Optional: Update the existing memory to point back (bidirectional) - skipped for simplicity
                except Exception as e:
                    logger.error(f"Linking check failed: {e}")

        # 3. Store New Memory
        metadata = {
            "user_id": user_id,
            "links": json.dumps(links),
            "created_at": str(json.dumps(str(time.time()))) # Simple timestamp
        }
        import time # Import locally to avoid top-level clutter if not needed elsewhere
        
        mem_id = add_memory(fact, metadata)
        print(f"DEBUG: EMMA: Added memory {mem_id} for fact: {fact}")
        logger.info(f"EMMA: Added memory {mem_id} for fact: {fact}")
