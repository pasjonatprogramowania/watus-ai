import json
import time
import logging
import uvicorn
from enum import Enum
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import CURRENT_MODEL, DECISION_VECTOR_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPR, \
    FUNNY_SYSTEM_PROMPT, CHOOSE_ACTION_SYSTEM_PROMPT, CHOOSE_TOOL_SYSTEM_PROMPT, WARNING_SYSTEM_PROMPR, \
    BASE_API_HOST, BASE_API_PORT, MAIN_PROCESS_QUESTION, MAIN_HEALTH, MAIN_WEBHOOK

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from src.vectordb import search_vector_db, return_collection
from src.emma import retrieve_relevant_memories, consolidate_memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_search.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

TOOL_REQUIRED = "is_tool_required"
SERIOUS = "is_serious"
ACTIONS_REQUIRED = "is_actions_required"
ALLOWED = "is_allowed"
DISTANCES = 'distances'
METADATAS = 'metadatas'
DOCUMENTS = 'documents'

# Tool choices
DUCKDUCKGO_TOOL = "google"
WATOZNAWCA_TOOL = "watoznawca"


class Action(str, Enum):
    follow_action = "sledzenie"
    end_action = "koniec_sledzenia"


class WatusActiveState(BaseModel):
    following: bool


class DecisionVector(BaseModel):
    """[Dozwolone?, Czy_działanie?, Poważne?, Potrzeba_info?]"""
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: str = Field(DUCKDUCKGO_TOOL, description=f"Tool needed: {DUCKDUCKGO_TOOL} for web search, {WATOZNAWCA_TOOL} for WAT knowledge.")


class Question(BaseModel):
    content: str


class VectorSearchRequest(BaseModel):
    query: str
    n_results: int = 3


class VectorSearchResult(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class Answer(BaseModel):
    answer: str
    decisionVector: DecisionVector



def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    """Loguje odpowiedź LLM z pomiarem czasu."""
    print(f"Pytanie: {user_query}")
    print(f"Agent: {agent_name}")
    print(f"Odpowiedz: {response}")
    print(f"Czas odpowiedzi: {response_time:.3f} sekund")
    print("-" * 50)


def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:
    """Uruchamia agenta z logowaniem czasu odpowiedzi."""
    agent = Agent(
        model=CURRENT_MODEL,
        output_type=output_type,
        system_prompt=system_prompt
    )
    start_time = time.time()
    result = agent.run_sync(content)
    response_time = time.time() - start_time
    output = result.output
    log_llm_response(content, agent_name, str(output), response_time)
    return output, response_time


def get_decision_vector(content: str) -> DecisionVector:
    """Sprawdza pytanie pod kątem wszystkich kategorii decyzyjnych w jednym requestcie."""
    decision_vector, _ = run_agent_with_logging(
        content, "decision_vector_agent", DECISION_VECTOR_SYSTEM_PROMPT, DecisionVector
    )
    return decision_vector


def handle_default_response(content: str, decision_data: dict[str, Any]) -> str:
    """Obsługuje odpowiedź domyślną."""
    tools = []
    if decision_data[TOOL_REQUIRED] == DUCKDUCKGO_TOOL:
        tools.append(duckduckgo_search_tool())

    agent = Agent(
        model=CURRENT_MODEL,
        output_type=str,
        system_prompt=DEFAULT_SYSTEM_PROMPR,
        tools=tools
    )
    result = agent.run_sync(content)
    return result.output


def handle_warning_response(content: str,decision_data: dict[str, bool]) -> str:
    """Obsługuje ostrzeżenie dla niedozwolonych pytań."""
    decision_data_s = json.dumps(decision_data)
    content = content+decision_data_s
    response, _ = run_agent_with_logging(
        content, "warning_agent", WARNING_SYSTEM_PROMPR, str
    )
    return response


def handle_funny_response(content: str ,decision_data: dict[str, bool]) -> str:
    """Obsługuje żartobliwą odpowiedź dla niepoważnych pytań."""
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "funny_agent", FUNNY_SYSTEM_PROMPT, str
    )
    return response


def handle_action_selection(content: str) -> str:
    """Obsługuje wybór i wykonanie akcji."""
    selected_action, _ = run_agent_with_logging(
        content, "action_agent", CHOOSE_ACTION_SYSTEM_PROMPT, Action
    )

    if selected_action == Action.follow_action:
        WatusActiveState.following = True
        return "Rozpoczęto śledzenie."
    elif selected_action == Action.end_action:
        WatusActiveState.following = False
        return "Zakończono śledzenie."
    else:
        return "Nieznana akcja."

def handle_context_response(content: str) -> str:
    """Obsługuje odpowiedź na podstawie kontekstu."""
    context = check_context(content)
    return f"Odpowiedź na podstawie kontekstu: {context}"

def vector_search(query: str) -> VectorSearchResult:
    """Perform vector search in the database with enhanced error handling and logging"""
    n_results = 3
    try:
        if n_results <= 0:
            logger.error(f"Invalid number of results requested: {n_results}")
            raise HTTPException(
                status_code=400,
                detail=f"Number of results must be positive, got {n_results}"
            )
        collection = return_collection()
        if not collection:
            logger.error("Vector database initialization failed")
            raise HTTPException(
                status_code=500,
                detail="Vector database not initialized. Please check database configuration."
            )
        if not query or len(query.strip()) == 0:
            logger.error("Empty search query provided")
            raise HTTPException(
                status_code=400,
                detail="Search query cannot be empty"
            )
        search_results = search_vector_db(collection, query, n_results)
        if not search_results or not search_results.get(DOCUMENTS, []):
            logger.warning(f"No results found for query: {query}")
            raise HTTPException(
                status_code=404,
                detail="No matching documents found in the vector database"
            )
        logger.info(f"Search Results Found: {len(search_results[DOCUMENTS][0])} documents")
        result = VectorSearchResult(
            documents=search_results[DOCUMENTS][0],
            metadatas=search_results[METADATAS][0],
            distances=search_results[DISTANCES][0]
        )
        logger.info("Vector search completed successfully")
        return result

    except HTTPException as http_err:
        # Log HTTP exceptions
        logger.error(f"HTTP Error in Vector Search: {http_err.detail}")
        raise
    except Exception as e:
        # Log and handle unexpected errors
        logger.error(f"Unexpected error during vector search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during vector search: {str(e)}"
        )


app = FastAPI(
    title="Asystent AI - Proces Przetwarzania - Zoptymalizowany",
    description="API demonstrujące zoptymalizowany schemat blokowy przetwarzania zapytań przez AI z early stopping.",
    version="1.2.0",
)


def process_question(content: str) -> Answer:
    """Główna funkcja przetwarzająca pytanie z optymalizacją early stopping."""
    try:
        decision_vector = get_decision_vector(content)

        decision_data = {
            ALLOWED: decision_vector.is_allowed,
            ACTIONS_REQUIRED: decision_vector.is_actions_required,
            SERIOUS: decision_vector.is_serious,
            TOOL_REQUIRED: decision_vector.is_tool_required
        }

        if not decision_vector.is_allowed:
            response = handle_warning_response(content, decision_data)
            return Answer(answer=response, decisionVector=decision_vector)

        if not decision_vector.is_serious:
            response = handle_funny_response(content, decision_data)
            return Answer(answer=response, decisionVector=decision_vector)

        # if decision_vector.is_actions_required:
        #     response = handle_action_selection(content)
        #     return Answer(answer=response, decisionVector=decision_vector)

        if decision_data[TOOL_REQUIRED] != None:
            try:
                vector_result = vector_search(content)
                additional_info = f"\nInformacje z bazy wektorowej: {''.join([doc for doc in vector_result.documents])}"
                content += additional_info
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # --- EMMA Integration ---
        # 1. Retrieve Memory
        user_id = "default_user" # Hardcoded for now as per plan
        try:
            memory_context = retrieve_relevant_memories(user_id, content)
            if memory_context:
                content += f"\n\n{memory_context}"
        except Exception as e:
            logger.error(f"EMMA retrieval failed: {e}")

        response = handle_default_response(content, decision_data)

        # 2. Consolidate Memory (Post-Generation)
        try:
            # In production, use BackgroundTasks
            consolidate_memory(user_id, content, response)
        except Exception as e:
            logger.error(f"EMMA consolidation failed: {e}")

        return Answer(answer=response, decisionVector=decision_vector)


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing: {str(e)}")


def check_context(content: str) -> Dict[str, Any]:
    """Sprawdza kontekst dla danego pytania."""
    return {"context": "Przykładowy kontekst na podstawie zapytania."}


@app.post(MAIN_PROCESS_QUESTION, response_model=Answer)
def process_question_endpoint(question: Question):
    """Endpoint do przetwarzania pytań."""
    return process_question(question.content)


@app.post("/api1/webhook")
def webhook_api1(payload: Dict[str, Any]):
    """Webhook endpoint for audio agent."""
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question_endpoint(question)
    return {"output": answer.answer}

@app.post(MAIN_WEBHOOK)
def webhook(payload: Dict[str, Any]):
    """Webhook endpoint dla zewnętrznych integracji."""
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question_endpoint(question)
    return {"output": answer.answer}


@app.get(MAIN_HEALTH)
def health():
    """Health check endpoint."""
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host=BASE_API_HOST, port=BASE_API_PORT)