import json
import os
import time
import uvicorn
from enum import Enum
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import CURRENT_MODEL, DECISION_VECTOR_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPR, \
    FUNNY_SYSTEM_PROMPT, CHOOSE_ACTION_SYSTEM_PROMPT, CHOOSE_TOOL_SYSTEM_PROMPT, WARNING_SYSTEM_PROMPR, \
    BASE_API_HOST, BASE_API_PORT, MAIN_PROCESS_QUESTION, MAIN_HEALTH, MAIN_WEBHOOK

from pydantic_ai import Agent

# --- NEW: ChromaDB ---
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src.vectordb import CHROMA_DB_PATH, COLLECTION_NAME


TOOL_REQUIRED = "is_tool_required"
SERIOUS = "is_serious"
ACTIONS_REQUIRED = "is_actions_required"
ALLOWED = "is_allowed"


class Tool(str, Enum):
    none = "none"
    search_google = "google"
    watoznawca = "watoznawca"


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
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")


class Question(BaseModel):
    content: str


class Answer(BaseModel):
    answer: str
    decisionVector: DecisionVector


# ---------- LOG + AGENT RUNNER ----------

def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    print(f"Pytanie: {user_query}")
    print(f"Agent: {agent_name}")
    print(f"Odpowiedz: {response}")
    print(f"Czas odpowiedzi: {response_time:.3f} sekund")
    print("-" * 50)


def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:
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


# ---------- CHROMA DB ----------

_chroma_collection = None

def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction()
        )
    return _chroma_collection


def _answer_from_chroma(question: str) -> str | None:
    col = _get_chroma_collection()
    res = col.query(query_texts=[question], n_results=1)
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    # Try to get answer from metadata first (if you stored it there), else from document text
    if metas and metas[0] and "answer" in metas[0][0]:
        return metas[0][0]["answer"]
    if docs and docs[0]:
        try:
            obj = json.loads(docs[0][0])
            # If your jsonl lines are {"question": ..., "answer": ...}
            if isinstance(obj, dict) and "answer" in obj:
                return obj["answer"]
        except Exception:
            return docs[0][0]
    return None


# ---------- DECISION FLOW ----------

def get_decision_vector(content: str) -> DecisionVector:
    decision_vector, _ = run_agent_with_logging(
        content, "decision_vector_agent", DECISION_VECTOR_SYSTEM_PROMPT, DecisionVector
    )
    return decision_vector


def handle_default_response(content: str, decision_data: dict[str, bool]) -> str:
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "default_agent", DEFAULT_SYSTEM_PROMPR, str
    )
    return response


def handle_warning_response(content: str, decision_data: dict[str, bool]) -> str:
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "warning_agent", WARNING_SYSTEM_PROMPR, str
    )
    return response


def handle_funny_response(content: str, decision_data: dict[str, bool]) -> str:
    decision_data_s = json.dumps(decision_data)
    content = content + decision_data_s
    response, _ = run_agent_with_logging(
        content, "funny_agent", FUNNY_SYSTEM_PROMPT, str
    )
    return response


def handle_action_selection(content: str) -> str:
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


def use_tool(question: str, dane: dict) -> str:
    """Używa wybranego narzędzia do odpowiedzi na pytanie."""
    tool = dane.get("tool")
    if tool in (Tool.watoznawca, Tool.search_google):
        ans = _answer_from_chroma(question)
        if ans:
            return ans
        return "Nie znalazłem jednoznacznej odpowiedzi w lokalnej bazie."
    return "Nieznane narzędzie."


def handle_tool_selection(content: str) -> str:
    selected_tool, _ = run_agent_with_logging(
        content, "tool_agent", CHOOSE_TOOL_SYSTEM_PROMPT, Tool
    )
    return use_tool(content, {"tool": selected_tool})


# ---------- API ----------

app = FastAPI(
    title="Asystent AI - Proces Przetwarzania - Zoptymalizowany",
    description="API demonstrujące zoptymalizowany schemat blokowy przetwarzania zapytań przez AI z early stopping.",
    version="1.2.0",
)


def process_question(content: str) -> Answer:
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

        if decision_vector.is_actions_required:
            response = handle_action_selection(content)
            return Answer(answer=response, decisionVector=decision_vector)

        if decision_vector.is_tool_required:
            response = handle_tool_selection(content)
            return Answer(decisionVector=decision_vector, answer=response)
        else:
            response = handle_default_response(content, decision_data)
            return Answer(decisionVector=decision_vector, answer=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing: {str(e)}")


@app.post(MAIN_PROCESS_QUESTION, response_model=Answer)
def process_question_endpoint(question: Question):
    return process_question(question.content)


@app.post(MAIN_WEBHOOK)
def webhook(payload: Dict[str, Any]):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question_endpoint(question)
    return {"output": answer.answer}


@app.get(MAIN_HEALTH)
def health():
    return {"ok": True}

@app.get("/debug/search")
def debug_search(q: str):
    """
    Debug endpoint: search ChromaDB directly for the closest QA entry.
    """
    from src.main import _answer_from_chroma
    answer = _answer_from_chroma(q)
    return {"query": q, "answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host=BASE_API_HOST, port=BASE_API_PORT)
