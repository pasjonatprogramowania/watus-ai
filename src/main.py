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


def handle_default_response(content: str,decision_data: dict[str, bool]) -> str:
    """Obsługuje ostrzeżenie dla niedozwolonych pytań."""
    decision_data_s = json.dumps(decision_data)
    content = content+decision_data_s
    response, _ = run_agent_with_logging(
        content, "default_agent", DEFAULT_SYSTEM_PROMPR, str
    )
    return response

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




def handle_tool_selection(content: str) -> str:
    """Obsługuje wybór i użycie narzędzia."""
    selected_tool, _ = run_agent_with_logging(
        content, "tool_agent", CHOOSE_TOOL_SYSTEM_PROMPT, Tool
    )

    result = use_tool(content, {"tool": selected_tool})
    return str(result)


def handle_context_response(content: str) -> str:
    """Obsługuje odpowiedź na podstawie kontekstu."""
    context = check_context(content)
    return f"Odpowiedź na podstawie kontekstu: {context}"


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


def check_context(content: str) -> Dict[str, Any]:
    """Sprawdza kontekst dla danego pytania."""
    return {"context": "Przykładowy kontekst na podstawie zapytania."}


def use_tool(question: str, dane: dict) -> Dict[str, Any]:
    """Używa wybranego narzędzia do odpowiedzi na pytanie."""
    tool = dane.get("tool")
    if tool == Tool.search_google:
        return {"result": "Wyniki wyszukiwania z Google."}
    elif tool == Tool.watoznawca:
        return {"result": "Informacje z Watoznawcy o WAT."}
    return {"result": "Nieznane narzędzie."}


@app.post(MAIN_PROCESS_QUESTION, response_model=Answer)
def process_question_endpoint(question: Question):
    """Endpoint do przetwarzania pytań."""
    return process_question(question.content)


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