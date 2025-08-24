import os
import time
import uvicorn
from enum import Enum
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import CURRENT_MODEL, IS_ALLOWED_SYSTEM_PROMPT, WARNING_SYSTEM_PROMPT, IS_SERIOUS_SYSTEM_PROMPT, \
    FUNNY_SYSTEM_PROMPT, IS_ACTIONS_REQUIRED_SYSTEM_PROMPT, CHOOSE_ACTION_SYSTEM_PROMPT, IS_TOOL_REQUIRED_SYSTEM_PROMPT, \
    CHOOSE_TOOL_SYSTEM_PROMPT

from pydantic_ai import Agent


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


def check_if_allowed(content: str) -> bool:
    """Sprawdza czy pytanie jest dozwolone zgodnie z polityką."""
    is_allowed, _ = run_agent_with_logging(
        content, "allowed_agent", IS_ALLOWED_SYSTEM_PROMPT, bool
    )
    return is_allowed


def handle_warning_response(content: str) -> str:
    """Obsługuje ostrzeżenie dla niedozwolonych pytań."""
    response, _ = run_agent_with_logging(
        content, "warning_agent", WARNING_SYSTEM_PROMPT, str
    )
    return response


def check_if_serious(content: str) -> bool:
    """Sprawdza czy pytanie jest poważne."""
    is_serious, _ = run_agent_with_logging(
        content, "serious_agent", IS_SERIOUS_SYSTEM_PROMPT, bool
    )
    return is_serious


def handle_funny_response(content: str) -> str:
    """Obsługuje żartobliwą odpowiedź dla niepoważnych pytań."""
    response, _ = run_agent_with_logging(
        content, "funny_agent", FUNNY_SYSTEM_PROMPT, str
    )
    return response


def check_if_actions_required(content: str) -> bool:
    """Sprawdza czy pytanie wymaga wykonania akcji."""
    is_actions_required, _ = run_agent_with_logging(
        content, "actions_agent", IS_ACTIONS_REQUIRED_SYSTEM_PROMPT, bool
    )
    return is_actions_required


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


def check_if_tool_required(content: str) -> bool:
    """Sprawdza czy pytanie wymaga użycia dodatkowych narzędzi."""
    is_tool_required, _ = run_agent_with_logging(
        content, "more_info_agent", IS_TOOL_REQUIRED_SYSTEM_PROMPT, bool
    )
    return is_tool_required


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
    last_answer: str
    decisionVector: DecisionVector


def optimized_process_question(content: str) -> Answer:
    """Główna funkcja przetwarzająca pytanie z optymalizacją early stopping."""
    decision_data = {
        "is_allowed": True,
        "is_actions_required": False,
        "is_serious": True,
        "is_tool_required": False
    }

    try:
        is_allowed = check_if_allowed(content)
        decision_data["is_allowed"] = is_allowed

        if not is_allowed:
            response = handle_warning_response(content)
            decision_vector = DecisionVector(**decision_data)
            return Answer(last_answer=response, decisionVector=decision_vector)

        is_serious = check_if_serious(content)
        decision_data["is_serious"] = is_serious

        if not is_serious:
            response = handle_funny_response(content)
            decision_vector = DecisionVector(**decision_data)
            return Answer(last_answer=response, decisionVector=decision_vector)

        is_actions_required = check_if_actions_required(content)
        decision_data["is_actions_required"] = is_actions_required

        if is_actions_required:
            response = handle_action_selection(content)
            decision_vector = DecisionVector(**decision_data)
            return Answer(last_answer=response, decisionVector=decision_vector)

        is_tool_required = check_if_tool_required(content)
        decision_data["is_tool_required"] = is_tool_required

        if is_tool_required:
            response = handle_tool_selection(content)
        else:
            response = handle_context_response(content)

        decision_vector = DecisionVector(**decision_data)
        return Answer(last_answer=response, decisionVector=decision_vector)

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


@app.post("/process_question", response_model=Answer)
def process_question(question: Question):
    """Endpoint do przetwarzania pytań."""
    return optimized_process_question(question.content)


@app.post("/webhook")
def webhook(payload: Dict[str, Any]):
    """Webhook endpoint dla zewnętrznych integracji."""
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question(question)
    return {"output": answer.last_answer}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "following": app.state.following}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))