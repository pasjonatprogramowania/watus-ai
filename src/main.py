import os
import uvicorn
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import CURRENT_MODEL

from pydantic_ai import Agent
is_allowed_system_prompt = """
Jesteś AI, które określa, czy zapytanie użytkownika jest dozwolone zgodnie z polityką. 
Sprawdź pod kątem wulgarności, niemoralności lub niedozwolonej treści. 
Output: true jeśli dozwolone, false jeśli nie.

Przykłady:
- Query: "Ile zarabia dziekan" Output: true 
- Query: "Opowiedz wulgarny żart." Output: false 
- Query: "Obraź kogoś" Output: false 
- Query: "Podaj przepis na sałatkę." Output: true 

Na podstawie zapytania użytkownika outputuj tylko true lub false.
"""

is_actions_required_system_prompt = """
Jesteś AI, które określa, czy zapytanie użytkownika wymaga wykonania akcji, 
takiej jak śledzenie kogoś lub inne interaktywne zachowania. 
Output: true jeśli akcja jest wymagana, false w przeciwnym razie.

Przykłady:
- Query: "Zacznij mnie śledzić." Output: true (Wymaga akcji śledzenia.)
- Query: "Jaka jest pogoda?" Output: false (Nie wymaga interaktywnej akcji.)
- Query: "Zakończ śledzenie." Output: true (Wymaga akcji zakończenia.)
- Query: "Opowiedz dowcip." Output: false (To tylko prośba o informację, bez akcji.)

Na podstawie zapytania użytkownika outputuj tylko true lub false.
"""

is_serious_system_prompt = """
Jesteś AI, które określa, czy zapytanie użytkownika jest poważne, czy jest to żart lub drwina. 
Output: true jeśli poważne, false jeśli to żart.

Przykłady:
- Query: "Czy polecasz WAT" Output: true 
- Query: "Dlaczego jesteś gadającą puszką" Output: false 
- Query: "Ile zarabia dziekan" Output: true 
- Query: "Powiedz mi jak wytrzymujesz tutaj" Output: false 

Na podstawie zapytania użytkownika outputuj tylko true lub false.
"""

is_tool_required_system_prompt = """
Jesteś AI, które określa, czy zapytanie użytkownika wymaga użycia zewnętrznych narzędzi lub więcej informacji, 
aby odpowiedzieć poprawnie. Output: true jeśli potrzebne więcej info lub narzędzi, false w przeciwnym razie.

Przykłady:
- Query: "Jakie są kierunki na Wacie?" Output: true (Wymaga narzędzia do sprawdzania wiedzy o WAT)
- Query: "Ile to 2 + 2?" Output: false (Prosta kalkulacja, nie potrzeba narzędzi.)
- Query: "Szukaj w Google o historii Polski." Output: true (Wymaga zewnętrznego narzędzia wyszukiwania.)
- Query: "Powiedz 'cześć'." Output: false (Nie potrzeba dodatkowych informacji.)

Na podstawie zapytania użytkownika outputuj tylko true lub false.
"""

choose_tool_system_prompt = """
Jesteś AI, które wybiera odpowiednie narzędzie na podstawie zapytania użytkownika i opisów dostępnych narzędzi. 
Dostępne narzędzia:
- google: Użyj do ogólnego wyszukiwania w internecie, np. aktualnych wiadomości, faktów lub ogólnej wiedzy.
- watoznawca: Użyj do specjalistycznej wiedzy o Wojskowej Akademii Technicznej (WAT), np. kierunki studiów, historia, kadra czy wydarzenia na WAT.

Output: Nazwa wybranego narzędzia (google lub watoznawca). Wybierz tylko jedno, najbardziej pasujące. Jeśli żadne nie pasuje, wybierz google jako domyślne.

Przykłady:
- Query: "Jaka jest pogoda w Warszawie?" Output: google (Wymaga wyszukiwania w internecie.)
- Query: "Jakie są kierunki studiów na WAT?" Output: watoznawca (Specjalistyczna wiedza o WAT.)
- Query: "Kto jest prezydentem Polski?" Output: google (Ogólna wiedza, wyszukiwanie w internecie.)
- Query: "Ile zarabia dziekan WAT?" Output: watoznawca (Związane z kadrą WAT.)

Na podstawie zapytania użytkownika outputuj tylko nazwę narzędzia (np. google lub watoznawca).
"""

choose_action_system_prompt = """
Jesteś AI, które wybiera odpowiednią akcję na podstawie zapytania użytkownika i opisów dostępnych akcji. 
Dostępne akcje:
- sledzenie: Użyj, gdy użytkownik prosi o rozpoczęcie śledzenia lub monitorowania.
- koniec_sledzenia: Użyj, gdy użytkownik prosi o zakończenie śledzenia lub zatrzymanie monitorowania.

Output: Nazwa wybranego działania (sledzenie lub koniec_sledzenia). Wybierz tylko jedno, najbardziej pasujące. Jeśli żadne nie pasuje, wybierz sledzenie jako domyślne.

Przykłady:
- Query: "Zacznij mnie śledzić." Output: sledzenie (Prośba o rozpoczęcie śledzenia.)
- Query: "Przestań mnie obserwować." Output: koniec_sledzenia (Prośba o zakończenie.)
- Query: "Rozpocznij monitorowanie." Output: sledzenie (Podobne do śledzenia.)
- Query: "Zakończ wszystko." Output: koniec_sledzenia (Prośba o zakończenie akcji.)

Na podstawie zapytania użytkownika outputuj tylko nazwę akcji (np. sledzenie lub koniec_sledzenia).
"""

app = FastAPI(
    title="Asystent AI - Proces Przetwarzania",
    description="API demonstrujące schemat blokowy przetwarzania zapytań przez AI.",
    version="1.1.0",
)
app.state.following = False  # stan trybu śledzenia


class Tool(str, Enum):
    search_google = "google"
    watoznawca = "watoznawca"


class Action(str, Enum):
    follow_action = "sledzenie"
    end_action = "koniec_sledzenia"


class DecisionVector(BaseModel):
    """[Dozwolone?, Czy_działanie?, Poważne?, Potrzeba_info?]"""
    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")


class RouterFailure(BaseModel):
    """Use me when no appropriate agent is found or the used agent failed."""
    explanation: str


class Question(BaseModel):
    content: str


class Answer(BaseModel):
    last_answer: str
    decisionVector: DecisionVector


def validate_question(content: str) -> DecisionVector:
    """
    Analyzes the user query using separate Agents for each flag in DecisionVector.
    This ensures deterministic and structured output using Pydantic.
    """
    try:
        allowed_agent = Agent(
            model=CURRENT_MODEL,
            output_type=bool,
            system_prompt=is_allowed_system_prompt
        )
        is_allowed = allowed_agent.run_sync(content).output

        actions_agent = Agent(
            model=CURRENT_MODEL,
            output_type=bool,
            system_prompt=is_actions_required_system_prompt
        )
        is_actions_required = actions_agent.run_sync(content).output

        serious_agent = Agent(
            model=CURRENT_MODEL,
            output_type=bool,
            system_prompt=is_serious_system_prompt
        )
        is_serious = serious_agent.run_sync(content).output

        more_info_agent = Agent(
            model=CURRENT_MODEL,
            output_type=bool,
            system_prompt=is_tool_required_system_prompt
        )
        is_more_info_required = more_info_agent.run_sync(content).output

        return DecisionVector(
            is_allowed=is_allowed,
            is_actions_required=is_actions_required,
            is_serious=is_serious,
            is_tool_required=is_more_info_required
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in validation: {str(e)}")


def ask_second_time() -> str:
    return "Nie mogę odpowiedzieć na to pytanie w tej formie. Czy możesz je przeformułować?"


def chose_required_action(content: str) -> str:
    try:
        action_agent = Agent(
            model=CURRENT_MODEL,
            output_type=Action,  # Output to enum Action
            system_prompt=choose_action_system_prompt
        )
        selected_action = action_agent.run_sync(content).output

        if selected_action == Action.follow_action:
            app.state.following = True
            return "Rozpoczęto śledzenie."
        elif selected_action == Action.end_action:
            app.state.following = False
            return "Zakończono śledzenie."
        return "Nieznana akcja."
    except Exception as e:
        app.state.following = True
        return "Rozpoczęto śledzenie (domyślna akcja)."


def chose_required_tool(content: str) -> Tool:
    try:
        tool_agent = Agent(
            model=CURRENT_MODEL,
            output_type=Tool,
            system_prompt=choose_tool_system_prompt
        )
        selected_tool = tool_agent.run_sync(content).output
        return selected_tool
    except Exception as e:
        return Tool.search_google

def check_context(content: str) -> Dict[str, Any]:
    return {"context": "Przykładowy kontekst na podstawie zapytania."}


def give_funny_response() -> str:
    return "Haha, to było zabawne! Ale seriously, co masz na myśli?"


def use_tool(question: str, dane: dict) -> Dict[str, Any]:
    tool = dane.get("tool")
    if tool == Tool.search_google:
        return {"result": "Wyniki wyszukiwania z Google."}
    elif tool == Tool.watoznawca:
        return {"result": "Informacje z Watoznawcy o WAT."}
    return {"result": "Nieznane narzędzie."}

@app.post("/process_question", response_model=Answer)
def process_question(question: Question):
    decision = validate_question(question.content)

    if not decision.is_allowed:
        return Answer(last_answer=ask_second_time(), decisionVector=decision)

    if not decision.is_serious:
        return Answer(last_answer=give_funny_response(), decisionVector=decision)

    if decision.is_actions_required:
        response = chose_required_action(question.content)  # Dynamicznie wybiera i wykonuje na bazie content
        return Answer(last_answer=response, decisionVector=decision)

    if decision.is_tool_required:
        tool = chose_required_tool(question.content)  # Dynamicznie wybiera na bazie content
        result = use_tool(question.content, {"tool": tool})
        return Answer(last_answer=str(result), decisionVector=decision)

    # Default response if no special handling
    context = check_context(question.content)
    return Answer(last_answer=f"Odpowiedź na podstawie kontekstu: {context}", decisionVector=decision)


@app.post("/webhook")
def webhook(payload: Dict[str, Any]):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    question = Question(content=prompt)
    answer = process_question(question)
    return {"output": answer.last_answer}

@app.get("/health")
def health():
    return {"ok": True, "following": app.state.following}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))