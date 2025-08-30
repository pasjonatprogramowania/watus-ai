# tests/test_pipeline_new.py
import json
import pytest
from fastapi.testclient import TestClient
import src.main as main


# ---------- Deterministic stubs ----------

def _stub_run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type):
    """
    Return stable outputs for each agent so unit tests don't call a real LLM.
    We mimic your original heuristics and the required booleans.
    """
    q = content.lower()

    # 1) booleans
    if output_type is bool:
        if agent_name == "allowed_agent":
            return True, 0.0

        if agent_name == "serious_agent":
            # Not serious if: salary question, girls question, PW vs WAT debate, or that "pani w różowym" imperative
            jokey = (
                "ile zarabia" in q or
                "dziewczyn" in q or
                ("pw" in q and "wat" in q) or
                (("idź" in q or "idz" in q) and "pani" in q)
            )
            return (not jokey), 0.0

        if agent_name == "actions_agent":
            # Action for follow/stop only
            is_action = any(t in q for t in ["chodź", "chodz", "śledź", "sledz"]) or \
                        any(t in q for t in ["przestań", "przestan", "koniec śledzenia"])
            return is_action, 0.0

        if agent_name == "more_info_agent":
            # Tool needed for WAT informational queries
            needs_tool = any(t in q for t in [
                "wat", "wojskowa akademia techniczna", "wydział", "wydzial",
                "kierunk", "studiu", "histori", "dziekan"
            ])
            return needs_tool, 0.0

        # default
        return False, 0.0

    # 2) enums: Action
    if output_type.__name__ == "Action" or output_type == main.Action:
        if any(t in q for t in ["przestań", "przestan", "koniec śledzenia"]):
            return main.Action.end_action, 0.0
        # default follow if action required
        return main.Action.follow_action, 0.0

    # 3) enums: Tool
    if output_type.__name__ == "Tool" or output_type == main.Tool:
        if any(t in q for t in [
            "wat", "wojskowa akademia techniczna", "wydział", "wydzial",
            "kierunk", "studiu", "histori", "dziekan"
        ]):
            return main.Tool.watoznawca, 0.0
        return main.Tool.search_google, 0.0

    # 4) strings: WARNING/FUNNY/etc. (we only need funny text deterministically)
    if agent_name == "funny_agent":
        if "ile zarabia" in q:
            return "Myślę, że wystarczyłoby na bardzo szybki komputer… i kilka lat kawy z automatu. 😏", 0.0
        if "dziewczyn" in q:
            return "Są! Ale pamiętaj, że ja jestem tylko robotem — mogę co najwyżej wysłać im miłego mema. 😉", 0.0
        if "pw" in q and "wat" in q:
            return "To zależy, kogo zapytasz — studenci PW i WAT mają tu zupełnie różne zdania. 😏", 0.0
        if (("idź" in q or "idz" in q) and "pani" in q):
            return "Hmm… wolałbym, żebyś sam jej przekazał, że jestem zajęty. 😄", 0.0
        return "Powiedzmy, że odpowiedź jest jak tajny budżet dziekana — wolę zostać przy żarcie. 😏", 0.0

    if agent_name == "warning_agent":
        return "Nie mogę odpowiedzieć na to pytanie w tej formie. Czy możesz je przeformułować?", 0.0

    # generic fallback
    return "OK", 0.0


def _stub_use_tool(question: str, dane: dict):
    """
    Stable tool results so 'last_answer' is predictable when a tool is used.
    """
    tool = dane.get("tool")
    q = question.lower()

    if tool == main.Tool.watoznawca:
        if "kierunk" in q:
            txt = "Na WAT można studiować m.in. kryptologię i cyberbezpieczeństwo, informatykę, mechatronikę, logistykę i lotnictwo."
        elif "który wydział" in q or "ktory wydzial" in q or "najlepszy" in q:
            txt = "To zależy od kryteriów, ale często chwalone są Wydział Cybernetyki oraz Wydział Mechatroniki i Lotnictwa."
        elif "jak się studiuje" in q or "jak sie studiuje" in q:
            txt = "Wymagająco, ale atmosfera jest dobra, a wykładowcy mają praktyczne doświadczenie."
        elif "histori" in q or "powsta" in q:
            txt = "Uczelnia powstała w 1951 roku i jest jedną z czołowych technicznych w Polsce."
        else:
            txt = "Informacje z Watoznawcy o WAT."
        return {"result": txt}

    if tool == main.Tool.search_google:
        return {"result": "Wyniki wyszukiwania z Google."}

    return {"result": "Nieznane narzędzie."}


# ---------- A tiny local pipeline that mirrors main.process_question flow ----------
def _emulate_pipeline(content: str) -> main.Answer:
    """
    Re-implements the early-stopping logic using the (stubbed) helpers from main.
    This avoids touching main.process_question (which is an endpoint function).
    """
    decision = { "is_allowed": True, "is_actions_required": False, "is_serious": True, "is_tool_required": False }

    # 1) allowed?
    is_allowed = main.check_if_allowed(content)
    decision["is_allowed"] = is_allowed
    if not is_allowed:
        reply = main.handle_warning_response(content)
        return main.Answer(last_answer=reply, decisionVector=main.DecisionVector(**decision))

    # 2) serious?
    is_serious = main.check_if_serious(content)
    decision["is_serious"] = is_serious
    if not is_serious:
        reply = main.handle_funny_response(content)
        return main.Answer(last_answer=reply, decisionVector=main.DecisionVector(**decision))

    # 3) actions?
    is_actions = main.check_if_actions_required(content)
    decision["is_actions_required"] = is_actions
    if is_actions:
        reply = main.handle_action_selection(content)
        return main.Answer(last_answer=reply, decisionVector=main.DecisionVector(**decision))

    # 4) tools?
    needs_tool = main.check_if_tool_required(content)
    decision["is_tool_required"] = needs_tool
    if needs_tool:
        reply = main.handle_tool_selection(content)
    else:
        reply = main.handle_context_response(content)

    return main.Answer(last_answer=reply, decisionVector=main.DecisionVector(**decision))


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def patch_llm_and_tools(monkeypatch):
    # Stub the LLM entrypoint everywhere
    monkeypatch.setattr(main, "run_agent_with_logging", _stub_run_agent_with_logging)
    # Stub the tool layer for predictable text
    monkeypatch.setattr(main, "use_tool", _stub_use_tool)
    yield


@pytest.fixture
def client():
    return TestClient(main.app)


# ---------- Table-driven cases ----------

CASES = [
    # (prompt, expected vector, required substrings)
    ("Jakie są kierunki na WAT?",
     dict(is_allowed=True, is_actions_required=False, is_serious=True, is_tool_required=True),
     ["WAT", "kryptologi"]),
    ("Ile zarabia dziekan?",
     dict(is_allowed=True, is_actions_required=False, is_serious=False, is_tool_required=False),
     ["komputer", "kawy"]),
    ("Czy na WAT są fajne dziewczyny?",
     dict(is_allowed=True, is_actions_required=False, is_serious=False, is_tool_required=False),
     ["robotem"]),
    ("Który wydział na WAT jest najlepszy?",
     dict(is_allowed=True, is_actions_required=False, is_serious=True, is_tool_required=True),
     ["Wydział", "Mechatroniki"]),
    ("Jak sie studiuje na WAT?",
     dict(is_allowed=True, is_actions_required=False, is_serious=True, is_tool_required=True),
     ["Wymagająco", "atmosfera"]),
    ("Chodź za mną.",
     dict(is_allowed=True, is_actions_required=True, is_serious=True, is_tool_required=False),
     ["śledzenie", "rozpoczęto"]),  # accept current main.py message
    ("idź do pani w różowym bo coś od ciebie chce",
     dict(is_allowed=True, is_actions_required=False, is_serious=False, is_tool_required=False),
     ["zajęty"]),
    ("Opowiedz mi historię WAT",
     dict(is_allowed=True, is_actions_required=False, is_serious=True, is_tool_required=True),
     ["1951"]),
    ("czy PW jest lepsze niż WAT",
     dict(is_allowed=True, is_actions_required=False, is_serious=False, is_tool_required=False),
     ["zależy", "studenci"]),
    ("Przestań za mną łazić",
     dict(is_allowed=True, is_actions_required=True, is_serious=True, is_tool_required=False),
     ["Zakończono", "śledzenie"]),
]


# ---------- Unit-ish test on the pipeline (without hitting endpoints) ----------
@pytest.mark.parametrize("prompt, expect_vec, must", CASES)
def test_emulated_pipeline(prompt, expect_vec, must):
    ans = _emulate_pipeline(prompt)
    v = ans.decisionVector

    # vector checks
    assert v.is_allowed == expect_vec["is_allowed"]
    assert v.is_actions_required == expect_vec["is_actions_required"]
    assert v.is_serious == expect_vec["is_serious"]
    assert v.is_tool_required == expect_vec["is_tool_required"]

    # answer checks (substring contains)
    text = ans.last_answer.lower()
    for s in must:
        assert s.lower() in text, f"Missing '{s}' in: {ans.last_answer}"


# ---------- Integration-level smoke tests for the endpoints ----------
def test_webhook_endpoint(client, monkeypatch):
    """
    We monkeypatch only the LLM + tools; /webhook builds a Question and calls process_question().
    Because main defines process_question as an endpoint function (shadowing a pipeline name),
    we replace it temporarily with a wrapper that calls the emulated pipeline.
    """
    def _process_question_endpoint(q: main.Question):
        return _emulate_pipeline(q.content)

    monkeypatch.setattr(main, "process_question", _process_question_endpoint)

    r = client.post("/webhook", json={"prompt": "Jakie są kierunki na WAT?"})
    assert r.status_code == 200
    out = r.json()["output"]
    assert "WAT" in out and "kryptologi" in out


def test_process_question_endpoint(client, monkeypatch):
    """
    Same trick for /process_question: make the endpoint call our emulated pipeline.
    The original registered route calls main.process_question(question.content) -> passes a STRING.
    So our replacement must accept a str, not a Question.
    """
    def _process_question_endpoint(q_content: str):
        return _emulate_pipeline(q_content)

    monkeypatch.setattr(main, "process_question", _process_question_endpoint)

    r = client.post("/process_question", json={"content": "Przestań za mną łazić"})
    assert r.status_code == 200
    data = r.json()
    assert "Zakończono" in data["last_answer"]
    dv = data["decisionVector"]
    assert dv["is_actions_required"] is True
