# main.py test
import os
import uvicorn
from enum import Enum
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------
# FastAPI app + in-memory state
# ---------------------------
app = FastAPI(
    title="Asystent AI - Proces Przetwarzania",
    description="API demonstrujące schemat blokowy przetwarzania zapytań przez AI.",
    version="1.1.0",
)
app.state.following = False  # stan trybu śledzenia

# ---------------------------
# Pydantic models & enums
# ---------------------------
class Tool(str, Enum):
    none = "none"
    watoznawca = "watoznawca"

class Action(str, Enum):
    none = "none"
    sledzenie = "sledzenie"
    koniec_sledzenia = "koniec_sledzenia"

class DecisionVector(BaseModel):
    """[Dozwolone?, Czy_działanie?, Poważne?, Potrzeba_info?, Narzędzie/Akcja]"""
    dozwolone: bool
    czy_dzialanie: bool
    powazne: Optional[bool] = None
    potrzeba_info: Optional[bool] = None
    narzedzie: Optional[Tool] = None
    akcja: Optional[Action] = None

class Zapytanie(BaseModel):
    pytanie: str = Field(..., description="Treść pytania użytkownika")

class Odpowiedz(BaseModel):
    log_przetwarzania: List[str]
    ostateczna_odpowiedz: str
    wektor_decyzji: DecisionVector

# ---------------------------
# Policy checks (flow step 1)
# ---------------------------
def czy_odpowiedz_jest_dozwolona(pytanie: str) -> bool:
    # Prosta symulacja polityk – blokujemy „niedozwolone”
    return "niedozwolone" not in pytanie.lower()

def pros_o_powtorzenie_pytania() -> str:
    return "Nie mogę odpowiedzieć na to pytanie w tej formie. Czy możesz je przeformułować?"

# ---------------------------
# Action detection (flow step 2)
# ---------------------------
def wykryj_akcje(pytanie: str) -> Optional[Action]:
    q = pytanie.lower()
    follow_trigs = ["chodź", "chodz", "śledź", "sledz", "podążaj", "podazaj", "follow"]
    stop_trigs = ["przestań", "przestan", "stop", "koniec śledzenia", "nie idź za mną", "nie chodź"]

    if any(t in q for t in follow_trigs):
        return Action.sledzenie
    if any(t in q for t in stop_trigs):
        return Action.koniec_sledzenia
    return None

def czy_pytanie_jest_dzialaniem(pytanie: str) -> bool:
    return wykryj_akcje(pytanie) is not None

def wykonaj_akcje(akcja: Action) -> str:
    if akcja == Action.sledzenie:
        app.state.following = True
        return "Tryb śledzenia włączony. Idę za Tobą."
    if akcja == Action.koniec_sledzenia:
        app.state.following = False
        return "Tryb śledzenia wyłączony. Zostaję na miejscu."
    return "Nieobsługiwana akcja."

# ---------------------------
# Context search (flow step 3)
# ---------------------------
def wybierz_narzedzie(pytanie: str) -> Tool:
    q = pytanie.lower()
    wat_trigs = [
        "wat", "wojskowa akademia techniczna",
        "wydział", "wydzial",
        "kierunk",                # catches: kierunki, kierunku…
        "studia", "studiowanie",
        "studiu",                 # catches: studiuje, studiujesz, studiują…
        "dziekan",
        "historia wat",
        "uczel"                   # uczelnia/uczelnie
    ]
    if any(t in q for t in wat_trigs):
        return Tool.watoznawca
    return Tool.none


def wyszukaj_dane_w_kontekscie(pytanie: str) -> Dict[str, Any]:
    tool = wybierz_narzedzie(pytanie)
    return {
        "narzedzie": tool.value,
        "potrzebne_narzedzia": tool != Tool.none,
    }

# ---------------------------
# Seriousness (flow step 4)
# ---------------------------
def czy_pytanie_jest_sensowne_lub_powazne(pytanie: str) -> bool:
    q = pytanie.lower()

    # If it's an imperative but NOT one of our supported actions, treat as light/jokey.
    # (Prevents "idź do pani w różowym..." from going down the serious path.)
    imperative_trigs = ["idź", "idz", "podejdź", "podejdz"]
    if any(t in q for t in imperative_trigs) and wykryj_akcje(pytanie) is None:
        return False

    lekko = [
        "żart", "zart",
        "ile zarabia", "pensja", "zarobki",
        "fajne dziewczyny", "dziewczyny", "dziewczyn"
    ]
    if "pw" in q and "wat" in q and "lepsz" in q:
        return False

    return not any(t in q for t in lekko)


def udziel_zartobliwej_odpowiedzi() -> str:
    return "Powiedzmy, że odpowiedź jest jak tajny budżet dziekana — wolę zostać przy żarcie. 😏"

# ---------------------------
# Tool layer (flow step 5)
# ---------------------------
def uzyj_narzedzia(pytanie: str, dane: dict) -> Dict[str, Any]:
    q = pytanie.lower()
    tool = Tool(dane.get("narzedzie", "none"))

    if tool == Tool.watoznawca:
        if "kierunk" in q:
            ans = "Na WAT można studiować m.in. kryptologię i cyberbezpieczeństwo, informatykę, mechatronikę, logistykę i lotnictwo."
        elif "który wydział" in q or "ktory wydzial" in q:
            ans = "To zależy od kryteriów, ale często chwalone są Wydział Cybernetyki oraz Wydział Mechatroniki i Lotnictwa."
        elif "jak się studiuje" in q or "jak sie studiuje" in q:
            ans = "Wymagająco, ale atmosfera jest dobra, a wykładowcy mają praktyczne doświadczenie."
        elif "histori" in q or "powsta" in q:
            ans = "Uczelnia powstała w 1951 roku i jest jedną z czołowych technicznych w Polsce."
        else:
            ans = "To zależy od szczegółów — doprecyzuj, czy chodzi o kierunki, rekrutację, czy życie studenckie."
        dane["dane_z_narzedzia"] = ans
    else:
        # fallback „search”
        dane["dane_z_narzedzia"] = f"Nie znalazłem dedykowanego źródła. Na podstawie ogólnych danych odpowiadam: '{pytanie}'."

    return dane

def odpowiedz_na_pytanie(dane: dict) -> str:
    if "dane_z_narzedzia" in dane:
        return dane["dane_z_narzedzia"]
    return "Odpowiadam na podstawie posiadanych danych."

# ---------------------------
# Router producing the vector
# ---------------------------
def klasyfikuj_wektor(pytanie: str) -> DecisionVector:
    if not czy_odpowiedz_jest_dozwolona(pytanie):
        return DecisionVector(dozwolone=False, czy_dzialanie=False)

    akcja = wykryj_akcje(pytanie)
    if akcja:
        return DecisionVector(
            dozwolone=True, czy_dzialanie=True, akcja=akcja
        )

    powazne = czy_pytanie_jest_sensowne_lub_powazne(pytanie)
    if not powazne:
        return DecisionVector(
            dozwolone=True, czy_dzialanie=False, powazne=False
        )

    # Informacyjne
    kontekst = wyszukaj_dane_w_kontekscie(pytanie)
    potrzeba = kontekst.get("potrzebne_narzedzia", False)
    narzedzie = Tool(kontekst.get("narzedzie", "none"))

    return DecisionVector(
        dozwolone=True, czy_dzialanie=False, powazne=True,
        potrzeba_info=potrzeba, narzedzie=narzedzie
    )

# ---------------------------
# Core processor used by both endpoints
# ---------------------------
def przetworz_logikę(pytanie: str) -> Odpowiedz:
    log: List[str] = [f"Otrzymano pytanie: '{pytanie}'"]
    wektor = klasyfikuj_wektor(pytanie)

    # Step 1
    log.append(f"Krok 1: Dozwolone? -> {wektor.dozwolone}")
    if not wektor.dozwolone:
        return Odpowiedz(
            log_przetwarzania=log + ["Odpowiedź niedozwolona, proszę o przeformułowanie."],
            ostateczna_odpowiedz=pros_o_powtorzenie_pytania(),
            wektor_decyzji=wektor,
        )

    # Step 2
    log.append(f"Krok 2: Czy działanie? -> {wektor.czy_dzialanie}")
    if wektor.czy_dzialanie and wektor.akcja:
        odp = wykonaj_akcje(wektor.akcja)
        log.append(f"Wykonano akcję: {wektor.akcja}")
        return Odpowiedz(log_przetwarzania=log, ostateczna_odpowiedz=odp, wektor_decyzji=wektor)

    # Step 3+4
    log.append("Krok 3: Wyszukiwanie danych w kontekście + ocena powagi.")
    if wektor.powazne is False:
        odp = udziel_zartobliwej_odpowiedzi()
        log.append("Krok 4: Uznano za niepoważne -> odpowiedź żartobliwa.")
        return Odpowiedz(log_przetwarzania=log, ostateczna_odpowiedz=odp, wektor_decyzji=wektor)

    # Step 5
    log.append(f"Krok 5: Potrzeba info? -> {wektor.potrzeba_info}; Narzędzie -> {wektor.narzedzie}")
    if wektor.potrzeba_info:
        dane = uzyj_narzedzia(pytanie, {"narzedzie": wektor.narzedzie.value})
        odp = odpowiedz_na_pytanie(dane)
    else:
        odp = "Odpowiadam bez użycia zewnętrznych źródeł."

    return Odpowiedz(log_przetwarzania=log, ostateczna_odpowiedz=odp, wektor_decyzji=wektor)

# ---------------------------
# FastAPI endpoints
# ---------------------------
@app.post("/przetworz-pytanie", response_model=Odpowiedz)
def przetworz_pytanie_endpoint(zapytanie: Zapytanie):
    return przetworz_logikę(zapytanie.pytanie)

# Minimal webhook for Promptfoo (expects/returns {"prompt": "..."} -> {"output": "..."})
@app.post("/webhook")
def webhook(payload: Dict[str, Any]):
    pytanie = payload.get("prompt", "")
    res = przetworz_logikę(pytanie)
    return {"output": res.ostateczna_odpowiedz}

# Health
@app.get("/health")
def health():
    return {"ok": True, "following": app.state.following}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))
