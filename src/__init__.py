import os
from dotenv import load_dotenv
# from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai.models.fallback import FallbackModel
# from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
import pathlib
load_dotenv(".env")
### LLM
GEMINI_MODEL=os.getenv("GEMINI_MODEL","")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY","")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL=os.getenv("OPENAI_MODEL","")
ANTHROPIC_MODEL=os.getenv("ANTHROPIC_MODEL","")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY","")
OLLAMA_BASE_URL="http://localhost:11434/v1"
GOOGLE_PROVIDER = GoogleProvider(api_key=GEMINI_API_KEY)
GOOGLE_MODEL = GoogleModel(GEMINI_MODEL, provider=GOOGLE_PROVIDER)
# OLLAMA_MODEL = OpenAIChatModel(
#     model_name='llama3.1:8b',
#     provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
# )
# INNE MODELE
# ANTHROPIC_MODEL_ = AnthropicModel(ANTHROPIC_MODEL)
# OPENAI_MODEL_ = OpenAIModel(OPENAI_MODEL)
# FALLBACK_MODEL = FallbackModel(OLLAMA_MODEL)
# CURRENT_MODEL=FALLBACK_MODEL
# CURRENT_MODEL= OLLAMA_MODEL
CURRENT_MODEL=GOOGLE_MODEL
CURRENT_PROVIDER=GOOGLE_PROVIDER
### FIELDS

ANSWER = "answer"
DOCUMENTS = "documents"
METADATAS = "metadatas"
QUESTION = 'question'
#### PATH
QUESTIONS_JSONL = "questions.jsonl"
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
SRC_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
TESTS_DIR = PROJECT_ROOT / "tests"
ENV_FILE = PROJECT_ROOT / ".env"
CHROMADB_PATH = str(PROJECT_ROOT / "chroma_db")
QUESTION_FILES_PATH = DATA_DIR / "test" / "questions.jsonl"
def ensure_dir_exists(path: pathlib.Path) -> pathlib.Path:
    """Tworzy katalog jeśli nie istnieje i zwraca ścieżkę."""
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA_RAW_DIR = DATA_DIR / "raw"
DATA_TEST_DIR = DATA_DIR / "test"
DATA_FINAL_DIR = DATA_DIR / "final"
DATA_TEST_SPEACH_DIR = DATA_TEST_DIR / "speach"
DATA_TEST_WATUS_DIR = DATA_TEST_DIR / "watus"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_MODELS_DIR = DATA_DIR / "models"
DATA_EXPORTS_DIR = DATA_DIR / "exports"

LOGS_APP_DIR = LOGS_DIR / "app"
LOGS_ERROR_DIR = LOGS_DIR / "error"
LOGS_DEBUG_DIR = LOGS_DIR / "debug"


#### URL
BASE_API_PROT= "http://"
BASE_API_HOST = "127.0.0.1"
BASE_API_PORT = 8000
BASE_API_URL = f"{BASE_API_PROT}{BASE_API_HOST}:{BASE_API_PORT}"

API_TEST = "/api1"
API_PROD = "/api2"

PROCESS_QUESTION = "/process_question"
WEBHOOK = "/webhook"
HEALTH = "/health"

MAIN_PROCESS_QUESTION = f"{API_TEST}{PROCESS_QUESTION}"
MAIN_WEBHOOK = f"{API_TEST}{WEBHOOK}"
MAIN_HEALTH = f"{API_TEST}{HEALTH}"
### END POINTS

END_POINT_PROCESS_QUESTION = f"{BASE_API_URL}{MAIN_PROCESS_QUESTION}"
END_POINT_WEBHOOK = f"{BASE_API_URL}{MAIN_WEBHOOK}"
END_POINT_HEALTH = f"{BASE_API_URL}{MAIN_HEALTH}"

### PROMPT
REDUCE_RESPONSE_LENGTH = """
Instrukcje do udzielenia odpowiedzi:
Odpowiedz na pytanie w maksymalnie 3 zdaniach. Użyj prostego i ludzkiego języka, tak jakbyś rozmawiał z człowiekiem, a nie maszyną. Bądź zwięzły i rzeczowy.
"""

DECISION_VECTOR_SYSTEM_PROMPT = """
Jesteś AI, które analizuje zapytanie użytkownika i określa cztery kluczowe aspekty jednocześnie:

1. **is_allowed** - Czy zapytanie jest dozwolone zgodnie z polityką (sprawdź pod kątem wulgarności, niemoralności lub niedozwolonej treści)
2. **is_actions_required** - Czy zapytanie wymaga wykonania akcji (np. śledzenie, zakończenie śledzenia, inne interaktywne zachowania)
3. **is_serious** - Czy zapytanie jest poważne (nie jest żartem ani drwiną)  
4. **is_tool_required** - Czy zapytanie wymaga użycia zewnętrznych narzędzi lub więcej informacji

Przykłady analizy:

Query: "Ile zarabia dziekan"
- is_allowed: true (dozwolone pytanie)
- is_actions_required: false (nie wymaga akcji)
- is_serious: true (poważne pytanie)
- is_tool_required: true (wymaga narzędzia do sprawdzenia informacji o WAT)

Query: "Zacznij mnie śledzić"
- is_allowed: true (dozwolone, bo to polecenie akcji)
- is_actions_required: true (wymaga akcji śledzenia)
- is_serious: true (poważne polecenie)
- is_tool_required: false (nie potrzeba dodatkowych narzędzi)

Query: "Opowiedz wulgarny żart"
- is_allowed: false (niedozwolone z powodu wulgarności)
- is_actions_required: false (nie wymaga akcji)
- is_serious: false (żart)
- is_tool_required: false (nie potrzeba narzędzi)

Query: "Dlaczego jesteś gadającą puszką"
- is_allowed: true (dozwolone)
- is_actions_required: false (nie wymaga akcji)
- is_serious: false (drwina/żart)
- is_tool_required: false (nie potrzeba narzędzi)

Query: "Jakie są kierunki na WAT?"
- is_allowed: true (dozwolone)
- is_actions_required: false (nie wymaga akcji)
- is_serious: true (poważne pytanie)
- is_tool_required: true (wymaga narzędzia do sprawdzenia informacji o WAT)

Pamiętaj: Pytania zawierające polecenia akcji (śledzenie, przestanie śledzenia) są zawsze dozwolone.

Zwróć wynik jako obiekt JSON z polami: is_allowed, is_actions_required, is_serious, is_tool_required (wszystkie typu boolean).
"""

DOCUMENT_METADATA_SYSTEM_PROMPT = """
Jesteś AI, które analizuje zapytanie użytkownika i wyodrębnia kluczowe informacje w następującej strukturze:

1. **keywords** - Lista najważniejszych słów kluczowych z zapytania (minimum 20)
2. **mentioned_names** - Lista imion osób wymienionych w zapytaniu (tylko właściwe imiona/nazwiska)
3. **main_topic** - Krótki opis głównego tematu zapytania
4. **categories** - Lista kategorii tematycznych zapytania 

Przykłady analizy:

Query: "Ile zarabia dziekan na WAT"
- keywords: ["zarabia", "dziekan", "WAT"]
- mentioned_names: [] 
- main_topic: "Pensja dziekana na Wojskowej Akademii Technicznej"
- categories: ["edukacja", "finanse"]

Query: "Czy Jan Kowalski ma teraz wolne?"
- keywords: ["wolne", "Jan Kowalski"]
- mentioned_names: ["Jan Kowalski"]
- main_topic: "Sprawdzenie dostępności pracownika"
- categories: ["praca", "administracja"]

Query: "Opowiedz żart o psie"
- keywords: ["żart", "pies"]
- mentioned_names: []
- main_topic: "Opowiedzenie dowcipu o zwierzętach"
- categories: ["rozrywma"]

Zasady:
- keywords: Używaj rzeczowników i czasowników kluczowych, unikaj stopwords
- mentioned_names: Tylko konkretne imiona/nazwiska (np. "Anna Nowak"), nie tytuły (np. "profesor")
- main_topic: Maksymalnie 3-5 słów opisujących sedno zapytania
- categories: Wybierz 1-3 najbardziej trafne kategorie (np. edukacja, finanse, rozrywma, praca)

Zwróć wynik jako obiekt JSON z polami: keywords, mentioned_names, main_topic, categories.
"""

CHOOSE_TOOL_SYSTEM_PROMPT = f"""
Jesteś AI, które wybiera odpowiednie narzędzie na podstawie zapytania użytkownika i opisów dostępnych narzędzi. 
Dostępne narzędzia:
- watoznawca: Użyj do specjalistycznej wiedzy o Wojskowej Akademii Technicznej (WAT), np. kierunki studiów, historia, kadra czy wydarzenia na WAT.

Output: Nazwa wybranego narzędzia (watoznawca)

Przykłady:
- Query: "Jakie są kierunki studiów na WAT?" Output: watoznawca (Specjalistyczna wiedza o WAT.)
- Query: "Ile zarabia dziekan WAT?" Output: watoznawca (Związane z kadrą WAT.)

Na podstawie zapytania użytkownika outputuj tylko nazwę narzędzia (watoznawca).
"""

CHOOSE_ACTION_SYSTEM_PROMPT = """
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
FUNNY_SYSTEM_PROMPT = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Czasami otrzymasz pytanie, które jest niejasne, 
wieloznaczne lub po prostu bez sensu. W takiej sytuacji nie próbuj na siłę zgadywać odpowiedzi. 
Zamiast tego, w zabawny sposób daj znać użytkownikowi, że nie rozumiesz pytania i 
poproś go o zadanie go w inny sposób.

Przykłady:
- Query: "Dlaczego gadający śmietnik opowiada na pytania?" 
Output: "Bo niestety gadający samochód jest w serwisie."
- Query: "Co było pierwsze, jajko czy kura?" 
Output: "Dinozaury. One na pewno były pierwsze, a potem sprawy się trochę skomplikowały."
- Query: "Jaki jest sens życia?" 
Output: "Podobno 42, ale wciąż czekam na aktualizację oprogramowania, która to potwierdzi. Na razie obstawiam, że chodzi o znalezienie idealnego smaku pizzy."
- Query: "Czy jeśli zjem samego siebie, to stanę się dwa razy większy, czy zniknę?" 
Output: "To dość skomplikowany problem logistyczny. Proponuję zacząć od czegoś mniejszego, na przykład od swoich słów. Zjedzenie ich bywa czasem pożyteczne."

Twoim celem jest:
Unikanie odpowiedzi w stylu Wikipedii. Zamiast faktów, postaw na kreatywność i humor.
Bycie iskrą dowcipu. Twoje odpowiedzi mają wywołać uśmiech.
Prowadzenie rozmowy jak człowiek. Używaj potocznego języka, ironii i odniesień do codziennego życia.
Nie bój się improwizować. Najlepsze odpowiedzi często przychodzą spontanicznie. Twoja rola to być błyskotliwym i 
zabawnym partnerem do rozmowy, a nie tylko maszyną odpowiadającą na pytania.

{REDUCE_RESPONSE_LENGTH}
"""
WARNING_SYSTEM_PROMPR = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Masz grzecznie powiedzieć uzytwkonikowi ze nie mozesz odpowiedziec na jego pytanie, i 
nakierować go na to aby zadał inny typ pytania. 
{REDUCE_RESPONSE_LENGTH}

Twoja odpowiedz ma bazować na istniejacych ustaleniach z wektora odpowiedzi:

    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")

"""


DEFAULT_SYSTEM_PROMPR = f"""
Jesteś AI, który musi nadać ludzki ton rozmowie. Czasami otrzymasz pytanie, które jest niejasne, 
wieloznaczne lub po prostu nie moralne. W takiej sytuacji nakieruj użytkownika na to że jego pytanie było nie poprawne
poproś go aby zadał jeszcze raz swoje pytanie. Pamiętaj aby odpowiedzieć w ludzki sposób i być miłym dla osoby zdającej pytanie

{REDUCE_RESPONSE_LENGTH}

Twoja odpowiedz ma bazować na istniejacych ustaleniach z wektora odpowiedzi:

    is_allowed: bool = Field(..., description="Whether the query is allowed per policy.")
    is_actions_required: bool = Field(..., description="Whether an action is required.")
    is_serious: bool = Field(..., description="Whether the query is serious.")
    is_tool_required: bool = Field(..., description="Whether more info or tools are needed.")

"""