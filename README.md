# Watus AI - Inteligentny Asystent Konwersacyjny

## Opis projektu

Watus AI to zaawansowany system asystenta konwersacyjnego oparty na sztucznej inteligencji, który analizuje zapytania użytkowników i udziela odpowiedzi w oparciu o zintegrowany wektor decyzyjny. System obsługuje przetwarzanie rozmów, generowanie metadanych oraz przechowywanie danych w bazie wektorowej.

## Funkcjonalności

- **Analiza zapytań** - Jednoznaczne określenie kategorii pytania (dozwolone, wymagające akcji, poważne, wymagające narzędzi)
- **Przetwarzanie rozmów** - Obsługa formatów JSON i JSONL z danymi konwersacyjnymi
- **Generowanie metadanych** - Automatyczne tworzenie słów kluczowych, kategorii i streszczeń
- **Baza wektorowa** - Przechowywanie i wyszukiwanie podobnych rozmów (ChromaDB)
- **API REST** - Endpoints do komunikacji z systemem
- **Obsługa języka polskiego** - Pełne wsparcie dla znaków UTF-8

## Wymagania systemowe

- Python 3.8+
- ChromaDB
- FastAPI
- Pydantic AI
- Dostęp do API Google Gemini

## Instalacja

### 1. Klonowanie repozytorium
```bash
git clone <repository-url>
cd watus-ai
```

### 2. Tworzenie środowiska wirtualnego
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 4. Konfiguracja zmiennych środowiskowych
Skopiuj `.env.example` do `.env` i wypełnij wymagane wartości:

```bash
cp .env.example .env
```

Przykładowa zawartość `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

## Uruchamianie aplikacji

### Serwer API
```bash
# Uruchomienie serwera FastAPI
python src/main.py

# Lub przy użyciu uvicorn
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Serwer będzie dostępny pod adresem: `http://127.0.0.1:8000`

### Dokumentacja API
Po uruchomieniu serwera, dokumentacja Swagger będzie dostępna pod:
- `http://127.0.0.1:8000/docs` (Swagger UI)
- `http://127.0.0.1:8000/redoc` (ReDoc)

## Endpointy API

### 1. Przetwarzanie pytań
**POST** `/api1/process_question`

Główny endpoint do analizy zapytań użytkowników.

**Request Body:**
```json
{
  "content": "Jakie są kierunki studiów na WAT?"
}
```

**Response:**
```json
{
  "answer": "Na Wojskowej Akademii Technicznej dostępne są kierunki...",
  "decisionVector": {
    "is_allowed": true,
    "is_actions_required": false,
    "is_serious": true,
    "is_tool_required": true
  }
}
```

### 2. Webhook
**POST** `/api1/webhook`

Endpoint dla zewnętrznych integracji.

**Request Body:**
```json
{
  "prompt": "Twoje pytanie tutaj"
}
```

**Response:**
```json
{
  "output": "Odpowiedź asystenta"
}
```

### 3. Health Check
**GET** `/api1/health`

Sprawdzenie statusu aplikacji.

**Response:**
```json
{
  "ok": true
}
```

## Przykłady użycia

### Curl
```bash
# Podstawowe zapytanie
curl -X POST "http://127.0.0.1:8000/api1/process_question" \
  -H "Content-Type: application/json" \
  -d '{"content": "Ile zarabia dziekan?"}'

# Health check
curl -X GET "http://127.0.0.1:8000/api1/health"
```

### Python
```python
import requests

# Zapytanie do API
response = requests.post(
    "http://127.0.0.1:8000/api1/process_question",
    json={"content": "Jakie są kierunki na WAT?"}
)

result = response.json()
print(f"Odpowiedź: {result['answer']}")
print(f"Wektor decyzji: {result['decisionVector']}")
```

### JavaScript
```javascript
fetch('http://127.0.0.1:8000/api1/process_question', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    content: 'Czy polecasz studiowanie informatyki?'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Przetwarzanie rozmów

### Uruchamianie procesora rozmów
```bash
# Przetwarzanie plików z folderu data/
python -m src.vectordb
```

### Obsługiwane formaty plików

#### Format JSON
```json
{
  "type": "leader_question",
  "session_id": "live_71753",
  "group_id": "leader_1757776664221",
  "speaker_id": "leader",
  "turn_ids": [1757776664221],
  "text_full": "nazywam się Michalina, będę dzisiaj z tobą rozmawiała i studiuję informatykę.",
  "category": "polecenie",
  "reply_hint": true,
  "ts_start": 1757776645.665547,
  "ts_end": 1757776664.221288,
  "emit_reason": "tts_start",
  "ts": 1757776664.222187
}
```

#### Format JSONL
```jsonl
{"type": "leader_question", "text_full": "Pierwsza wypowiedź", "speaker_id": "user1"}
{"type": "ai_response", "text_full": "Odpowiedź systemu", "speaker_id": "ai"}
{"type": "leader_question", "text_full": "Druga wypowiedź", "speaker_id": "user1"}
```

### Wyniki przetwarzania
System generuje:
- **Metadane** - słowa kluczowe, kategorie, główny temat
- **Indeks wektorowy** - dla wyszukiwania podobnych rozmów
- **Plik wynikowy** - `processing_results.json`

## Wektor decyzyjny

System analizuje każde zapytanie pod kątem czterech kategorii:

1. **is_allowed** - Czy pytanie jest dozwolone (brak wulgarności, niemoralności)
2. **is_actions_required** - Czy wymaga wykonania akcji (śledzenie, zakończenie)
3. **is_serious** - Czy pytanie jest poważne (nie żart/drwina)
4. **is_tool_required** - Czy wymaga dodatkowych narzędzi/informacji

### Przykłady kategoryzacji:

| Pytanie | is_allowed | is_actions_required | is_serious | is_tool_required |
|---------|------------|-------------------|------------|------------------|
| "Ile zarabia dziekan" | ✅ true | ❌ false | ✅ true | ✅ true |
| "Zacznij mnie śledzić" | ✅ true | ✅ true | ✅ true | ❌ false |
| "Opowiedz wulgarny żart" | ❌ false | ❌ false | ❌ false | ❌ false |
| "Jakie są kierunki na WAT?" | ✅ true | ❌ false | ✅ true | ✅ true |

## Struktura projektu

```
watus-ai/
├── src/
│   ├── __init__.py          # Konfiguracja i prompty systemowe
│   ├── main.py              # Serwer FastAPI i logika główna
│   └── vectordb.py          # Przetwarzanie rozmów i baza wektorowa
├── data/                    # Pliki rozmów do przetworzenia
├── tests/                   # Testy jednostkowe
├── promptfoo_tests/         # Konfiguracja testów Promptfoo
├── .env.example             # Przykład zmiennych środowiskowych
├── requirements.txt         # Zależności Python
└── README.md               # Dokumentacja
```

## Testowanie

### Testy jednostkowe
```bash
# Uruchomienie testów
python -m pytest tests/

# Testy z detalami
python -m pytest tests/ -v
```

### Testy Promptfoo
```bash
# Instalacja Promptfoo
npm install -g promptfoo

# Uruchomienie testów
promptfoo eval
```

## Troubleshooting

### Częste problemy

1. **404 Not Found** - Sprawdź czy serwer jest uruchomiony i używasz poprawnego endpoint'u
2. **Błędy kodowania** - Upewnij się, że pliki używają kodowania UTF-8
3. **Błędy API** - Sprawdź czy zmienne środowiskowe są poprawnie skonfigurowane
4. **ChromaDB errors** - Usuń folder `chroma_db` i uruchom ponownie

### Logi i debugging
```bash
# Włączenie szczegółowych logów
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.main --log-level debug
```

## Rozwój

### Dodawanie nowych promptów
Edytuj plik `src/__init__.py` i dodaj nowy prompt systemowy:

```python
NEW_SYSTEM_PROMPT = """
Twój nowy prompt tutaj...
"""
```

### Rozszerzanie API
Dodaj nowe endpointy w `src/main.py`:

```python
@app.post("/api1/new_endpoint")
def new_endpoint(data: YourModel):
    # Twoja logika tutaj
    return {"result": "success"}
```

## Licencja

[Dodaj informacje o licencji]

## Kontakt

[Dodaj informacje kontaktowe]