# Dokumentacja Techniczna Folderu `src`

Niniejszy dokument stanowi szczegółowy opis techniczny kodu źródłowego znajdującego się w folderze `src`. Jego celem jest wyjaśnienie architektury, przepływu danych (workflow) oraz przeznaczenia poszczególnych plików i funkcji.

## 1. Przegląd Architektury i Workflow

Aplikacja jest asystentem AI opartym na architekturze RAG (Retrieval-Augmented Generation) z dodatkowym modułem pamięci epizodycznej EMMA (Egocentric Memory for Multimodal Agents).

### Główny Przepływ Danych (Workflow)

Proces obsługi zapytania użytkownika przebiega następująco:

1.  **Otrzymanie Zapytania (`main.py`)**: Endpoint `/api1/process_question` odbiera zapytanie POST.
2.  **Analiza Wektora Decyzyjnego (`main.py` -> `get_decision_vector`)**:
    *   Model AI analizuje pytanie pod kątem 4 cech: Czy dozwolone? Czy wymaga akcji? Czy poważne? Czy wymaga narzędzi?
    *   To pozwala na "Early Stopping" - np. szybkie odrzucenie pytań niedozwolonych lub odpowiedź żartem na pytania niepoważne, bez angażowania drogich zasobów.
3.  **Wyszukiwanie Wiedzy (`main.py` -> `vector_search` -> `vectordb.py`)**:
    *   Jeśli `is_tool_required` wskazuje na potrzebę wiedzy, system przeszukuje bazę wektorową ChromaDB (`knowledge_base`) w poszukiwaniu dokumentów o WAT.
4.  **Odzyskiwanie Pamięci EMMA (`main.py` -> `emma.retrieve_relevant_memories`)**:
    *   System przeszukuje pamięć epizodyczną (`conversation_memory`) w poszukiwaniu faktów o użytkowniku i poprzednich rozmowach.
    *   Znalezione wspomnienia są dołączane do kontekstu (promptu).
5.  **Generowanie Odpowiedzi (`main.py` -> `handle_default_response`)**:
    *   Główny agent generuje odpowiedź, mając do dyspozycji: pytanie użytkownika, wiedzę z bazy wektorowej oraz kontekst pamięci EMMA.
6.  **Konsolidacja Pamięci (`main.py` -> `emma.consolidate_memory`)**:
    *   Po wygenerowaniu odpowiedzi, system (w tle) analizuje rozmowę.
    *   Ekstrahuje nowe fakty i zapisuje je w bazie wektorowej, linkując je z powiązanymi wspomnieniami.

---

## 2. Szczegółowy Opis Plików

### 2.1. `src/__init__.py` - Centrum Konfiguracji

Ten plik pełni rolę globalnego pliku konfiguracyjnego. Definiuje stałe, ścieżki oraz inicjalizuje modele LLM. Dzięki temu konfiguracja jest scentralizowana i łatwa do zmiany.

**Kluczowe elementy:**
*   **Konfiguracja LLM**: Inicjalizacja modeli (np. `GoogleModel`, `OpenAIModel`) i providerów. Zmienna `CURRENT_MODEL` pozwala na łatwe przełączanie silnika AI dla całej aplikacji.
*   **Ścieżki (`pathlib`)**: Definicje ścieżek do katalogów (`DATA_DIR`, `LOGS_DIR`, `CHROMADB_PATH`), co zapewnia niezależność od systemu operacyjnego.
*   **System Prompty**: Przechowuje stałe tekstowe z instrukcjami dla agentów (np. `DECISION_VECTOR_SYSTEM_PROMPT`, `DOCUMENT_METADATA_SYSTEM_PROMPT`). Oddzielenie promptów od kodu logiki zwiększa czytelność.
*   **Konfiguracja API**: Adresy hosta i portu (`BASE_API_PORT = 8001`).

### 2.2. `src/main.py` - Główny Kontroler (API)

To serce aplikacji. Definiuje API w oparciu o framework **FastAPI** i orkiestruje pracę agentów.

**Kluczowe Funkcje:**

*   `process_question(content: str) -> Answer`:
    *   **Rola**: Główna funkcja sterująca logiką biznesową.
    *   **Działanie**: Wywołuje analizę decyzyjną, decyduje o użyciu narzędzi, pobiera pamięć EMMA, generuje odpowiedź i uruchamia konsolidację pamięci.
*   `get_decision_vector(content: str)`:
    *   **Rola**: Klasyfikacja zapytania.
    *   **Działanie**: Używa agenta `decision_vector_agent` do zwrócenia obiektu `DecisionVector` (flagi: allowed, serious, action, tool).
*   `vector_search(query: str)`:
    *   **Rola**: Wrapper na wyszukiwanie w bazie wiedzy.
    *   **Działanie**: Waliduje zapytanie i wywołuje funkcje z `vectordb.py`.
*   `handle_default_response`, `handle_warning_response`, `handle_funny_response`:
    *   **Rola**: Specjalistyczne handlery odpowiedzi.
    *   **Działanie**: Każdy używa innego System Promptu, aby nadać odpowiedzi odpowiedni ton (np. zabawny lub ostrzegawczy).
*   `run_agent_with_logging(...)`:
    *   **Rola**: Wrapper do uruchamiania agentów `pydantic-ai`.
    *   **Działanie**: Mierzy czas wykonania i loguje zapytanie oraz odpowiedź, co jest kluczowe dla debugowania i monitoringu.

### 2.3. `src/vectordb.py` - Warstwa Danych (ChromaDB)

Odpowiada za całą interakcję z bazą wektorową ChromaDB. Obsługuje zarówno statyczną bazę wiedzy (o uczelni), jak i dynamiczną pamięć rozmów.

**Kluczowe Funkcje:**

*   `initialize_vector_db()`:
    *   Tworzy lub łączy się z trwałą instancją ChromaDB na dysku.
*   `batch_process(folder)`:
    *   **Rola**: Przetwarzanie plików z danymi (ETL).
    *   **Działanie**: Czyta pliki `.jsonl`, parsuje je i przygotowuje do wstawienia do bazy.
*   `generate_metadata(conversation_content)`:
    *   **Rola**: Wzbogacanie danych.
    *   **Działanie**: Używa LLM do wygenerowania słów kluczowych, tematów i kategorii dla każdego fragmentu rozmowy. To pozwala na lepsze wyszukiwanie niż tylko po treści.
*   `add_to_vector_db(...)`:
    *   Zapisuje przetworzone dokumenty wraz z metadanymi do kolekcji `knowledge_base`.
*   **Funkcje Pamięci EMMA**:
    *   `get_memory_collection()`: Zwraca kolekcję `conversation_memory`.
    *   `add_memory(...)`: Dodaje pojedynczy fakt do pamięci.
    *   `search_memory(...)`: Wyszukuje fakty, umożliwiając filtrowanie po `user_id`.

### 2.4. `src/emma.py` - Moduł Kognitywny (Pamięć)

Implementuje logikę architektury EMMA. Odpowiada za to, by AI "pamiętało" użytkownika i kontekst.

**Kluczowe Funkcje:**

*   `retrieve_relevant_memories(user_id, query)`:
    *   **Rola**: Przypominanie.
    *   **Działanie**:
        1.  Wyszukuje w ChromaDB fakty podobne do obecnego zapytania.
        2.  **Ekspansja**: Sprawdza metadane `links` znalezionych faktów, aby pobrać też wspomnienia powiązane (nawet jeśli nie są podobne tekstowo). To kluczowa cecha EMMA - asocjacyjność pamięci.
*   `consolidate_memory(user_id, user_input, ai_response)`:
    *   **Rola**: Zapamiętywanie (Uczenie się).
    *   **Działanie**:
        1.  **Ekstrakcja**: Agent `MemoryExtractionAgent` analizuje ostatnią wymianę zdań i wyciąga "suche fakty" (np. "Użytkownik ma na imię Paweł").
        2.  **Linkowanie**: Agent `MemoryLinkingAgent` sprawdza, czy nowy fakt wiąże się z czymś, co już jest w bazie.
        3.  **Zapis**: Zapisuje nowy fakt w ChromaDB, w tym informacje o powiązaniach (linkach).

## Podsumowanie

Kod jest zorganizowany modularnie:
- **`__init__.py`**: Konfiguracja.
- **`main.py`**: Logika sterująca i API.
- **`vectordb.py`**: Niskopoziomowe operacje na bazie danych.
- **`emma.py`**: Wysokopoziomowa logika pamięci i wnioskowania.

Taki podział ułatwia rozwój (np. zmiana bazy danych wymaga zmian tylko w `vectordb.py`, a zmiana modelu AI tylko w `__init__.py`).
