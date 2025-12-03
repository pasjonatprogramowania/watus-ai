# Instrukcja Weryfikacji Integracji EMMA

Poniżej znajdują się kroki niezbędne do ręcznego zweryfikowania poprawności integracji modułu pamięci EMMA z aplikacją.

## 1. Stan Kodubazy

Plik `src/main.py` został zaktualizowany i zawiera teraz logikę EMMA:
- Importy: `from src.emma import retrieve_relevant_memories, consolidate_memory`
- Funkcja `process_question`:
    - Przed wygenerowaniem odpowiedzi: pobieranie pamięci (`retrieve_relevant_memories`).
    - Po wygenerowaniu odpowiedzi: konsolidacja pamięci (`consolidate_memory`).

Port serwera został zmieniony na **8001** w `src/__init__.py` oraz w skrypcie testowym, aby uniknąć konfliktów.

## 2. Przygotowanie Środowiska

Upewnij się, że jesteś w katalogu projektu `c:\Users\pawel\Desktop\watus-ai` i masz aktywne środowisko wirtualne.

```powershell
cd c:\Users\pawel\Desktop\watus-ai
```

## 3. Uruchomienie Serwera

Przed uruchomieniem upewnij się, że żaden stary proces nie blokuje portu.

1. **Zabij stare procesy (opcjonalnie, jeśli port jest zajęty):**
   ```powershell
   # Sprawdź co działa na porcie 8001 (lub 8000)
   netstat -ano | findstr :8001
   # Jeśli coś znajdziesz, zabij proces po PID
   taskkill /PID <PID> /F
   ```

2. **Uruchom serwer:**
   Użyj poniższej komendy, aby uruchomić serwer z poprawnym `PYTHONPATH`:
   ```powershell
   $env:PYTHONPATH="c:\Users\pawel\Desktop\watus-ai"; .venv\Scripts\python.exe -m src.main
   ```
   Powinieneś zobaczyć logi startowe Uvicorn na porcie 8001.

## 4. Uruchomienie Testu Integracyjnego

W nowym oknie terminala (lub po uruchomieniu serwera w tle) wykonaj skrypt testowy:

```powershell
.venv\Scripts\python.exe test_emma_integration.py
```

### Czego oczekiwać?
Skrypt wykonuje dwa kroki:
1. **Przedstawienie się**: "My name is Pawel...". AI powinno odpowiedzieć i zapisać to w pamięci (zobaczysz logi `DEBUG: EMMA: Consolidating memory...`).
2. **Oczekiwanie**: Skrypt czeka 10 sekund na konsolidację.
3. **Pytanie o imię**: "What is my name...". AI powinno odpowiedzieć używając Twojego imienia ("Pawel" lub "Pawle"), co potwierdza, że pamięć została poprawnie odczytana.

Jeśli zobaczysz komunikat `[SUCCESS] Memory successfully retrieved!`, integracja działa poprawnie.

## 5. Rozwiązywanie Problemów

- **Błąd 403/Permission Denied**: Sprawdź klucz API w pliku `.env`.
- **Błąd połączenia**: Upewnij się, że serwer działa na porcie 8001.
- **Brak pamięci w odpowiedzi**: Sprawdź logi serwera. Powinieneś widzieć wpisy zaczynające się od `DEBUG: EMMA:`. Jeśli ich nie ma, upewnij się, że używasz najnowszej wersji `src/main.py`.
