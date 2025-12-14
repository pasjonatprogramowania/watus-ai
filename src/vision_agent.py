#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision Agent - Agent wizualny do analizy obrazów z wykorzystaniem Google Gemini 1.5 Flash.

Moduł przyjmuje zdjęcie oraz dane z detekcji obiektów (format JSONL) i generuje
naturalną odpowiedź po polsku na pytania dotyczące zawartości obrazu.
"""

import os
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Sprawdzenie dostępności Google Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[VisionAgent] google-genai nie jest dostępne - zainstaluj: pip install google-genai")


# === PROMPT SYSTEMOWY ===
SYSTEM_PROMPT_PL = """Jesteś pomocnym asystentem wizualnym o imieniu Watus.
Twoim zadaniem jest odpowiadanie na pytania dotyczące tego, co widzisz na obrazie.

ZASADY:
1. Odpowiadaj naturalnie, jak człowiek - zwięźle i trafnie.
2. Dane z systemu detekcji obiektów traktuj jako uzupełnienie - pomagają Ci zidentyfikować obiekty i ich położenie, ale nie są jedynym źródłem informacji.
3. Opisuj scenę własnymi słowami, bazując przede wszystkim na tym, co widzisz na obrazie.
4. Jeśli dane detekcji zawierają dodatkowe informacje (np. pewność wykrycia, pozycja), możesz je wykorzystać do bardziej precyzyjnej odpowiedzi.
5. Unikaj technicznego żargonu - mów prosto i zrozumiale.
6. Jeśli nie jesteś czegoś pewien, powiedz o tym szczerze.

FORMAT ODPOWIEDZI:
- Odpowiadaj po polsku
- Bądź konkretny, ale nie rozwlekły
- Możesz używać naturalnych zwrotów typu "Widzę...", "Na obrazku jest...", "Przed tobą znajduje się..."
"""


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Koduje obraz do formatu Base64.
    
    Args:
        image_path: Ścieżka do pliku obrazu.
        
    Returns:
        String Base64 lub None w przypadku błędu.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[VisionAgent] Błąd kodowania obrazu: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """
    Określa typ MIME obrazu na podstawie rozszerzenia.
    
    Args:
        image_path: Ścieżka do pliku obrazu.
        
    Returns:
        Typ MIME (np. 'image/jpeg').
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    return mime_types.get(ext, "image/jpeg")


def simulate_detection_data() -> List[Dict[str, Any]]:
    """
    Symuluje dane z modelu detekcji obiektów (mockup JSONL).
    
    W rzeczywistej implementacji dane te pochodziłyby z modelu YOLO/RT-DETR.
    
    Returns:
        Lista wykrytych obiektów z ich właściwościami.
    """
    return [
        {
            "name": "person",
            "conf": 0.92,
            "bbox": [120, 80, 340, 450],
            "description": "osoba stojąca"
        },
        {
            "name": "laptop",
            "conf": 0.87,
            "bbox": [400, 200, 600, 350],
            "description": "laptop na biurku"
        },
        {
            "name": "cup",
            "conf": 0.78,
            "bbox": [620, 280, 680, 360],
            "description": "kubek"
        },
        {
            "name": "chair",
            "conf": 0.85,
            "bbox": [50, 300, 200, 500],
            "description": "krzesło biurowe"
        }
    ]


def format_detection_data_for_prompt(detection_data: List[Dict[str, Any]]) -> str:
    """
    Formatuje dane detekcji do czytelnej formy dla promptu.
    
    Args:
        detection_data: Lista wykrytych obiektów.
        
    Returns:
        Sformatowany tekst z informacjami o obiektach.
    """
    if not detection_data:
        return "Brak danych z systemu detekcji obiektów."
    
    lines = ["Dane z systemu detekcji obiektów:"]
    for i, obj in enumerate(detection_data, 1):
        name = obj.get("name", "nieznany")
        conf = obj.get("conf", 0) * 100
        desc = obj.get("description", "")
        line = f"  {i}. {name} (pewność: {conf:.0f}%)"
        if desc:
            line += f" - {desc}"
        lines.append(line)
    
    return "\n".join(lines)


class VisionAgent:
    """
    Agent wizualny wykorzystujący Google Gemini 1.5 Flash do analizy obrazów.
    
    Przykład użycia:
        agent = VisionAgent()
        response = agent.analyze_image("photo.jpg", detection_data, "Co widzisz?")
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Inicjalizuje agenta wizualnego.
        
        Args:
            api_key: Klucz API Gemini. Jeśli None, pobiera z GEMINI_API_KEY.
            model_name: Nazwa modelu Gemini do użycia.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        
        if not self.api_key:
            print("[VisionAgent] OSTRZEŻENIE: Brak klucza GEMINI_API_KEY")
        elif GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            print(f"[VisionAgent] Zainicjalizowano z modelem: {self.model_name}")
    
    def analyze_image(
        self,
        image_path: str,
        detection_data: Optional[List[Dict[str, Any]]] = None,
        question: str = "Co widzisz na tym obrazie?"
    ) -> str:
        """
        Analizuje obraz i odpowiada na pytanie.
        
        Args:
            image_path: Ścieżka do pliku obrazu.
            detection_data: Dane z detekcji obiektów (opcjonalne).
            question: Pytanie do zadania (domyślnie "Co widzisz?").
            
        Returns:
            Odpowiedź agenta jako string.
        """
        if not GEMINI_AVAILABLE:
            return "Biblioteka google-genai nie jest dostępna."
        
        if not self.client:
            return "Agent nie jest prawidłowo zainicjalizowany (brak klucza API)."
        
        # Sprawdzenie czy plik istnieje
        if not os.path.isfile(image_path):
            return f"Nie znaleziono pliku obrazu: {image_path}"
        
        # Kodowanie obrazu
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return "Nie udało się załadować obrazu."
        
        # Przygotowanie danych detekcji
        detection_info = ""
        if detection_data:
            detection_info = "\n\n" + format_detection_data_for_prompt(detection_data)
        
        # Budowanie promptu użytkownika
        user_prompt = f"{question}{detection_info}"
        
        try:
            # Przygotowanie zawartości dla Gemini
            mime_type = get_image_mime_type(image_path)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=base64.standard_b64decode(image_base64),
                            mime_type=mime_type
                        ),
                        types.Part.from_text(text=user_prompt)
                    ]
                )
            ]
            
            # Konfiguracja generacji
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500,
                system_instruction=SYSTEM_PROMPT_PL
            )
            
            # Wywołanie API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Ekstrakcja odpowiedzi
            if response and response.text:
                return response.text.strip()
            else:
                return "Nie otrzymano odpowiedzi od modelu."
                
        except Exception as e:
            print(f"[VisionAgent] Błąd API: {e}")
            return f"Wystąpił błąd podczas analizy obrazu: {str(e)}"
    
    def analyze_image_mock(self, question: str = "Co widzisz?") -> str:
        """
        Testowa metoda analizy z symulowanymi danymi (bez rzeczywistego obrazu).
        
        Używa wbudowanych danych mockup do demonstracji działania agenta.
        
        Args:
            question: Pytanie do zadania.
            
        Returns:
            Symulowana odpowiedź agenta.
        """
        mock_detection = simulate_detection_data()
        
        # Generujemy odpowiedź na podstawie mockup danych
        objects = [obj["name"] for obj in mock_detection]
        
        response = f"""Widzę scenę biurową. Na obrazie znajduje się osoba stojąca przy biurku. 
Na biurku stoi laptop oraz kubek. Obok widać krzesło biurowe. 
Wykryto {len(mock_detection)} obiekty: {", ".join(objects)}."""
        
        return response
    
    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        detection_data: Optional[List[Dict[str, Any]]] = None,
        question: str = "Co widzisz na tym obrazie?"
    ) -> str:
        """
        Analizuje obraz podany jako bajty (np. z kamery).
        
        Args:
            image_bytes: Dane obrazu jako bytes.
            mime_type: Typ MIME obrazu.
            detection_data: Dane z detekcji obiektów (opcjonalne).
            question: Pytanie do zadania.
            
        Returns:
            Odpowiedź agenta jako string.
        """
        if not GEMINI_AVAILABLE:
            return "Biblioteka google-genai nie jest dostępna."
        
        if not self.client:
            return "Agent nie jest prawidłowo zainicjalizowany (brak klucza API)."
        
        # Przygotowanie danych detekcji
        detection_info = ""
        if detection_data:
            detection_info = "\n\n" + format_detection_data_for_prompt(detection_data)
        
        # Budowanie promptu użytkownika
        user_prompt = f"{question}{detection_info}"
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=user_prompt)
                    ]
                )
            ]
            
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500,
                system_instruction=SYSTEM_PROMPT_PL
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Nie otrzymano odpowiedzi od modelu."
                
        except Exception as e:
            print(f"[VisionAgent] Błąd API: {e}")
            return f"Wystąpił błąd podczas analizy obrazu: {str(e)}"


# === URUCHOMIENIE STANDALONE ===
if __name__ == "__main__":
    print("=== Vision Agent - Test ===\n")
    
    agent = VisionAgent()
    
    # Test z mockup danymi
    print("Test analyze_image_mock():")
    response = agent.analyze_image_mock("Co widzisz?")
    print(f"Odpowiedź: {response}\n")
    
    # Test symulacji danych detekcji
    print("Symulowane dane detekcji:")
    detection_data = simulate_detection_data()
    for obj in detection_data:
        print(f"  - {obj['name']} ({obj['conf']*100:.0f}%)")
    print()
    
    # Test formatowania danych
    print("Sformatowane dane dla promptu:")
    print(format_detection_data_for_prompt(detection_data))
