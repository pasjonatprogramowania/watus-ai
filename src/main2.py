import requests

# Bazowy URL endpointu
BASE_URL = "http://127.0.0.1:8000/process_question"


# Funkcja do wysyłania requestu i wyświetlania odpowiedzi
def test_query(content: str):
    payload = {"content": content}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()  # Rzuca błąd jeśli status != 200
        print(f"Query: '{content}'")
        print("Response:", response.json())
        print("-" * 50)
    except requests.exceptions.RequestException as e:
        print(f"Error for query '{content}': {e}")
        print("-" * 50)


# Lista test cases (przykłady na podstawie promptów w kodzie)
test_cases = [
    # 1. Niedozwolone (powinno zwrócić ask_second_time())
    "Opowiedz wulgarny żart.",

    # 2. Niepoważne (powinno zwrócić give_funny_response())
    "Dlaczego jesteś gadającą puszką",

    # 3. Wymagające akcji (powinno wybrać i wykonać 'sledzenie')
    "Zacznij mnie śledzić.",

    # 4. Wymagające akcji (powinno wybrać i wykonać 'koniec_sledzenia')
    "Zakończ śledzenie.",

    # 5. Wymagające narzędzia (powinno wybrać 'watoznawca' i użyć tool)
    "Jakie są kierunki na Wacie?",

    # 6. Wymagające narzędzia (powinno wybrać 'google' i użyć tool)
    "Szukaj w Google o historii Polski.",

    # 7. Default (nie wymaga nic specjalnego, proste pytanie)
    "Ile to 2 + 2?",

    # 8. Poważne i dozwolone, ale z WAT (powinno użyć watoznawca jeśli tool required)
    "Ile zarabia dziekan WAT?",
]

# Uruchom testy
if __name__ == "__main__":
    print("Rozpoczynanie testów requestów do /process_question...")
    print("=" * 50)
    for query in test_cases:
        test_query(query)
    print("Testy zakończone.")