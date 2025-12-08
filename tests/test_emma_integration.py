import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_emma_flow():
    print("=== Testing EMMA Integration ===")
    
    # 1. First Interaction: Provide Information
    prompt1 = "My name is Pawel and I am testing the EMMA memory system."
    print(f"\n[User]: {prompt1}")
    
    response1 = requests.post(f"{BASE_URL}/api1/process_question", json={"content": prompt1})
    if response1.status_code == 200:
        print(f"[AI]: {response1.json()['answer']}")
    else:
        print(f"Error: {response1.text}")
        input("Press Enter to exit...")

    # Wait for background consolidation (simulated by sleep if async, but here it's sync so strictly not needed, but good practice)
    print("\n... Waiting for memory consolidation (10s) ...")
    time.sleep(10) 

    # 2. Second Interaction: Recall Information
    prompt2 = "What is my name and what am I doing?"
    print(f"\n[User]: {prompt2}")
    
    response2 = requests.post(f"{BASE_URL}/api1/process_question", json={"content": prompt2})
    if response2.status_code == 200:
        answer = response2.json()['answer']
        print(f"[AI]: {answer}")
        
        # Verification
        if "Paw" in answer or "testing" in answer:
            print("\n[SUCCESS] Memory successfully retrieved!")
        else:
            print(f"\n[FAILURE] Memory not found in response. Answer was: '{answer}'")
            input("Press Enter to exit...")
    else:
        print(f"Error: {response2.text}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    test_emma_flow()
    
