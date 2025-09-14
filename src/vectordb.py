import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src import CURRENT_MODEL, END_POINT_PROCESS_QUESTION

DATA_FOLDER = "data"
SUPPORTED_EXTENSIONS = [".json", ".jsonl"]
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "conversations"
TOPIC = "Topic"
KEYWORDS = "Keywords"
MENTIONED_NAMES = "Mentioned_names"
CATEGORIES = "Categories"
CONTENT = "Content"

METADATA_GENERATION_PROMPT = """
Jesteś ekspertem w analizie rozmów. Przeanalizuj podaną rozmowę i wygeneruj metadane.

Metadane powinny zawierać:
- keywords: lista najważniejszych słów kluczowych z rozmowy (z wypowiedzi użytkowników)
- mentioned_names: lista imion i nazwisk które były wymienione podczas rozmowy 
- main_topic: krótki opis głównego tematu rozmowy na podstawie wypowiedzi użytkowników (1-2 zdania)
- categories: lista kategorii lub etykiet grupujących rozmowę (między 10-20 słów)

Skup się na wypowiedziach użytkowników, ignoruj nasze odpowiedzi systemowe.

Treść Konwersacji:
{conversation_content}
"""

SPEAKER_ANALYSIS_PROMPT = """
Przeanalizuj rozmowę i zidentyfikuj uczestników. Zwróć strukturę danych zgodną z wymaganym formatem.

Dla każdego uczestnika:
- Zbierz wszystkie jego wypowiedzi
- Napisz krótkie streszczenie (2-4 zdania)

Treść Konwersacji:
{conversation_content}
"""


class ConversationMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")

class Speaker(BaseModel):
    name: str = Field(description="Imię lub identyfikator uczestnika")
    messages: List[str] = Field(description="Lista wypowiedzi uczestnika")
    summary: str = Field(description="Streszczenie wypowiedzi uczestnika")

class ProcessingResult(BaseModel):
    filename: str
    conversation_content: Dict[str, Any]
    metadata: ConversationMetadata
    processing_id: str


def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    """Loguje odpowiedź LLM z pomiarem czasu."""
    print(f"Query: {user_query[:100]}...")
    print(f"Agent: {agent_name}")
    print(f"Response time: {response_time:.3f} seconds")
    print("-" * 50)

def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:

    agent = Agent(
        model=CURRENT_MODEL,
        output_type=output_type,
        system_prompt=system_prompt
    )
    start_time = time.time()
    result = agent.run_sync(content)
    response_time = time.time() - start_time
    output = result.output
    log_llm_response(content, agent_name, str(output), response_time)
    return output, response_time

def check_file(file_path: str) -> bool:
    """Check if file exists and has supported extension"""
    file = Path(file_path)
    
    if not file.exists():
        print(f"Error: File {file_path} does not exist.")
        return False
    
    if file.suffix not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported extension {file.suffix}. Supported: {SUPPORTED_EXTENSIONS}")
        return False
    
    return True


def load_json_file(file: Path) -> str:
    """Load JSON file and convert to text"""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

def load_jsonl_file(file: Path) -> str:
    """Load JSONL file and convert to text"""
    lines = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                lines.append(json.dumps(data, ensure_ascii=False))
    return '\n'.join(lines)

def load_data(file_path: str) -> Optional[str]:
    """Load data from file based on extension"""
    file = Path(file_path)
    
    try:
        if file.suffix == ".json":
            return load_json_file(file)
        elif file.suffix == ".jsonl":
            return load_jsonl_file(file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def generate_metadata(conversation_content: str) -> Optional[ConversationMetadata]:
    """Generate metadata for conversation using AI agent"""
    try:
        prompt = METADATA_GENERATION_PROMPT.format(conversation_content=conversation_content)
        metadata, _ = run_agent_with_logging(
            prompt, "metadata_agent", "Jesteś ekspertem w analizie rozmów i generowaniu metadanych.", ConversationMetadata
        )
        return metadata
    except Exception as e:
        print(f"Error generating metadata: {e}")
        return None
def get_nested_value(data, key_path):
    keys = key_path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def process_conversation_content(conversation_content):
    if isinstance(conversation_content, str):
        try:
            conversation_data = json.loads(conversation_content)
        except json.JSONDecodeError:
            lines = conversation_content.strip().split('\n')
            conversation_data = []
            for line in lines:
                if line.strip():
                    try:
                        conversation_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    else:
        conversation_data = conversation_content
    keys_to_extract = [
        "type", "session_id", "group_id", "speaker_id", "turn_ids", 
        "text_full", "category", "reply_hint"
    ]

    if isinstance(conversation_data, list):
        result = {"conversations": []}
        for item in conversation_data:
            item_result = {}
            for key in keys_to_extract:
                value = get_nested_value(item, key)
                if value is not None:
                    item_result[key] = value

            if item_result or not keys_to_extract:
                if not item_result:
                    item_result = item
                result["conversations"].append(item_result)
        return result
    else:
        result = {}
        for key in keys_to_extract:
            value = get_nested_value(conversation_data, key)
            if value is not None:
                if isinstance(value, list):
                    result[key] = value
                else:
                    result[key] = value
        if not result:
            result = conversation_data
            
        return result


def process_file(file_path: str) -> Optional[ProcessingResult]:
    """Process single conversation file"""
    print(f"Processing file: {file_path}")

    if not check_file(file_path):
        return None

    conversation_content = load_data(file_path)
    processed_conversation_content = process_conversation_content(conversation_content)
    
    print(f"Loaded {len(processed_conversation_content)} characters from {file_path}")

    print("Generating metadata...")
    metadata = generate_metadata(json.dumps(processed_conversation_content, ensure_ascii=False))
    if metadata is None:
        return None

    result = ProcessingResult(
        filename=Path(file_path).name,
        conversation_content=processed_conversation_content,
        metadata=metadata,
        processing_id=str(uuid.uuid4())
    )
    
    print(f"Successfully processed file: {file_path}")
    return result

def batch_process(folder: str = DATA_FOLDER) -> List[ProcessingResult]:
    """Process all supported files in folder"""
    results = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder} does not exist.")
        return results

    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))
    
    if not files_to_process:
        print(f"No supported files found in folder {folder}")
        return results
    
    print(f"Found {len(files_to_process)} files to process")

    for file in files_to_process:
        result = process_file(str(file))
        if result:
            results.append(result)
    
    print(f"Successfully processed {len(results)} of {len(files_to_process)} files")
    return results

def save_results(results: List[ProcessingResult], output_file: str = "processing_results.json"):
    """Save processing results to JSON file"""
    try:
        data_to_save = []
        for result in results:
            data_to_save.append(result.model_dump())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2, separators=(',', ': '))
        
        print(f"Results saved to file: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def initialize_vector_db():
    """Initialize ChromaDB client and collection"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction()
        )
        print(f"Vector database initialized at: {CHROMA_DB_PATH}")
        return client, collection
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        return None, None

def add_to_vector_db(collection, results: List[ProcessingResult]):
    """Add processed conversations to vector database"""
    if not collection:
        print("No vector database collection available")
        return
    
    documents = []
    metadatas = []
    ids = []

    for result in results:
        doc_text = json.dumps(result.conversation_content, ensure_ascii=False)

        documents.append(doc_text)

        metadata = {
            TOPIC: result.metadata.main_topic,
            KEYWORDS: ", ".join(result.metadata.keywords[:20]),
            CATEGORIES: ", ".join(result.metadata.categories[:5]),
            MENTIONED_NAMES: ", ".join(result.metadata.mentioned_names)
        }
        metadatas.append(metadata)
        ids.append(result.processing_id)
    
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} conversations to vector database")
    except Exception as e:
        print(f"Error adding to vector database: {e}")

def search_vector_db(collection, query: str, n_results: int = 3):
    """Search vector database for similar conversations"""
    if not collection:
        print("No vector database collection available")
        return None
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error searching vector database: {e}")
        return None

def main():
    """Main program function"""
    print("=== Conversation File Processor ===")

    print("Initializing vector database...")
    client, collection = initialize_vector_db()

    results = batch_process()

    if results:

        save_results(results)
        if collection:
            print("Testing API with processed conversations...")
            for res in results:
                text_content = ""
                if 'conversations' in res.conversation_content:
                    for conv in res.conversation_content['conversations']:
                        if 'text_full' in conv:
                            text_content += conv['text_full'] + " "
                elif 'text_full' in res.conversation_content:
                    text_content = res.conversation_content['text_full']
                
                if text_content.strip():
                    try:
                        data = {'content': text_content.strip()}
                        response = requests.post(END_POINT_PROCESS_QUESTION, json=data)
                        print(f"API Response for '{text_content[:50]}...': {response.status_code}")
                        if response.status_code == 200:
                            print(f"Response: {response.json()}")
                        else:
                            print(f"Error: {response.text}")
                    except Exception as e:
                        print(f"Error calling API: {e}")
            print("Adding conversations to vector database...")
            add_to_vector_db(collection, results)
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    main()