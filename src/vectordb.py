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
"""

class ConversationMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")

class ProcessingResult(BaseModel):
    filename: str
    conversation_content: Dict[str, Any]
    metadata: ConversationMetadata
    processing_id: str

# ---------------------- LLM utils ----------------------

def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    print(f"Query: {user_query[:100]}...")
    print(f"Agent: {agent_name}")
    print(f"Response time: {response_time:.3f} seconds")
    print("-" * 50)

def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:
    agent = Agent(model=CURRENT_MODEL, output_type=output_type, system_prompt=system_prompt)
    start_time = time.time()
    result = agent.run_sync(content)
    response_time = time.time() - start_time
    output = result.output
    log_llm_response(content, agent_name, str(output), response_time)
    return output, response_time

# ---------------------- File loading ----------------------

def check_file(file_path: str) -> bool:
    file = Path(file_path)
    if not file.exists():
        print(f"Error: File {file_path} does not exist.")
        return False
    if file.suffix not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported extension {file.suffix}. Supported: {SUPPORTED_EXTENSIONS}")
        return False
    return True

def load_json_file(file: Path) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

def load_jsonl_file(file: Path) -> str:
    lines = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                lines.append(json.dumps(data, ensure_ascii=False))
    return '\n'.join(lines)

def load_data(file_path: str) -> Optional[str]:
    file = Path(file_path)
    try:
        if file.suffix == ".json":
            return load_json_file(file)
        elif file.suffix == ".jsonl":
            return load_jsonl_file(file)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# ---------------------- Processing ----------------------

def generate_metadata(conversation_content: str) -> Optional[ConversationMetadata]:
    try:
        prompt = METADATA_GENERATION_PROMPT.format(conversation_content=conversation_content)
        metadata, _ = run_agent_with_logging(
            prompt, "metadata_agent",
            "Jesteś ekspertem w analizie rozmów i generowaniu metadanych.",
            ConversationMetadata
        )
        return metadata
    except Exception as e:
        print(f"Error generating metadata: {e}")
        return None

def process_file(file_path: str) -> Optional[ProcessingResult]:
    print(f"Processing file: {file_path}")
    if not check_file(file_path):
        return None

    conversation_content = load_data(file_path)
    if not conversation_content:
        return None

    metadata = generate_metadata(conversation_content)
    if metadata is None:
        return None

    result = ProcessingResult(
        filename=Path(file_path).name,
        conversation_content={"raw": conversation_content},
        metadata=metadata,
        processing_id=str(uuid.uuid4())
    )
    return result

def batch_process(folder: str = DATA_FOLDER) -> List[ProcessingResult]:
    results = []
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Error: Folder {folder} does not exist.")
        return results

    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))

    for file in files_to_process:
        result = process_file(str(file))
        if result:
            results.append(result)
    return results

# ---------------------- Chroma DB ----------------------

def initialize_vector_db():
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
    if not collection:
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
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(documents)} items to vector database")

def ingest_qa_jsonl(file_path="data/questions.jsonl"):
    """Ingests Q&A pairs into ChromaDB as question->answer entries"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction()
    )
    with open(file_path, encoding="utf-8") as f:
        qa_data = [json.loads(line) for line in f if line.strip()]

    documents = [item["question"] for item in qa_data]
    metadatas = [{"answer": item["answer"]} for item in qa_data]
    ids = [str(uuid.uuid4()) for _ in qa_data]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Ingested {len(documents)} QA pairs into ChromaDB")

# ---------------------- Main ----------------------

def main():
    print("=== Conversation File Processor ===")
    client, collection = initialize_vector_db()
    results = batch_process()
    if results and collection:
        add_to_vector_db(collection, results)
    print("Done")

if __name__ == "__main__":
    main()
