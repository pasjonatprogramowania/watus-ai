import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src import CURRENT_MODEL, END_POINT_PROCESS_QUESTION, DOCUMENT_METADATA_SYSTEM_PROMPT, CHROMADB_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

DATA_FOLDER = "data"
DATA_FILE = "question.jsonl"
SUPPORTED_EXTENSIONS = [".jsonl"]  # Only process question.jsonl
CHROMADB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
TOPIC = "Topic"
KEYWORDS = "Keywords"
MENTIONED_NAMES = "Mentioned_names"
CATEGORIES = "Categories"
CONTENT = "Content"
DOCUMENTS = "documents"
METADATAS = "metadatas"

class DocumentMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")

class ProcessingResult(BaseModel):
    filename: str
    document_content: Dict[str, Any]
    metadata: DocumentMetadata
    processing_id: str

def log_llm_response(user_query: str, agent_name: str, response: str, response_time: float):
    """Loguje odpowiedź LLM z pomiarem czasu."""
    print(f"Pytanie: {user_query}")
    print(f"Agent: {agent_name}")
    print(f"Odpowiedz: {response}")
    print(f"Czas odpowiedzi: {response_time:.3f} sekund")
    print("-" * 50)

def run_agent_with_logging(content: str, agent_name: str, system_prompt: str, output_type: type) -> tuple:
    """Uruchamia agenta z logowaniem czasu odpowiedzi."""
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


def generate_metadata(conversation_content: str) -> DocumentMetadata:
    """Generate metadata for conversation"""
    # minutes_timeout = 1
    # try:
    #     metadata, _ = run_agent_with_logging(
    #         conversation_content, "metadata_agent", DOCUMENT_METADATA_SYSTEM_PROMPT, DocumentMetadata
    #     )
    #     return metadata
    # except:
    #     time.sleep(minutes_timeout*60)

    metadata, _ = run_agent_with_logging(
        conversation_content, "metadata_agent", DOCUMENT_METADATA_SYSTEM_PROMPT, DocumentMetadata
    )
    return metadata

def process_file(file_path: str) -> List[ProcessingResult]:
    """Process single conversation file and return list of results"""
    logger.info(f"Processing file: {file_path}")

    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    document_content = json.loads(line)
                    metadata = generate_metadata(json.dumps(document_content))
                    result = ProcessingResult(
                        filename=Path(file_path).name,
                        document_content=document_content,
                        metadata=metadata,
                        processing_id=str(uuid.uuid4())
                    )
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num} in {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num} in {file_path}: {e}")

        logger.info(f"Successfully processed {len(results)} conversations from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

def batch_process(folder: str = DATA_FOLDER):
    """Process all supported files in folder"""
    results = []
    folder_path = Path(folder)

    if not folder_path.exists():
        logger.error(f"Folder {folder} does not exist.")
        return results

    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))

    if not files_to_process:
        logger.warning(f"No supported files found in folder {folder}")
        return results

    logger.info(f"Found {len(files_to_process)} files to prrocess")

    for file in files_to_process:
        file_results = process_file(str(file))
        if file_results:
            results.extend(file_results)  # Dodajemy poszczególne rezultaty, nie listy

    logger.info(f"Successfully processed {len(results)} conversations from {len(files_to_process)} files")
    return results

def initialize_vector_db():
    """Initialize ChromaDB client and collection"""
    try:
        if not Path(CHROMADB_PATH).exists():
            os.makedirs(CHROMADB_PATH, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMADB_PATH)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=DefaultEmbeddingFunction()
            )
            logger.info(f"Vector database initialized at: {CHROMADB_PATH}")
            return client, collection
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        return None, None

def add_to_vector_db(collection, results):
    """Add processed conversations to vector database"""
    if not collection:
        logger.error("No vector database collection available")
        return

    documents = []
    metadatas = []
    ids = []

    for result in results:
        doc_text = json.dumps(result.document_content, ensure_ascii=False)
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
        logger.info(f"Added {len(documents)} conversations to vector database")
    except Exception as e:
        logger.error(f"Error adding to vector database: {e}")

def return_collection(colection=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return client.get_collection(
        name=colection,
        embedding_function=DefaultEmbeddingFunction()
    )

def search_vector_db(collection, query: str, n_results: int = 3):
    """Search vector database for similar conversations"""
    if not collection:
        logger.error("No vector database collection available")
        return None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        logger.info(f"Vector search performed for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        return None

def main():
    """Main program function"""
    logger.info("=== Conversation File Processor ===")

    logger.info("Initializing vector database...")
    _, collection = initialize_vector_db()

    results = batch_process()

    if results:
        if collection:
            logger.info("Adding conversations to vector database...")
            add_to_vector_db(collection, results)
            logger.info(f"Total conversations added: {len(results)}")
    else:
        logger.warning("No files were successfully processed.")

# Automatically process files when ran as script
if __name__ == "__main__":
    main()

# --- EMMA Memory Extensions ---

MEMORY_COLLECTION_NAME = "conversation_memory"

def get_memory_collection():
    """Returns the ChromaDB collection for conversation memory."""
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction()
    )

def add_memory(content: str, metadata: Dict[str, Any], memory_id: str = None):
    """Adds a single memory node to the collection."""
    collection = get_memory_collection()
    if not memory_id:
        memory_id = str(uuid.uuid4())
    
    try:
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        logger.info(f"Added memory: {content[:50]}... (ID: {memory_id})")
        return memory_id
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return None

def search_memory(query: str, n_results: int = 3, where: Dict[str, Any] = None):
    """Searches the memory collection."""
    collection = get_memory_collection()
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where # Useful for filtering by user_id
        )
        return results
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return None

def get_memory_by_id(memory_id: str):
    """Retrieves a specific memory node by ID."""
    collection = get_memory_collection()
    try:
        return collection.get(ids=[memory_id])
    except Exception as e:
        logger.error(f"Error getting memory by ID: {e}")
        return None