import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Disable ChromaDB telemetry to avoid warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from src import CURRENT_MODEL
# Configuration constants
DATA_FOLDER = "data"
SUPPORTED_EXTENSIONS = [".txt", ".json", ".jsonl"]
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "conversations"

# Polish prompts
METADATA_GENERATION_PROMPT = """
Jesteś ekspertem w analizie rozmów. Przeanalizuj podaną rozmowę i wygeneruj metadane.

UWAGA: Dane mogą być w różnych formatach:
- Plik .txt: transkrypt rozmowy w formacie "user_1: tekst", "user_2: tekst" (wypowiedzi mogą być przerywane)
- Plik .json/.jsonl: strukturalne dane z polami "user_id", "user_message", "our_response"

Metadane powinny zawierać:
- keywords: lista najważniejszych słów kluczowych z rozmowy (z wypowiedzi użytkowników)
- speaker_names: lista identyfikatorów uczestników (np. ["user_1", "user_2"])
- main_topic: krótki opis głównego tematu rozmowy na podstawie wypowiedzi użytkowników
- categories: lista kategorii lub etykiet grupujących rozmowę

Skup się na wypowiedziach użytkowników, ignoruj nasze odpowiedzi systemowe.

Treść rozmowy:
{conversation_content}
"""

SPEAKER_ANALYSIS_PROMPT = """
Przeanalizuj rozmowę i zidentyfikuj uczestników. Zwróć strukturę danych zgodną z wymaganym formatem.

Dla każdego uczestnika:
- Zbierz wszystkie jego wypowiedzi
- Napisz krótkie streszczenie (1-2 zdania)

Treść rozmowy:
{conversation_content}
"""

# Pydantic models for types only
class ConversationMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    speaker_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")

class Speaker(BaseModel):
    name: str = Field(description="Imię lub identyfikator uczestnika")
    messages: List[str] = Field(description="Lista wypowiedzi uczestnika")
    summary: str = Field(description="Streszczenie wypowiedzi uczestnika")

class SpeakerAnalysis(BaseModel):
    speakers: Dict[str, Speaker] = Field(description="Słownik uczestników rozmowy")

class ProcessingResult(BaseModel):
    filename: str
    conversation_content: str
    metadata: ConversationMetadata
    speaker_analysis: SpeakerAnalysis
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

def load_txt_file(file: Path) -> str:
    """Load text file as raw text"""
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

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
        if file.suffix == ".txt":
            return load_txt_file(file)
        elif file.suffix == ".json":
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

def parse_speakers_locally(conversation_content: str, filename: str) -> SpeakerAnalysis:
    """Parse speakers locally without AI"""
    speakers_data = {}
    
    try:
        # Try to parse as JSON first (for .json/.jsonl files)
        if filename.endswith(('.json', '.jsonl')):
            data = json.loads(conversation_content) if filename.endswith('.json') else None
            if data and 'conversation' in data:
                # Handle JSON format
                messages = data['conversation'].get('messages', [])
                for msg in messages:
                    user_id = msg.get('user_id', 'unknown')
                    user_message = msg.get('user_message', '')
                    if user_id not in speakers_data:
                        speakers_data[user_id] = []
                    if user_message:
                        speakers_data[user_id].append(user_message)
            elif filename.endswith('.jsonl'):
                # Handle JSONL format
                for line in conversation_content.split('\n'):
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            user_id = msg.get('user_id', 'unknown')
                            user_message = msg.get('user_message', '')
                            if user_id not in speakers_data:
                                speakers_data[user_id] = []
                            if user_message:
                                speakers_data[user_id].append(user_message)
                        except:
                            continue
        else:
            # Handle .txt format
            lines = conversation_content.split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        message = parts[1].strip()
                        if speaker not in speakers_data:
                            speakers_data[speaker] = []
                        if message:
                            speakers_data[speaker].append(message)
    except Exception as e:
        print(f"Error parsing speakers locally: {e}")
    
    # Create Speaker objects
    speakers = {}
    for speaker_id, messages in speakers_data.items():
        if messages:
            summary = f"Uczestnik {speaker_id} wypowiedział {len(messages)} wiadomości"
            speakers[speaker_id] = Speaker(
                name=speaker_id,
                messages=messages,
                summary=summary
            )
    
    # Ensure at least 2 speakers
    if len(speakers) == 0:
        speakers = {
            "user_1": Speaker(name="user_1", messages=["Nie znaleziono wypowiedzi"], summary="Brak danych"),
            "user_2": Speaker(name="user_2", messages=["Nie znaleziono wypowiedzi"], summary="Brak danych")
        }
    elif len(speakers) == 1:
        existing_key = list(speakers.keys())[0]
        speakers["user_2"] = Speaker(name="user_2", messages=["Drugi uczestnik nie zidentyfikowany"], summary="Brak danych")
    
    return SpeakerAnalysis(speakers=speakers)

def analyze_speakers(conversation_content: str, filename: str = "") -> Optional[SpeakerAnalysis]:
    """Analyze speakers using local parsing (no AI)"""
    try:
        return parse_speakers_locally(conversation_content, filename)
    except Exception as e:
        print(f"Error analyzing speakers: {e}")
        # Create fallback response
        fallback_analysis = SpeakerAnalysis(
            speakers={
                "user_1": Speaker(name="user_1", messages=["Fallback message 1"], summary="Fallback summary for user_1"),
                "user_2": Speaker(name="user_2", messages=["Fallback message 2"], summary="Fallback summary for user_2")
            }
        )
        print("Using fallback speaker analysis")
        return fallback_analysis

def process_file(file_path: str) -> Optional[ProcessingResult]:
    """Process single conversation file"""
    print(f"Processing file: {file_path}")
    
    # Check file
    if not check_file(file_path):
        return None
    
    # Load data
    conversation_content = load_data(file_path)
    if conversation_content is None:
        return None
    
    print(f"Loaded {len(conversation_content)} characters from {file_path}")
    
    # Generate metadata
    print("Generating metadata...")
    metadata = generate_metadata(conversation_content)
    if metadata is None:
        return None
    
    # Analyze speakers
    print("Analyzing speakers...")
    speaker_analysis = analyze_speakers(conversation_content, Path(file_path).name)
    if speaker_analysis is None:
        return None
    
    # Create result
    result = ProcessingResult(
        filename=Path(file_path).name,
        conversation_content=conversation_content,
        metadata=metadata,
        speaker_analysis=speaker_analysis,
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
    
    # Find all supported files
    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))
    
    if not files_to_process:
        print(f"No supported files found in folder {folder}")
        return results
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
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
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
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
        # Create document text from conversation content and metadata
        doc_text = f"""
        Filename: {result.filename}
        Topic: {result.metadata.main_topic}
        Keywords: {', '.join(result.metadata.keywords)}
        Speakers: {', '.join(result.metadata.speaker_names)}
        Categories: {', '.join(result.metadata.categories)}
        
        Conversation Content:
        {result.conversation_content}
        
        Speaker Analysis:
        """
        
        # Add speaker summaries to document
        for speaker_id, speaker in result.speaker_analysis.speakers.items():
            doc_text += f"\n{speaker.name}: {speaker.summary}"
        
        documents.append(doc_text)
        
        # Create metadata for ChromaDB (only strings, numbers, bools allowed)
        metadata = {
            "filename": result.filename,
            "processing_id": result.processing_id,
            "main_topic": result.metadata.main_topic,
            "keywords": ", ".join(result.metadata.keywords[:5]),  # Convert list to string
            "speaker_count": len(result.speaker_analysis.speakers),
            "categories": ", ".join(result.metadata.categories[:3]),  # Convert list to string
            "speaker_names": ", ".join(result.metadata.speaker_names)  # Add speaker names as string
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
    
    # Initialize vector database
    print("Initializing vector database...")
    client, collection = initialize_vector_db()
    
    # Batch process all files in data folder
    results = batch_process()
    
    if results:
        # Save results to JSON
        save_results(results)
        
        # Add to vector database
        if collection:
            print("Adding conversations to vector database...")
            add_to_vector_db(collection, results)
        
        # Display summary
        print("\n=== PROCESSING SUMMARY ===")
        for result in results:
            print(f"\nFile: {result.filename}")
            print(f"ID: {result.processing_id}")
            print(f"Main topic: {result.metadata.main_topic}")
            print(f"Number of speakers: {len(result.speaker_analysis.speakers)}")
            print(f"Keywords: {', '.join(result.metadata.keywords[:5])}")
        
        # Demo search functionality
        if collection:
            print("\n=== VECTOR DATABASE SEARCH DEMO ===")
            demo_queries = [
                "rozmowa o bazie danych",
                "planowanie wakacji",
                "testy jednostkowe"
            ]
            
            for query in demo_queries:
                print(f"\nSearching for: '{query}'")
                search_results = search_vector_db(collection, query, n_results=2)
                if search_results and search_results['documents']:
                    for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
                        print(f"  Result {i+1}: {metadata['filename']} - {metadata['main_topic']}")
                else:
                    print("  No results found")
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    main()