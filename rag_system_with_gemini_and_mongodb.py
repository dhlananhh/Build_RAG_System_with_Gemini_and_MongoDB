import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time

load_dotenv()

# Cấu hình MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env file.")

DB_NAME = os.getenv("MONGO_DB_NAME", "rag_gemini_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "pdf_chunks_gemini")
VECTOR_INDEX_NAME = "vector_index_gemini"

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Failed to configure Gemini API: {e}")
    exit()

# Cấu hình Model
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = 'gemini-2.0-flash'

# Tải file PDF từ URL
def download_pdf(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading PDF from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded successfully to {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return False
    else:
        print(f"PDF file already exists at {save_path}. Skipping download.")
    return True

# Load text từ file PDF đã được download thành công
def load_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return None
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print(f"Successfully loaded text from {file_path}")
        return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

# Chia văn bản thành các đoạn văn dựa trên các khoảng trắng
def split_text(text: str) -> List[str]:
    """Split text into paragraphs based on blank lines."""
    if not text:
        return []
    chunks = re.split(r'\n\s*\n+', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Tạo Embeddings với Gemini
def generate_embeddings(texts: List[str], batch_size: int = 100) -> Optional[List[List[float]]]:
    """
    Create embeddings for lists of text paragraphs using the Gemini API.
    Batch processing to avoid 'Request payload size exceeds the limit' error.
    """
    all_embeddings = []
    print(f"Generating embeddings for {len(texts)} text chunks...")
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_texts,
                task_type="retrieval_document",
                title="PDF Document Segments"
            )
            all_embeddings.extend(response["embedding"])
            time.sleep(1)
        print("Embeddings generated successfully.")
        return all_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Tương tác với MongoDB Atlas
def get_mongodb_collection():
    """Kết nối tới MongoDB Atlas và trả về collection object."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        client.admin.command('ping')
        print(f"Successfully connected to MongoDB Atlas. DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        exit()

def insert_data_to_mongodb(collection, texts: List[str], embeddings: List[List[float]]):
    """Insert data (text and embedding) into MongoDB collection."""
    if len(texts) != len(embeddings):
        print("Error: Mismatch between number of texts and embeddings.")
        return

    documents = []
    for i in range(len(texts)):
        documents.append({
            "text": texts[i],
            "embedding": embeddings[i],
            "source_doc": pdf_path
        })

    if not documents:
        print("No documents to insert.")
        return

    print(f"Inserting {len(documents)} documents into MongoDB collection '{collection.name}'...")
    try:
        collection.insert_many(documents)
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")

def check_data_exists(collection) -> bool:
    """Kiểm tra xem collection đã có dữ liệu hay chưa."""
    try:
        count = collection.count_documents({})
        return count > 0
    except Exception as e:
        print(f"Error checking data existence in MongoDB: {e}")
        return False

def get_relevant_passages_mongodb(query: str, collection, n_results: int) -> List[str]:
    """Search for related text in MongoDB using Vector Search."""
    print(f"Generating embedding for query: '{query}'")
    try:
        query_embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=[query],
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response["embedding"][0]
        print(f"Query embedding generated. Dimension: {len(query_embedding)}")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    print(f"Querying MongoDB Atlas Vector Search (index: {VECTOR_INDEX_NAME})...")
    # Sử dụng aggregation pipeline với $vectorSearch
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": n_results
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
        print(f"Found {len(results)} relevant passages from MongoDB.")
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return [result['text'] for result in results]
    except Exception as e:
        print(f"Error querying MongoDB Atlas Vector Search: {e}")
        return []

# Tạo Prompt và Generate ra câu trả lời
def make_rag_prompt(query: str, relevant_passages: List[str]) -> str:
    """Generate complete prompts for LLM based on queries and related text."""
    combined_passage = "\n---\n".join(relevant_passages)
    escaped_passage = combined_passage.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")

    prompt = f"""
        You are a helpful and informative bot that answers questions using text from the reference passage(s) included below.
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
        strike a friendly and conversational tone. Use only the information from the PASSAGE(s) to answer the question.

        QUESTION: '{query}'

        PASSAGE(S):
        '{escaped_passage}'

        ANSWER:
    """
    return prompt

def generate_answer(prompt: str) -> str:
    """Generate answers using the Gemini model."""
    print("Generating answer using Gemini...")
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        if not response.candidates:
            print(f"Warning: Gemini response was empty or blocked.")
            try:
                print(f"Prompt feedback: {response.prompt_feedback}")
            except Exception:
                pass
            return "Sorry, I cannot create a response to this request. Content may be blocked or an error may have occurred."
        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def run_interactive_qa(collection):
    print("\n--- RAG Query Interface ---")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        query = input("Please enter your query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break
        if not query:
            print("No query provided. Please try again.")
            continue

        print("\nSearching for relevant information in MongoDB...")
        relevant_passages = get_relevant_passages_mongodb(query, collection, n_results=3)
        if not relevant_passages:
            print("No relevant information found for the given query in the document.")
            continue

        print("\nGenerating answer...")
        final_prompt = make_rag_prompt(query, relevant_passages)
        answer = generate_answer(final_prompt)
        print("\nGenerated Answer:")
        print(answer)
        print("-" * 30 + "\n")

if __name__ == "__main__":
    # pdf_url = "https://services.google.com/fh/files/misc/ai_adoption_framework_whitepaper.pdf"
    # pdf_path = "ai_adoption_framework_whitepaper.pdf"
    # pdf_url = "https://services.google.com/fh/files/misc/a_platform-centric_approach_to_scaling_generative_ai_in_the_enterprise.pdf"
    # pdf_path = "a_platform-centric_approach_to_scaling_generative_ai_in_the_enterprise.pdf"
    # pdf_url = "https://services.google.com/fh/files/misc/improve_llm_performance_reliability.pdf"
    # pdf_path = "improve_llm_performance_reliability.pdf"
    pdf_url = "https://services.google.com/fh/files/misc/generative_ai_value_evaluation_framework.pdf"
    pdf_path = "generative_ai_value_evaluation_framework.pdf"

    if not download_pdf(pdf_url, pdf_path):
        exit()

    pdf_text = load_pdf(pdf_path)
    if not pdf_text:
        exit()

    chunked_text = split_text(pdf_text)
    if not chunked_text:
        print("Cannot split PDF document.")
        exit()
    print(f"Split PDF into {len(chunked_text)} chunks.")

    collection = get_mongodb_collection()

    if not check_data_exists(collection):
        print("Collection is empty. Generating embeddings and inserting data...")
        embeddings = generate_embeddings(chunked_text)
        if embeddings:
            insert_data_to_mongodb(collection, chunked_text, embeddings)
            print("Data preparation complete. Make sure the Vector Search Index is 'Active' in Atlas before proceeding.")
            # time.sleep(180)
        else:
            print("Failed to generate embeddings. Exiting.")
            exit()
    else:
        print("Data already exists in MongoDB collection. Skipping embedding generation and insertion.")
        print("Make sure the Vector Search Index is 'Active' in Atlas.")

    run_interactive_qa(collection)
