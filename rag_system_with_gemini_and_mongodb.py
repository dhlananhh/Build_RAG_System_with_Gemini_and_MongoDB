import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel # Cần thiết nếu muốn tạo index bằng code
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time # Để theo dõi thời gian

# --- 1. Tải và Thiết lập ban đầu ---
load_dotenv()

# Cấu hình MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env file.")

DB_NAME = os.getenv("MONGO_DB_NAME", "rag_gemini_db") # Tên database
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "pdf_chunks_gemini") # Tên collection
VECTOR_INDEX_NAME = "vector_index_gemini" # Tên Vector Search Index đã tạo trên Atlas

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
GENERATION_MODEL = 'gemini-1.5-flash'

# --- 2. Các hàm tiện ích (Giữ nguyên từ code gốc) ---

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
            return False # Trả về False nếu lỗi
    else:
        print(f"PDF file already exists at {save_path}. Skipping download.")
    return True # Trả về True nếu thành công hoặc file đã tồn tại

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

def split_text(text: str) -> List[str]:
    """Chia văn bản thành các đoạn dựa trên dòng trắng."""
    if not text:
        return []
    chunks = re.split(r'\n\s*\n+', text) # Regex chặt chẽ hơn để xử lý nhiều dòng trắng
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- 3. Tạo Embeddings với Gemini ---

def generate_embeddings(texts: List[str], batch_size: int = 100) -> Optional[List[List[float]]]:
    """
    Tạo embeddings cho danh sách các đoạn văn bản sử dụng Gemini API.
    Xử lý theo batch để tránh lỗi 'Request payload size exceeds the limit'.
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
                title="PDF Document Segments" # Có thể tùy chỉnh title
            )
            all_embeddings.extend(response["embedding"])
            time.sleep(1) # Thêm độ trễ nhỏ để tránh rate limit (nếu cần)
        print("Embeddings generated successfully.")
        return all_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Xử lý lỗi chi tiết hơn (ví dụ: retry, logging) nếu cần
        return None

# --- 4. Tương tác với MongoDB Atlas ---

def get_mongodb_collection():
    """Kết nối tới MongoDB Atlas và trả về collection object."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Kiểm tra kết nối (tùy chọn nhưng hữu ích)
        client.admin.command('ping')
        print(f"Successfully connected to MongoDB Atlas. DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        exit()

def insert_data_to_mongodb(collection, texts: List[str], embeddings: List[List[float]]):
    """Chèn dữ liệu (text và embedding) vào MongoDB collection."""
    if len(texts) != len(embeddings):
        print("Error: Mismatch between number of texts and embeddings.")
        return

    documents = []
    for i in range(len(texts)):
        documents.append({
            "text": texts[i],
            "embedding": embeddings[i],
            "source_doc": pdf_path # Thêm thông tin nguồn nếu muốn
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
        # Giả sử chưa có dữ liệu nếu có lỗi để thử chạy tiếp
        return False

def get_relevant_passages_mongodb(query: str, collection, n_results: int) -> List[str]:
    """Tìm kiếm các đoạn văn bản liên quan trong MongoDB bằng Vector Search."""
    print(f"Generating embedding for query: '{query}'")
    try:
        query_embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=[query],
            task_type="retrieval_query" # Sử dụng task_type phù hợp cho query
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
                "path": "embedding",        # Trường chứa vector
                "queryVector": query_embedding, # Vector của câu query
                "numCandidates": 150,       # Số lượng ứng viên tiềm năng để xem xét (tăng độ chính xác, giảm tốc độ)
                "limit": n_results         # Số lượng kết quả cuối cùng trả về
            }
        },
        {
            "$project": { # Chỉ lấy những trường cần thiết
                "_id": 0,            # Bỏ trường _id mặc định
                "text": 1,           # Lấy nội dung text
                "score": {           # Lấy điểm tương đồng (tùy chọn)
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
        print(f"Found {len(results)} relevant passages from MongoDB.")
        # Sắp xếp lại theo score nếu cần (thường Atlas đã sắp xếp)
        # results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return [result['text'] for result in results]
    except Exception as e:
        print(f"Error querying MongoDB Atlas Vector Search: {e}")
        # Kiểm tra lỗi thường gặp:
        # - Index chưa được tạo hoặc chưa Active.
        # - Tên index hoặc trường 'path' sai.
        # - Sai số chiều ('numDimensions') khi định nghĩa index.
        # - Lỗi mạng hoặc cấu hình quyền truy cập Atlas.
        return []

# --- 5. Tạo Prompt và Sinh câu trả lời (Giữ nguyên từ code gốc) ---

def make_rag_prompt(query: str, relevant_passages: List[str]) -> str:
    """Tạo prompt hoàn chỉnh cho LLM dựa trên query và các đoạn văn bản liên quan."""
    combined_passage = "\n---\n".join(relevant_passages)
    # Thoát các ký tự đặc biệt cơ bản (có thể cần kỹ hơn tùy vào nội dung)
    escaped_passage = combined_passage.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")

    prompt = f"""
Bạn là một trợ lý AI hữu ích và cung cấp thông tin dựa trên các đoạn văn bản tham khảo được cung cấp dưới đây.
Hãy trả lời một cách đầy đủ, bao gồm các thông tin nền tảng liên quan nếu có trong văn bản.
Tuy nhiên, hãy dùng ngôn ngữ dễ hiểu như đang nói chuyện với người không có chuyên môn kỹ thuật, giải thích các khái niệm phức tạp nếu cần.
Chỉ sử dụng thông tin từ các ĐOẠN VĂN BẢN để trả lời câu hỏi. KHÔNG được bịa thêm thông tin.
Luôn trả lời bằng tiếng Việt.

Câu hỏi: '{query}'

ĐOẠN VĂN BẢN THAM KHẢO:
'{escaped_passage}'

TRẢ LỜI:
    """
    return prompt

def generate_answer(prompt: str) -> str:
    """Sinh câu trả lời bằng mô hình Gemini."""
    print("Generating answer using Gemini...")
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        # Kiểm tra xem response có lỗi không (safety ratings, etc.)
        if not response.candidates:
             print(f"Warning: Gemini response was empty or blocked.")
             # Kiểm tra xem có phải do safety không
             try:
                 print(f"Prompt feedback: {response.prompt_feedback}")
             except Exception:
                 pass
             return "Xin lỗi, tôi không thể tạo câu trả lời cho yêu cầu này. Có thể nội dung bị chặn hoặc có lỗi xảy ra."

        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "Xin lỗi, tôi gặp lỗi khi cố gắng tạo câu trả lời."

# --- 6. Hàm chạy chính và Tương tác ---

def run_interactive_qa(collection):
    """Chạy vòng lặp hỏi đáp tương tác với người dùng."""
    print("\n--- Giao diện Hỏi-Đáp RAG với MongoDB & Gemini ---")
    print("Nhập 'exit' hoặc 'quit' để thoát.")
    while True:
        query = input("Nhập câu hỏi của bạn: ")
        if query.lower() in ['exit', 'quit']:
            print("Đang thoát...")
            break
        if not query:
            print("Bạn chưa nhập câu hỏi. Vui lòng thử lại.")
            continue

        print("\nĐang tìm kiếm thông tin liên quan trong MongoDB...")
        relevant_passages = get_relevant_passages_mongodb(query, collection, n_results=3) # Lấy 3 đoạn liên quan nhất

        if not relevant_passages:
            print("Không tìm thấy thông tin liên quan cho câu hỏi này trong tài liệu PDF.")
            continue

        # print("\n--- Các đoạn văn bản liên quan tìm thấy ---")
        # for i, passage in enumerate(relevant_passages):
        #     print(f"Passage {i+1}:\n{passage}\n---")

        print("\nĐang tạo câu trả lời...")
        final_prompt = make_rag_prompt(query, relevant_passages)
        # print("\n--- Final Prompt to Gemini ---") # Bỏ comment nếu muốn debug prompt
        # print(final_prompt)
        # print("--- End of Prompt ---")

        answer = generate_answer(final_prompt)

        print("\n=> Câu trả lời:")
        print(answer)
        print("-" * 30 + "\n")


if __name__ == "__main__":
    # Tải PDF
    pdf_url = "https://services.google.com/fh/files/misc/ai_adoption_framework_whitepaper.pdf"
    pdf_path = "ai_adoption_framework_whitepaper.pdf"
    if not download_pdf(pdf_url, pdf_path):
        exit() # Thoát nếu không tải được PDF

    # Load và chunk text từ PDF
    pdf_text = load_pdf(pdf_path)
    if not pdf_text:
        exit() # Thoát nếu không đọc được PDF

    chunked_text = split_text(pdf_text)
    if not chunked_text:
        print("Không thể chia nhỏ văn bản PDF.")
        exit()
    print(f"Split PDF into {len(chunked_text)} chunks.")

    # Kết nối tới MongoDB
    collection = get_mongodb_collection()

    # Kiểm tra xem dữ liệu đã tồn tại chưa, nếu chưa thì tạo embeddings và chèn vào DB
    if not check_data_exists(collection):
        print("Collection is empty. Generating embeddings and inserting data...")
        embeddings = generate_embeddings(chunked_text)
        if embeddings:
            insert_data_to_mongodb(collection, chunked_text, embeddings)
            print("Data preparation complete. Make sure the Vector Search Index is 'Active' in Atlas before proceeding.")
            # Bạn có thể thêm đoạn chờ ở đây nếu muốn tự động hóa hơn
            # time.sleep(180) # Chờ 3 phút để index build (ví dụ)
        else:
            print("Failed to generate embeddings. Exiting.")
            exit()
    else:
        print("Data already exists in MongoDB collection. Skipping embedding generation and insertion.")
        print("Make sure the Vector Search Index is 'Active' in Atlas.")


    # --- Chạy câu hỏi đầu tiên (tùy chọn) ---
    # initial_query = "What is the AI Maturity Scale?"
    # print(f"\n--- Running initial query: '{initial_query}' ---")
    # relevant_passages = get_relevant_passages_mongodb(initial_query, collection, n_results=3)
    # if relevant_passages:
    #     final_prompt = make_rag_prompt(initial_query, relevant_passages)
    #     answer = generate_answer(final_prompt)
    #     print("\nInitial Query Answer:")
    #     print(answer)
    # else:
    #     print("Could not find relevant info for the initial query.")
    # print("-" * 30)

    # --- Bắt đầu chế độ Hỏi-Đáp tương tác ---
    run_interactive_qa(collection)
