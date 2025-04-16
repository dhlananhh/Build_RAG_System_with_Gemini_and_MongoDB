import os, pymongo, pprint
from pymongo.operations import SearchIndexModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter, FilterOperator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

Settings.llm = OpenAI()
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.chunk_size = 100
Settings.chunk_overlap = 10

sample_data = SimpleDirectoryReader(input_files=["./data/MongoDB-Atlas-Best-Practices-White-Paper.pdf"]).load_data()

# Connect to your Atlas cluster
mongo_client = pymongo.MongoClient(mongo_uri)

# Instantiate the vector store
atlas_vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name = "movies",
    collection_name = "movie_collection_2",
    vector_index_name = "vector_index"
)
vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_store)

vector_store_index = VectorStoreIndex.from_documents(
  sample_data, storage_context=vector_store_context, show_progress=True
)

retriever = vector_store_index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("MongoDB Atlas security")

for node in nodes:
    print(node)

# Instantiate Atlas Vector Search as a retriever
vector_store_retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=5)

# Pass the retriever into the query engine
query_engine = RetrieverQueryEngine(retriever=vector_store_retriever)

# Prompt the LLM
response = query_engine.query('How can I secure my MongoDB Atlas cluster?')

print(response)
print("\nSource documents: ")
pprint.pprint(response.source_nodes)
