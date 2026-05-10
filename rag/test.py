# imports
from langchain_text_splitters import CharacterTextSplitter

# Read in State of the Union Address File
with open("./documents/2024_state_of_the_union.txt", encoding="utf-8") as f:
    state_of_the_union = f.read()

# Initialize Text Splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Create Documents (Chunks) From File
texts = text_splitter.create_documents([state_of_the_union])

# imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Get Embeddings Model
api_key = ""
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings
)

# Save Document Chunks to Vector Store
ids = vector_store.add_documents(texts)

# Query the Vector Store
results = vector_store.similarity_search(
    'Who invaded Ukraine?',
    k=2
)

# Print Resulting Chunks
for res in results:
    print(f"* {res.page_content} [{res.metadata}]\n\n")