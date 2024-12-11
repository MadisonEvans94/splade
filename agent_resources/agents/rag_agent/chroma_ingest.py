from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain_chroma import Chroma
from uuid import uuid4

import os

# directory of chroma_ingest.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_langchain_db")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=OpenAIEmbeddings(),
    persist_directory=PERSIST_DIR,
)

# 1. Load documents from a Wikipedia page
loader = WikipediaLoader("Nvidia", load_max_docs=3)
raw_documents = loader.load()

# 2. Split the documents into chunks using semantic chunking
text_splitter = SemanticChunker(OpenAIEmbeddings())
docs = text_splitter.create_documents(
    [doc.page_content for doc in raw_documents])

# 3. Add the chunked documents to the vector store
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

print("ingestion complete")
print(f"created {len(docs)} chunks")
