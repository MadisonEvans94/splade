from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
# For example, let's load the "LangChain" page from Wikipedia
# load just this one page
loader = WikipediaLoader("Atlanta", load_max_docs=1)
raw_documents = loader.load()

# 2. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(raw_documents)
print(f"created {len(docs)} chunks")
for d in docs: 
    print(d.page_content)

# 3. Add the chunked documents to the vector store
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

print("ingestion complete")
