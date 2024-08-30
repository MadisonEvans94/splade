import os
import click
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from constants import CONNECTION_ARGS, COLLECTION_NAME, VECTOR_DIM

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to Milvus


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)

# Check if the collection exists and create it if necessary


def create_collection():
    connect_to_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        print(
            f"Collection '{COLLECTION_NAME}' does not exist. Creating collection...")
        # Define schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]
        schema = CollectionSchema(
            fields, description="QA Embeddings collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        collection = Collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    return collection

# Load text files from the source directory


def load_documents(source_dir):
    documents = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Preprocess documents using LangChain's RecursiveCharacterTextSplitter


def preprocess_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks

# Generate dense embeddings using OpenAI


def generate_dense_embeddings(chunks):
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )
    embeddings = embeddings_model.embed_documents(chunks)
    return embeddings

# Generate sparse embeddings using BM25


def generate_sparse_embeddings(corpus):
    from langchain.embeddings import BM25Embeddings
    bm25_model = BM25Embeddings(corpus)
    embeddings = bm25_model.embed(corpus)
    return embeddings

# Insert embeddings into Milvus


# Insert embeddings into Milvus
def insert_embeddings(embeddings):
    collection = create_collection()  # Ensure collection exists or create it

    # Prepare the data in the correct format
    entities = [{"embedding": embedding}
                for embedding in embeddings]  # Align data with schema

    # Insert into the collection
    collection.insert(entities)
    print(
        f"Inserted {len(embeddings)} embeddings into Milvus collection '{COLLECTION_NAME}'.")


@click.command()
@click.option('--sparse', is_flag=True, help='Use BM25 sparse embeddings instead of dense embeddings.')
def main(sparse):
    source_dir = "./SOURCE_DOCUMENTS"
    documents = load_documents(source_dir)
    chunks = preprocess_documents(documents)

    if sparse:
        print("Using BM25 sparse embeddings...")
        embeddings = generate_sparse_embeddings(chunks)
    else:
        print("Using OpenAI dense embeddings...")
        embeddings = generate_dense_embeddings(chunks)

    insert_embeddings(embeddings)


if __name__ == "__main__":
    main()
