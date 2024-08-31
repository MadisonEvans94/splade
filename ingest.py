import os
import uuid
import click
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, Index
from pdfminer.high_level import extract_text
from constants import CONNECTION_ARGS, COLLECTION_NAME, VECTOR_DIM

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)


def create_collection():
    connect_to_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        logging.info(
            f"Collection '{COLLECTION_NAME}' does not exist. Creating collection...")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR,
                        max_length=36, is_primary=True),
            FieldSchema(name="embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192)
        ]
        schema = CollectionSchema(
            fields, description="QA Embeddings collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        logging.info(f"Collection '{COLLECTION_NAME}' created.")
    else:
        collection = Collection(COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
    return collection


def load_documents(source_dir):
    documents = []
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filename.endswith(".pdf"):
            try:
                pdf_text = extract_text(file_path)
                documents.append(pdf_text)
            except Exception as e:
                logging.error(f"Failed to extract text from {filename}: {e}")
        else:
            logging.warning(f"Unsupported file type: {filename}, skipping...")
    return documents


def preprocess_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks


def generate_dense_embeddings(chunks):
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    embeddings = embeddings_model.embed_documents(chunks)
    return embeddings


def generate_sparse_embeddings(corpus):
    from langchain.embeddings import BM25Embeddings
    bm25_model = BM25Embeddings(corpus)
    embeddings = bm25_model.embed(corpus)
    return embeddings


def insert_embeddings(embeddings, chunks):
    collection = create_collection()
    ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

    # Prepare the data in the correct format
    data_to_insert = [
        ids,          # List of unique IDs
        embeddings,   # List of embedding vectors
        chunks        # Corresponding chunks of text
    ]

    # Insert data into the collection
    collection.insert(data=data_to_insert)
    collection.flush()

    # Create an index on the 'embedding' field
    index_params = {"index_type": "IVF_FLAT",
                    "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    logging.info(
        f"Index created on field 'embedding' for collection '{COLLECTION_NAME}'.")

    # Load the collection into memory after creating the index
    collection.load()


@click.command()
@click.option('--sparse', is_flag=True, help='Use BM25 sparse embeddings instead of dense embeddings.')
def main(sparse):
    source_dir = "./SOURCE_DOCUMENTS"
    documents = load_documents(source_dir)
    chunks = preprocess_documents(documents)

    if sparse:
        logging.info("Using BM25 sparse embeddings...")
        embeddings = generate_sparse_embeddings(chunks)
    else:
        logging.info("Using OpenAI dense embeddings...")
        embeddings = generate_dense_embeddings(chunks)

    insert_embeddings(embeddings, chunks)


if __name__ == "__main__":
    main()
