import os
import uuid
import click
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, Index
from pdfminer.high_level import extract_text
from constants import CONNECTION_ARGS, COLLECTION_NAME, VECTOR_DIM
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
import nltk

import pickle

nltk.download('punkt_tab')

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)


def create_collection(collection_name):
    connect_to_milvus()
    if not utility.has_collection(collection_name):
        logging.info(
            f"Collection '{collection_name}' does not exist. Creating collection...")

        # Define fields for the collection
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR,
                        max_length=36, is_primary=True),
            FieldSchema(name="dense_vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=VECTOR_DIM),
            FieldSchema(name="sparse_vector",
                        dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="filename", dtype=DataType.VARCHAR,
                        max_length=256)  # Add filename field
        ]

        schema = CollectionSchema(
            fields, description="QA Embeddings collection")
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    return collection


def load_documents(source_dir):
    documents = []
    for filename in tqdm(os.listdir(source_dir), desc="Loading documents"):
        file_path = os.path.join(source_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append((filename, text))
        elif filename.endswith(".pdf"):
            try:
                pdf_text = extract_text(file_path)
                documents.append((filename, pdf_text))
            except Exception as e:
                logging.error(f"Failed to extract text from {filename}: {e}")
        else:
            logging.warning(f"Unsupported file type: {filename}, skipping...")
    return documents


def preprocess_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = []
    for filename, doc in tqdm(documents, desc="Splitting documents"):
        split_texts = splitter.split_text(doc)
        for text in split_texts:
            chunks.append((filename, text))
    return chunks


def generate_dense_embeddings(chunks):
    try:
        embeddings_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
        embeddings = []
        batch_size = 10
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating dense embeddings"):
            batch_chunks = [chunk[1] for chunk in chunks[i:i + batch_size]]
            batch_embeddings = embeddings_model.embed_documents(batch_chunks)
            embeddings.extend(batch_embeddings)
    except Exception as e:
        logging.error(f"Error generating dense embeddings: {e}")
        raise e
    return embeddings


def generate_sparse_embeddings(chunks):
    analyzer = build_default_analyzer(language="en")
    embeddings_model = BM25EmbeddingFunction(analyzer)

    # Fit the model on the corpus to get the statistics of the corpus
    corpus = [chunk[1] for chunk in chunks]
    embeddings_model.fit(corpus)
    # Save the BM25 model after generating embeddings
    with open('bm25_model.pkl', 'wb') as f:
        pickle.dump(embeddings_model, f)

    # Create embeddings for the documents
    embeddings = embeddings_model.encode_documents(corpus)

    return embeddings


def insert_embeddings(dense_embeddings, sparse_embeddings, chunks, collection_name):
    collection = create_collection(collection_name)

    num_embeddings = len(dense_embeddings)
    ids = [str(uuid.uuid4()) for _ in range(num_embeddings)]
    filenames = [chunk[0] for chunk in chunks]  # Extract filename part

    # Prepare the data in the correct format
    data_to_insert = [
        ids,          # List of unique IDs
        dense_embeddings,   # List of dense embedding vectors
        sparse_embeddings,  # List of sparse embedding vectors
        [chunk[1] for chunk in chunks],  # Corresponding chunks of text
        filenames  # Corresponding filenames
    ]

    # Insert data into the collection
    logging.info("Inserting data into Milvus collection...")
    for i in tqdm(range(0, num_embeddings, 100), desc="Inserting batches"):
        batch_data = [data[i:i + 100] for data in data_to_insert]
        collection.insert(data=batch_data)
    collection.flush()

    # Create indexes on the 'dense_vector' and 'sparse_vector' fields
    dense_index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    sparse_index_params = {
        "index_type": "SPARSE_INVERTED_INDEX",  # or "SPARSE_WAND"
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0.2}
    }

    collection.create_index(field_name="dense_vector",
                            index_params=dense_index_params)
    collection.create_index(field_name="sparse_vector",
                            index_params=sparse_index_params)
    logging.info(
        f"Indexes created on fields 'dense_vector' and 'sparse_vector' for collection '{collection_name}'.")

    # Load the collection into memory after creating the indexes
    collection.load()


@click.command()
def main():
    source_dir = "./SOURCE_DOCUMENTS"
    documents = load_documents(source_dir)
    chunks = preprocess_documents(documents)

    dense_embeddings = generate_dense_embeddings(chunks)
    sparse_embeddings = generate_sparse_embeddings(chunks)
    collection_name = COLLECTION_NAME

    insert_embeddings(dense_embeddings, sparse_embeddings,
                      chunks, collection_name)


if __name__ == "__main__":
    main()
