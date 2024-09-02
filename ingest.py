import os
from typing import Dict, List, Tuple
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
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import model
import nltk

import pickle

nltk.download('punkt_tab')

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

splade_ef = model.sparse.SpladeEmbeddingFunction(
    model_name="naver/splade-cocondenser-ensembledistil", device="cpu")


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


def load_documents(source_dir: str) -> List[str]:
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


def preprocess_documents(texts: List[str]) -> List[Tuple[str, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = []
    for filename, text in tqdm(texts, desc="Splitting documents"):
        split_texts = splitter.split_text(text)
        for text in split_texts:
            chunks.append((filename, text))
    return chunks


def generate_dense_embeddings(chunks: List[Tuple[str, str]]):
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    embeddings = []
    batch_size = 10
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating dense embeddings"):
        batch_chunks = [chunk[1] for chunk in chunks[i:i + batch_size]]
        batch_embeddings = embeddings_model.embed_documents(batch_chunks)
        embeddings.extend(batch_embeddings)
    return embeddings


def sparse_to_dict(sparse_array) -> Dict[int, float]:
    row_indices, col_indices = sparse_array.nonzero()
    non_zero_values = sparse_array.data
    result_dict = {}
    for col_index, value in zip(col_indices, non_zero_values):
        result_dict[col_index] = value
    return result_dict


def generate_sparse_embeddings(chunks: List[Tuple[str, str]]):
    corpus = [chunk[1] for chunk in chunks]
    sparse_embeddings_func = BM25SparseEmbedding(corpus=corpus)

    with open('bm25_embeddings.pkl', 'wb') as f:
        pickle.dump(sparse_embeddings_func, f)
    print(f"\n\n\n{type(sparse_embeddings_func)}\n\n\n")
    sparse_embeddings = sparse_embeddings_func.embed_documents(corpus)
    # sparse_embeddings = splade_ef.encode_documents(corpus)
    # sparse_embeddings_formatted = [sparse_to_dict(embedding) for embedding in sparse_embeddings]

    # return sparse_embeddings_formatted
    return sparse_embeddings


def insert_embeddings(dense_embeddings, sparse_embeddings, chunks, collection_name):
    collection = create_collection(collection_name)

    num_embeddings = len(dense_embeddings)
    ids = [str(uuid.uuid4()) for _ in range(num_embeddings)]
    filenames = [chunk[0] for chunk in chunks]  # Extract filename part

    # Prepare the data in the correct format
    data_to_insert = [
        ids,                        # List of unique IDs
        dense_embeddings,           # List of dense embedding vectors
        sparse_embeddings,          # List of sparse embedding vectors
        [chunk[1] for chunk in chunks],  # Corresponding chunks of text
        filenames                   # Corresponding filenames
    ]
    print(f"DATA TO INSERT: {data_to_insert[0]}")

    # Insert data into the collection
    output = collection.insert(data_to_insert)
    print(output)

    # Flush the collection to ensure data is written
    collection.flush()

    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}

    collection.create_index(field_name="dense_vector",
                            index_params=dense_index)
    collection.create_index(field_name="sparse_vector",
                            index_params=sparse_index)
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
