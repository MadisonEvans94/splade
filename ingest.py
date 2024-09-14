import os
from typing import Dict, List, Tuple
import uuid
import click
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pdfminer.high_level import extract_text
from constants import CONNECTION_ARGS, COLLECTION_NAME, VECTOR_DIM
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import model
import nltk


nltk.download('punkt')

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
                        dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="sparse_vector",
                        dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="filename",
                        dtype=DataType.VARCHAR, max_length=256),
            # Store only the processed content embedding
            FieldSchema(name="processed_content_embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]

        schema = CollectionSchema(
            fields, description="QA Embeddings collection")
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")
    return collection


def load_documents(source_dir: str) -> List[Tuple[str, str]]:
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


def preprocess_documents(texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Splits documents into chunks and combines with the preamble.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = []
    for filename, text in tqdm(texts, desc="Splitting documents"):
        split_texts = splitter.split_text(text)
        for text_chunk in split_texts:
            chunks.append((filename, text_chunk))
    return chunks


def generate_combined_embeddings(chunks: List[Tuple[str, str]], embeddings_model) -> List:
    """
    Generates and caches embeddings for combined preamble + content.
    """
    processed_contents = [chunk[1]
                          for chunk in chunks]  # Extract combined content
    embeddings = embeddings_model.embed_documents(
        processed_contents)  # Embed combined content
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

    logging.info("Using BM25 for sparse embeddings.")
    sparse_embeddings_func = BM25SparseEmbedding(corpus=corpus)
    sparse_embeddings = sparse_embeddings_func.embed_documents(corpus)

    logging.info("Generated sparse embeddings")

    return sparse_embeddings


def insert_embeddings(dense_embeddings, sparse_embeddings, combined_embeddings, chunks, collection_name):
    """
    Inserts embeddings and additional fields into the Milvus collection.
    """
    collection = create_collection(collection_name)

    num_embeddings = len(combined_embeddings)
    ids = [str(uuid.uuid4()) for _ in range(num_embeddings)]
    filenames = [chunk[0] for chunk in chunks]  # Extract filename part
    # Original processed content for reference
    texts = [chunk[1] for chunk in chunks]

    # Insert data according to the schema
    data_to_insert = [
        ids,
        dense_embeddings,
        sparse_embeddings,
        texts,
        filenames,
        combined_embeddings  # Processed content embeddings
    ]

    output = collection.insert(data_to_insert)
    print(output)

    collection.flush()

    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    processed_content_index = {"index_type": "FLAT", "metric_type": "IP"}

    collection.create_index(field_name="dense_vector",
                            index_params=dense_index)
    collection.create_index(field_name="sparse_vector",
                            index_params=sparse_index)
    collection.create_index(field_name="processed_content_embedding",
                            index_params=processed_content_index)
    logging.info(
        f"Indexes created on fields 'dense_vector', 'sparse_vector', and 'processed_content_embedding' for collection '{collection_name}'.")

    collection.load()


# @click.command()
# @click.option('--hybrid', is_flag=True, help="Use BM25 for sparse embeddings.")
def main(hybrid):
    source_dir = "./SOURCE_DOCUMENTS"
    documents = load_documents(source_dir)
    # Example preamble
    chunks = preprocess_documents(documents)

    # Generate dense embeddings for content + preamble
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    combined_embeddings = generate_combined_embeddings(
        chunks, embeddings_model)

    # Generate sparse embeddings based on user preference
    # if hybrid:
        # sparse_embeddings = generate_sparse_embeddings(chunks, use_bm25=True)
    sparse_embeddings = generate_sparse_embeddings(chunks)
    collection_name = COLLECTION_NAME
    insert_embeddings(combined_embeddings, sparse_embeddings,
                      combined_embeddings, chunks, collection_name)


if __name__ == "__main__":
    main()
