# constants.py

CONNECTION_ARGS = {
    "host": "localhost",  # Replace 'localhost' with your actual host if different
    "port": "19530"       # Default port for Milvus
}

COLLECTION_NAME = "qna_collection"  # Replace with your desired collection name
SPARSE_COLLECTION_NAME = "qna_collection_SPARSE"
# You can also add other relevant constants as needed
VECTOR_DIM = 1536  # Example dimension size for embeddings
INDEX_TYPE = "IVF_FLAT"  # Index type for Milvus, adjust based on your needs
METRIC_TYPE = "L2"  # Metric type for similarity search, can be L2, IP, etc.
PARTITION_TAG = "default_partition"  # Optional: Specify a partition tag

