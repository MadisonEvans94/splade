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

TOP_K = 5
EXIT_COMMAND = 'exit'
CONV_HISTORY_SIZE = 5  # Example size of conversation memory buffer

# Define fields and collection
PK_FIELD = "pk"
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
TEXT_FIELD = "text"

# Define search parameters for dense and sparse fields
DENSE_SEARCH_PARAMS = {"metric_type": "IP", "params": {}}
SPARSE_SEARCH_PARAMS = {"metric_type": "IP"}
