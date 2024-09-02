from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define connection arguments (replace with your actual Milvus host and port)
CONNECTION_ARGS = {"host": "localhost", "port": "19530"}
COLLECTION_NAME = "test_collection"
VECTOR_DIM = 2  # Example dimension for simple testing


def connect_to_milvus():
    """Establish a connection to Milvus."""
    connections.connect(**CONNECTION_ARGS)
    logging.info("Connected to Milvus")


def create_test_collection(collection_name):
    """Create a simple collection for testing purposes."""
    if not utility.has_collection(collection_name):
        logging.info(
            f"Collection '{collection_name}' does not exist. Creating collection...")

        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,
                        dim=VECTOR_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
        ]

        schema = CollectionSchema(
            fields, description="Test collection for basic troubleshooting")
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(collection_name)
        logging.info(f"Collection '{collection_name}' already exists.")

    return collection


def insert_test_data(collection):
    """Insert test data into the collection."""
    test_vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Example vector data
    test_texts = ["test1", "test2", "test3"]  # Example strings

    # Prepare the data to match the schema (omit IDs since they are auto-generated)
    data_to_insert = [
        test_vectors,  # Vectors to insert
        test_texts     # Corresponding text
    ]

    logging.info("Inserting test data into collection...")
    collection.insert(data_to_insert)
    logging.info("Data insertion completed.")


def create_index(collection):
    """Create an index for the vector field."""
    index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
    collection.create_index(field_name="vector", index_params=index_params)
    logging.info("Index created on the 'vector' field.")


def main():
    connect_to_milvus()
    collection = create_test_collection(COLLECTION_NAME)

    try:
        insert_test_data(collection)
        create_index(collection)
    except Exception as e:
        logging.error(f"Error during data insertion or indexing: {e}")

    logging.info(
        f"Number of entities in collection '{COLLECTION_NAME}': {collection.num_entities}")


if __name__ == "__main__":
    main()
