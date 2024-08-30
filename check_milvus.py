from pymilvus import connections, Collection, utility
from pprint import pprint
from constants import CONNECTION_ARGS


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)


def check_collections():
    connect_to_milvus()
    # List all collections in Milvus
    collection_names = utility.list_collections()

    if not collection_names:
        print("No collections found in the database.")
        return

    for name in collection_names:
        collection = Collection(name)
        schema = collection.schema.to_dict()
        print(f"Collection Name: {name}")
        print("Schema:")
        pprint(schema)
        print("-" * 50)


if __name__ == "__main__":
    check_collections()
