from pymilvus import connections, Collection, utility
from constants import CONNECTION_ARGS


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)


def clear_collections():
    connect_to_milvus()
    # List all collections
    collection_names = utility.list_collections()

    if not collection_names:
        print("No collections found to clear.")
        return

    for name in collection_names:
        collection = Collection(name)
        collection.drop()
        print(f"Collection '{name}' has been dropped.")


if __name__ == "__main__":
    clear_collections()
