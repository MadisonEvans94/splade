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
        # Ensure the collection is loaded into memory
        collection.load()
        schema = collection.schema.to_dict()
        # This should correctly show the number of entities
        num_entities = collection.num_entities

        print(f"\nCollection Name: {name}")
        print("\nSchema:\n")
        pprint(schema)
        print(f"\nNumber of Entities: {num_entities}")
        print("-" * 50)


if __name__ == "__main__":
    check_collections()
