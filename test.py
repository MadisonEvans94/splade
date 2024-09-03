import pprint
from typing import Dict
from pymilvus import (
    MilvusClient, AnnSearchRequest
)
from constants import COLLECTION_NAME
from pymilvus import model
from pymilvus import (
    Collection,
    connections,
)
from constants import CONNECTION_ARGS
connections.connect(**CONNECTION_ARGS)
def sparse_to_dict(sparse_array) -> Dict[int, float]:
    # Convert sparse matrix to dictionary format
    row_indices, col_indices = sparse_array.nonzero()
    non_zero_values = sparse_array.data
    result_dict = {}
    for col_index, value in zip(col_indices, non_zero_values):
        result_dict[col_index] = value
    return result_dict


collection = Collection(COLLECTION_NAME)

# Initialize the sparse embedding function
sparse_ef = model.sparse.SpladeEmbeddingFunction(
    model_name="naver/splade-cocondenser-selfdistil",
    device="cpu",
)


# Connect to Milvus
milvus_client = MilvusClient("http://localhost:19530")

# Generate sparse embedding for the query
query = "What is Polyuria?"

# Generate sparse embedding for the query
query_sparse_emb = sparse_ef([query])

# Convert sparse embedding to dictionary format
sparse_dict = sparse_to_dict(query_sparse_emb)

# # Create the AnnSearchRequest with the sparse embedding in dictionary format
sparse_search_request = AnnSearchRequest(
    data=[sparse_dict],  # The dictionary format for sparse embedding
    anns_field="sparse_vector",  # Specify the field for sparse vectors in Milvus
    param={"metric_type": "IP"},  # Use Inner Product as the metric type
    limit=3  # Limit to 3 results
)

# Perform the search using the AnnSearchRequest
sparse_results = collection.search(
    anns_field="sparse_vector",
    # Use the converted sparse dictionary here directly
    data=[sparse_search_request.data[0]],
    param={"metric_type": "IP"},
    limit=2,
    output_fields=['pk', 'text']  # Specify output fields to return
)

# Print the results
print(f'Sparse Search Results:')
for result in sparse_results[0]:
    print(result)