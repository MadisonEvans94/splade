import logging
from pydantic import Field
from typing import Dict, List
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain.embeddings.base import Embeddings
from pymilvus import (
    AnnSearchRequest,
    Collection,
    WeightedRanker,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pymilvus import Collection
from pymilvus import model 

# Initialize the sparse embedding function
sparse_ef = model.sparse.SpladeEmbeddingFunction(
    model_name="naver/splade-cocondenser-selfdistil",
    device="cpu",
)


def sparse_to_dict(sparse_array) -> Dict[int, float]:
    # Convert sparse matrix to dictionary format
    row_indices, col_indices = sparse_array.nonzero()
    non_zero_values = sparse_array.data
    result_dict = {}
    for col_index, value in zip(col_indices, non_zero_values):
        result_dict[col_index] = value
    return result_dict

class SpladeSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on SPLADE.

    This class uses the SPLADE model in Milvus to implement sparse vector embedding.
    SPLADE leverages BERT's capabilities to create a learnable sparse vector.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str = "cpu"):
        from pymilvus.model.sparse import SpladeEmbeddingFunction  # type: ignore
       
        # Initialize SPLADE embedding function from Milvus
        self.splade_ef = SpladeEmbeddingFunction(
            model_name=model_name, device=device)

    def embed_query(self, query: str) -> Dict[int, float]:
        # Encode a query into a SPLADE sparse vector
        query_sparse_emb = self.splade_ef([query])
        # print(f"query_sparse_emb: {query_sparse_emb}")
        sparse_dict = self._sparse_to_dict(query_sparse_emb)
        # print(f"sparse_dict: {sparse_dict}")
        return sparse_dict

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        # Encode documents into SPLADE sparse vectors
        sparse_arrays = self.splade_ef.encode_documents(texts)
        return [self._sparse_to_dict(sparse_array) for sparse_array in sparse_arrays]

    def _sparse_to_dict(self, sparse_array) -> Dict[int, float]:
        # Convert sparse matrix to dictionary format
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        result_dict = {}
        for col_index, value in zip(col_indices, non_zero_values):
            result_dict[col_index] = value
        return result_dict
    
class StandardRetriever(BaseRetriever):
    """Retriever that performs standard dense retrieval using Milvus."""

    collection: Collection = Field(...)
    dense_field: str = Field(...)
    top_k: int = Field(...)
    embeddings_model: Embeddings = Field(...)

    def __init__(self, collection: Collection, dense_field: str, top_k: int, embeddings_model: Embeddings):
        # Initialize fields using pydantic's BaseModel mechanism
        super().__init__(collection=collection, dense_field=dense_field,
                         top_k=top_k, embeddings_model=embeddings_model)
        self.collection = collection
        self.dense_field = dense_field
        self.top_k = top_k
        self.embeddings_model = embeddings_model

    def _retrieve_dense_documents(self, query: str) -> List[Document]:
        """Perform standard dense retrieval using Milvus."""

        # Convert the query into dense embeddings
        dense_query_embedding = self.embeddings_model.embed_query(query)

        # Define search parameters for dense retrieval
        search_params = {"metric_type": "IP", "params": {}}

        # Perform the search using the dense vector field
        results = self.collection.search(
            data=[dense_query_embedding],
            anns_field=self.dense_field,
            param=search_params,
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

        # Extract the documents from the search results
        documents = [
            Document(
                page_content=hit.get("text"),
                metadata={"pk": hit.id, "retriever": "dense"}
            )
            for hits in results for hit in hits
        ]

        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve documents using standard dense retrieval."""
        # Call the dense retrieval function
        return self._retrieve_dense_documents(query)

class CustomHybridRetriever(BaseRetriever):
    """Custom retriever to retrieve documents using both dense and sparse embeddings."""

    collection: Collection = Field(...)
    dense_field: str = Field(...)
    sparse_field: str = Field(...)
    top_k: int = Field(...)
    embeddings_model: Embeddings = Field(...)
    sparse_embeddings_model: BaseSparseEmbedding = Field(...)

    def __init__(self, collection: Collection, dense_field: str, sparse_field: str, top_k: int, embeddings_model: Embeddings, sparse_embeddings_model: BaseSparseEmbedding, ratio: List[float] = [0.5, 0.5]):
        # Properly initialize fields using pydantic's BaseModel mechanism
        super().__init__(collection=collection, dense_field=dense_field,
                         top_k=top_k, embeddings_model=embeddings_model)
        self.collection = collection
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.top_k = top_k
        self.embeddings_model = embeddings_model
        self.sparse_embeddings_model = sparse_embeddings_model

    def _retrieve_dense_request(self, query: str) -> AnnSearchRequest:
        """Create ANN Search Request for dense embeddings."""
        # Convert the query into dense embeddings
        dense_query_embedding = self.embeddings_model.embed_query(query)

        # Create ANN Search Request for dense embeddings
        dense_request = AnnSearchRequest(
            data=[dense_query_embedding],
            anns_field=self.dense_field,
            param={"metric_type": "IP", "params": {}},
            limit=self.top_k
        )
        
        return dense_request

    def _retrieve_sparse_request(self, query: str) -> AnnSearchRequest:
        """Create ANN Search Request for sparse embeddings."""
        sparse_embedding = SpladeSparseEmbedding()
        sparse_dict = sparse_embedding.embed_query(query)

        # # Create the AnnSearchRequest with the sparse embedding in dictionary format
        sparse_request = AnnSearchRequest(
            data=[sparse_dict],  # The dictionary format for sparse embedding
            anns_field="sparse_vector",  # Specify the field for sparse vectors in Milvus
            param={"metric_type": "IP"},  # Use Inner Product as the metric type
            limit=3  # Limit to 3 results
        )
        return sparse_request

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve documents using both dense and sparse embeddings, with reranking."""

        # Create ANN Search Requests using dense and sparse retrieval methods
        dense_request = self._retrieve_dense_request(query)
        sparse_request = self._retrieve_sparse_request(query)
        
        # Perform dense and sparse searches separately for debugging
        dense_results = self.collection.search(
            data=[dense_request.data[0]],
            anns_field=self.dense_field,
            param={"metric_type": "IP", "params": {}},
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

        sparse_results = self.collection.search(
            data=[sparse_request.data[0]],
            anns_field=self.sparse_field,
            param={"metric_type": "IP"},
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

        # Log the results for debugging
        # logging.info(
        #     f"Dense results: {[hit.get('text') for hits in dense_results for hit in hits]}")
        logging.info(
            f"Sparse results: {[hit.get('text') for hits in sparse_results for hit in hits]}")


        # Execute hybrid search with reranking
        reranker = WeightedRanker(0.5, 0.5)  # Example weights for both routes
        results = self.collection.hybrid_search(
            [dense_request, sparse_request],
            rerank=reranker,
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

        # Extract documents from the search results
        documents = [
            Document(
                page_content=hit.get("text"),
                metadata={"pk": hit.id, "retriever": "hybrid"}
            )
            for hits in results for hit in hits
        ]

        return documents
