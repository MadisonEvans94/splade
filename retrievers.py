import logging
from pydantic import Field
from typing import Dict, List
from langchain_milvus.utils.sparse import BaseSparseEmbedding, BM25SparseEmbedding
from langchain.embeddings.base import Embeddings
from pymilvus import (
    AnnSearchRequest,
    Collection,
    WeightedRanker,
    RRFRanker
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


class SpladeSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on SPLADE."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str = "cpu"):
        from pymilvus.model.sparse import SpladeEmbeddingFunction  # type: ignore

        # Initialize SPLADE embedding function from Milvus
        self.splade_ef = SpladeEmbeddingFunction(
            model_name=model_name, device=device)

    def embed_query(self, query: str) -> Dict[int, float]:
        # Encode a query into a SPLADE sparse vector
        query_sparse_emb = self.splade_ef([query])
        sparse_dict = self._sparse_to_dict(query_sparse_emb)
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
        super().__init__(collection=collection, dense_field=dense_field,
                         top_k=top_k, embeddings_model=embeddings_model)
        self.collection = collection
        self.dense_field = dense_field
        self.top_k = top_k
        self.embeddings_model = embeddings_model

    def _retrieve_dense_documents(self, query: str) -> List[Document]:
        """Perform standard dense retrieval using Milvus."""
        dense_query_embedding = self.embeddings_model.embed_query(query)
        search_params = {"metric_type": "IP", "params": {}}
        results = self.collection.search(
            data=[dense_query_embedding],
            anns_field=self.dense_field,
            param=search_params,
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

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
        return self._retrieve_dense_documents(query)


class CustomHybridRetriever(BaseRetriever):
    """Custom retriever to retrieve documents using both dense and sparse embeddings."""

    collection: Collection = Field(...)
    dense_field: str = Field(...)
    sparse_field: str = Field(...)
    top_k: int = Field(...)
    embeddings_model: Embeddings = Field(...)
    sparse_embeddings_model: BaseSparseEmbedding = Field(...)
    ratio: List[float] = Field(default_factory=lambda: [0.5, 0.5])

    def _retrieve_dense_request(self, query: str) -> AnnSearchRequest:
        dense_query_embedding = self.embeddings_model.embed_query(query)
        dense_request = AnnSearchRequest(
            data=[dense_query_embedding],
            anns_field=self.dense_field,
            param={"metric_type": "IP", "params": {}},
            limit=self.top_k
        )
        return dense_request

    def _retrieve_sparse_request(self, query: str) -> AnnSearchRequest:
        sparse_dict = self.sparse_embeddings_model.embed_query(query)
        sparse_request = AnnSearchRequest(
            data=[sparse_dict],
            anns_field=self.sparse_field,
            param={"metric_type": "IP"},
            limit=self.top_k
        )
        return sparse_request

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        dense_request = self._retrieve_dense_request(query)
        sparse_request = self._retrieve_sparse_request(query)
        # reranker = WeightedRanker(*self.ratio)
        reranker = RRFRanker(k=60)
        results = self.collection.hybrid_search(
            [dense_request, sparse_request],
            rerank=reranker,
            limit=self.top_k,
            output_fields=["pk", "text"]
        )

        documents = [
            Document(
                page_content=hit.get("text"),
                metadata={"pk": hit.id, "retriever": "hybrid"}
            )
            for hits in results for hit in hits
        ]
        return documents
