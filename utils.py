# utils.py

import os
from typing import List
from pymilvus import connections, Collection
from tqdm import tqdm
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import OpenAIEmbeddings


from constants import (
    DENSE_FIELD, SPARSE_FIELD, TEXT_FIELD, TOP_K,
    DENSE_SEARCH_PARAMS, SPARSE_SEARCH_PARAMS
)
from langchain_milvus import MilvusCollectionHybridSearchRetriever
from retrievers import StandardRetriever
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymilvus import WeightedRanker


def get_collection(collection_name: str, connection_args: dict) -> Collection:
    connections.connect(**connection_args)
    return Collection(collection_name)


def get_corpus(collection: Collection) -> List[str]:
    results = collection.query(expr="pk != ''", output_fields=["text"])
    return [doc["text"] for doc in tqdm(results, desc="Fetching corpus")]


def get_sparse_embedding_func(corpus: List[str]) -> BM25SparseEmbedding:
    return BM25SparseEmbedding(corpus)


def get_dense_embedding_func(openai_api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")


class ChainSetup:
    def __init__(self, collection_name: str, connection_args: dict, openai_api_key: str, llm):
        self.collection = get_collection(collection_name, connection_args)
        self.corpus = get_corpus(self.collection)
        self.sparse_embedding_func = get_sparse_embedding_func(self.corpus)
        self.dense_embedding_func = get_dense_embedding_func(openai_api_key)
        self.llm = llm

    def setup_chain(self, hybrid: bool):
        if hybrid:
            retriever = MilvusCollectionHybridSearchRetriever(
                collection=self.collection,
                rerank=WeightedRanker(0.5, 0.5),
                anns_fields=[DENSE_FIELD, SPARSE_FIELD],
                field_embeddings=[self.dense_embedding_func,
                                  self.sparse_embedding_func],
                field_search_params=[
                    DENSE_SEARCH_PARAMS, SPARSE_SEARCH_PARAMS],
                top_k=TOP_K,
                text_field=TEXT_FIELD,
            )
        else:
            retriever = StandardRetriever(
                collection=self.collection,
                dense_field=DENSE_FIELD,
                top_k=TOP_K,
                embeddings_model=self.dense_embedding_func,
            )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Reformulate the user's query to be independent of prior context."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=contextualize_q_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant for answering questions."),
                MessagesPlaceholder("chat_history"),
                ("human", "{context}\n\nQuestion: {input}\nAnswer:"),
            ]
        )

        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )

        retrieval_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )

        return retrieval_chain
