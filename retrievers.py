from pydantic import Field
import os
import pickle
import logging
from typing import Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding, BaseSparseEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from pymilvus import (
    Collection,
    WeightedRanker,
    connections,
)
from langchain_core.runnables.base import RunnableSerializable
from constants import COLLECTION_NAME, CONNECTION_ARGS
import click

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pymilvus import Collection

TOP_K = 2
EXIT_COMMAND = 'exit'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embeddings
with open('bm25_embeddings.pkl', 'rb') as f:
    sparse_embeddings: BM25SparseEmbedding = pickle.load(f)

dense_embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Define fields and collection
pk_field = "pk"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"
collection = Collection(COLLECTION_NAME)

# Define search parameters
sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}

# Function to set up the retriever and chains


class CustomHybridRetriever(BaseRetriever):
    """Custom retriever to retrieve documents using both dense and sparse embeddings."""

    collection: Collection = Field(...)
    dense_field: str = Field(...)
    sparse_field: str = Field(...)
    top_k: int = Field(...)
    embeddings_model: Embeddings = Field(...)
    sparse_embeddings_model: BaseSparseEmbedding = Field(...)

    def __init__(self, collection: Collection, dense_field: str, sparse_field: str, top_k: int, embeddings_model: Embeddings, sparse_embeddings_model: BaseSparseEmbedding):
        # Properly initialize fields using pydantic's BaseModel mechanism
        super().__init__(collection=collection, dense_field=dense_field,
                         top_k=top_k, embeddings_model=embeddings_model)
        self.collection = collection
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.top_k = top_k
        self.embeddings_model = embeddings_model
        self.sparse_embeddings_model = sparse_embeddings_model

    def _retrieve_dense_documents(self, query: str) -> List[Document]:
        """Retrieve documents using dense embeddings."""
        # Convert the query into dense embeddings
        dense_query_embedding = self.embeddings_model.embed_query(query)

        # Define search parameters for dense retrieval
        dense_search_params = {"metric_type": "IP", "params": {}}

        # Perform search using dense embeddings
        dense_results = self.collection.search(
            data=[dense_query_embedding],
            anns_field=self.dense_field,
            param=dense_search_params,
            limit=self.top_k,
            expr=None,
            output_fields=["pk", "text"]
        )

        # Extract the documents from the dense search results
        dense_documents = [
            Document(
                page_content=hit.get("text"),
                metadata={"pk": hit.id, "retriever": "dense"}
            )
            for hits in dense_results for hit in hits
        ]

        return dense_documents

    def _retrieve_sparse_documents(self, query: str) -> List[Document]:
        """Retrieve documents using sparse embeddings."""
        # Convert the query into sparse embeddings
        sparse_query_embedding = self.sparse_embeddings_model.embed_query(
            query)

        # Define search parameters for sparse retrieval
        sparse_search_params = {"metric_type": "IP"}

        # Perform search using sparse embeddings
        sparse_results = self.collection.search(
            data=[sparse_query_embedding],
            anns_field=self.sparse_field,
            param=sparse_search_params,
            limit=self.top_k,
            expr=None,
            output_fields=["pk", "text"]
        )

        # Extract the documents from the sparse search results
        sparse_documents = [
            Document(
                page_content=hit.get("text"),
                metadata={"pk": hit.id, "retriever": "sparse"}
            )
            for hits in sparse_results for hit in hits
        ]

        return sparse_documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve documents using both dense and sparse embeddings."""

        # Retrieve documents using dense and sparse retrieval methods
        dense_documents = self._retrieve_dense_documents(query)
        sparse_documents = self._retrieve_sparse_documents(query)

        # Combine dense and sparse documents
        documents = dense_documents + sparse_documents

        return documents


def setup_chain(hybrid: bool):
    if hybrid:
        # Hybrid search using both dense and sparse embeddings
        retriever = MilvusCollectionHybridSearchRetriever(
            collection=collection,
            anns_fields=[dense_field, sparse_field],
            field_embeddings=[dense_embeddings, sparse_embeddings],
            field_search_params=[dense_search_params, sparse_search_params],
            rerank=WeightedRanker(0.5, 0.5),
            text_field=text_field,
            top_k=TOP_K
        )
        logging.info("Running in hybrid retrieval mode.")
    else:
        # Dense-only search using custom retriever
        retriever = CustomHybridRetriever(
            collection=collection,
            dense_field=dense_field,
            sparse_field=sparse_field,
            top_k=TOP_K,
            embeddings_model=dense_embeddings,
            sparse_embeddings_model=sparse_embeddings
        )
        logging.info("Running in dense-only retrieval mode.")

    # Define the chain using the configured retriever
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: {"context": format_docs(x["context"]), "sources": x["context"]}))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Adjust the final output to include source document information
    return RunnableParallel(
        {
            "context": retriever.invoke,
            "question": RunnablePassthrough()
        }
    ).assign(
        answer=rag_chain_from_docs,
        # Include source documents in the output
        sources=(lambda x: x["context"])
    )
    
# Define prompt template
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provide answers to questions by using fact-based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["context", "question"])

# Utility function to format documents




# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the chatbot loop to include sources in the response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_sources(sources):
    """Formats the source information for output."""
    formatted_sources = []
    for i, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_info = f"Source {i}:"
        source_info += f"\n- Document ID: {metadata.get('pk', 'Unknown')}"
        source_info += "----------------------------------"
        source_info += f"\n\n{doc.page_content[:200]}...\n\n"
        source_info += "----------------------------------"
        source_info += f"\n- Retrieved by: {metadata.get('retriever', 'Unknown')}"
        formatted_sources.append(source_info)
    return "\n\n".join(formatted_sources)


def chatbot_loop(rag_chain: RunnableSerializable[Any, str]):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            response = rag_chain.invoke(user_input)
            print(f"\n\nBot: \n{response['answer']}\n")

            # Print the sources
            if 'sources' in response:
                formatted_sources = format_sources(response['sources'])
                print(f"Sources:\n{formatted_sources}\n")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue

@click.command()
@click.option('--hybrid', is_flag=True, help="Enable hybrid retrieval mode.")
def main(hybrid):
    if hybrid: 
        logging.info("Running in hybrid retrieval mode.")
    else:
        logging.info("Running in dense-only retrieval mode.")
    rag_chain = setup_chain(hybrid)
    try:
        chatbot_loop(rag_chain)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()

