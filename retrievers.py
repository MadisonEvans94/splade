import logging
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import CONNECTION_ARGS, COLLECTION_NAME
from typing import Optional, Tuple

# Constants
DENSE_COLLECTION_NAME = COLLECTION_NAME
SPARSE_COLLECTION_NAME = f"{COLLECTION_NAME}_SPARSE"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"
INDEX_PARAMS = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based solely on the provided context. 
If the context does not provide enough information, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""


def connect_to_milvus() -> None:
    try:
        connections.connect(**CONNECTION_ARGS)
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Milvus: {e}")


def get_collection(name: str) -> Optional[Collection]:
    try:
        return Collection(name)
    except Exception as e:
        logging.error(f"Error accessing collection {name}: {e}")
        return None


def get_embeddings(api_key: str) -> Optional[OpenAIEmbeddings]:
    try:
        return OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
    except Exception as e:
        logging.error(f"Error initializing OpenAI embeddings: {e}")
        return None


def get_dense_vector_store(embeddings: OpenAIEmbeddings) -> Optional[Milvus]:
    try:
        return Milvus(
            embedding_function=embeddings,
            connection_args=CONNECTION_ARGS,
            collection_name=DENSE_COLLECTION_NAME,
            index_params=INDEX_PARAMS
        )
    except Exception as e:
        logging.error(f"Error setting up dense Milvus vector store: {e}")
        return None


def get_llm(api_key: str) -> Optional[ChatOpenAI]:
    try:
        return ChatOpenAI(api_key=api_key, model=LLM_MODEL, temperature=0)
    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI LLM: {e}")
        return None


def get_dense_retriever(vector_store: Milvus, top_k: int) -> Optional[Milvus]:
    try:
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    except Exception as e:
        logging.error(f"Error configuring dense retriever: {e}")
        return None


def setup_custom_rag_chain(openai_api_key: str, top_k: int) -> Optional[Tuple[RetrievalQA, Collection]]:
    connect_to_milvus()

    dense_collection = get_collection(DENSE_COLLECTION_NAME)
    sparse_collection = get_collection(SPARSE_COLLECTION_NAME)
    if not dense_collection or not sparse_collection:
        return None

    embeddings = get_embeddings(openai_api_key)
    if not embeddings:
        return None

    dense_vector_store = get_dense_vector_store(embeddings)
    if not dense_vector_store:
        return None

    llm = get_llm(openai_api_key)
    if not llm:
        return None

    dense_retriever = get_dense_retriever(dense_vector_store, top_k)
    if not dense_retriever:
        return None

    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    try:
        custom_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=dense_retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    except Exception as e:
        logging.error(f"Error creating custom RAG chain: {e}")
        return None

    return custom_chain, sparse_collection
