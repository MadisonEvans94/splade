# retrievers.py

import logging
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import CONNECTION_ARGS, COLLECTION_NAME


def connect_to_milvus():
    try:
        connections.connect(**CONNECTION_ARGS)
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Milvus: {e}")


def setup_custom_rag_chain(openai_api_key, top_k):
    connect_to_milvus()

    try:
        # Access existing collections
        dense_collection = Collection(COLLECTION_NAME)
        sparse_collection = Collection(f"{COLLECTION_NAME}_SPARSE")
    except Exception as e:
        logging.error(f"Error accessing collections: {e}")
        return None

    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model="text-embedding-ada-002")
    except Exception as e:
        logging.error(f"Error initializing OpenAI embeddings: {e}")
        return None

    # Set up dense vector store using LangChain
    try:
        dense_vector_store = Milvus(
            embedding_function=embeddings,
            connection_args=CONNECTION_ARGS,
            collection_name=COLLECTION_NAME,
            index_params={
                "index_type": "IVF_FLAT",  # or your preferred index type
                "metric_type": "L2",  # distance metric for dense vectors
                "params": {"nlist": 128}
            }
        )

    except Exception as e:
        logging.error(f"Error setting up dense Milvus vector store: {e}")
        return None

    try:
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0)
    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI LLM: {e}")
        return None

    try:
        dense_retriever = dense_vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k})
    except Exception as e:
        logging.error(f"Error configuring dense retriever: {e}")
        return None

    prompt_template = PromptTemplate(
        template="""
        You are a helpful assistant. Answer the question based solely on the provided context. 
        If the context does not provide enough information, respond with "I don't know."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )

    try:
        custom_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=dense_retriever,  # Pass the dense retriever directly, not as a list
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    except Exception as e:
        logging.error(f"Error creating custom RAG chain: {e}")
        return None

    return custom_chain, sparse_collection  # Return sparse collection
