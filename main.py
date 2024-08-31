import os
import uuid
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import CONNECTION_ARGS, COLLECTION_NAME
from langchain_core.documents.base import Document
import pickle

# Load the BM25 model
with open('bm25_model.pkl', 'rb') as f:
    bm25_model = pickle.load(f)

# Set hyperparameters
TOP_K = 5

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def connect_to_milvus():
    try:
        connections.connect(**CONNECTION_ARGS)
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Milvus: {e}")


def setup_custom_rag_chain():
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
            openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
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
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI LLM: {e}")
        return None

    try:
        dense_retriever = dense_vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K})
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


def main():
    logging.info("Starting the chatbot...")
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    rag_chain, sparse_collection = setup_custom_rag_chain()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")

        # Use the BM25 model to encode the user input
        try:
            sparse_results = sparse_collection.search(
                # Encode the query with BM25
                data=bm25_model.encode_queries([user_input]),
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=TOP_K
            )
        except Exception as e:
            logging.error(f"Error performing sparse retrieval: {e}")
            sparse_results = None

        try:
            response = rag_chain.invoke({"query": user_input})
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue

        if response and 'source_documents' in response and response['source_documents']:
            print(f"\n\nBot: \n{response['result']}\n")
            print("\nRetrieved Documents:\n")
            count = 0
            for doc in response['source_documents']:
                count += 1
                filename = doc.metadata.get("filename", "Unknown")
                print(f"{count}. {filename}")
            print("\n--------------------------\n")
        else:
            print("Bot: I don't know. No relevant documents were retrieved.")
            logging.warning("No source documents retrieved.")

        if sparse_results:
            print("\nSparse Retrieval Results:")
            for res in sparse_results:
                print(res)


if __name__ == "__main__":
    main()
