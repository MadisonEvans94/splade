import os
import uuid
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # Import PromptTemplate
from constants import CONNECTION_ARGS, COLLECTION_NAME
from langchain_core.documents.base import Document

# Set hyperparameters
TOP_K = 5  # Number of top documents to retrieve

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def connect_to_milvus():
    try:
        connections.connect(**CONNECTION_ARGS)
        logging.info("Connected to Milvus successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Milvus: {e}")


def load_documents(source_dir="./SOURCE_DOCUMENTS"):
    documents = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                doc_id = str(uuid.uuid4())
                doc = Document(page_content=content, metadata={
                               "filename": filename, "id": doc_id})
                documents.append(doc)
    return documents


def setup_custom_rag_chain():
    connect_to_milvus()

    try:
        collection = Collection(COLLECTION_NAME)
    except Exception as e:
        logging.error(f"Error initializing collection: {e}")
        return None

    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    except Exception as e:
        logging.error(f"Error initializing OpenAI embeddings: {e}")
        return None

    documents = load_documents()

    try:
        vector_store = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            connection_args=CONNECTION_ARGS
        )
    except Exception as e:
        logging.error(f"Error setting up Milvus vector store: {e}")
        return None

    try:
            # Use GPT-4 model
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)


    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI LLM: {e}")
        return None

    try:
        # Configure retriever to always return top k closest documents
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": TOP_K  # Ensure that TOP_K is set to the number of documents you want
            }
        )
    except Exception as e:
        logging.error(f"Error configuring retriever: {e}")
        return None

    # Define the prompt template to instruct the model
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
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            # Apply prompt template
            chain_type_kwargs={"prompt": prompt_template}
        )
    except Exception as e:
        logging.error(f"Error creating custom RAG chain: {e}")
        return None

    return custom_chain


def main():
    logging.info("Starting the chatbot...")
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    rag_chain = setup_custom_rag_chain()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            response = rag_chain.invoke({"query": user_input})
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue

        # Check if there are source documents retrieved
        if response and 'source_documents' in response and response['source_documents']:
            print(f"\n\nBot: \n{response['result']}\n")

            print("\nRetrieved Documents:\n")
            count = 0
            for doc in response['source_documents']:
                count += 1
                # Retrieve filename from metadata
                filename = doc.metadata.get("filename", "Unknown")
                print(f"{count}. {filename}")
            print("\n--------------------------\n")
        else:
            # No relevant documents found
            print("Bot: I don't know. No relevant documents were retrieved.")
            logging.warning("No source documents retrieved.")


if __name__ == "__main__":
    main()
