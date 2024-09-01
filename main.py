import os
import logging
from dotenv import load_dotenv
import pickle
from retrievers import setup_custom_rag_chain

# Constants
BM25_MODEL_PATH = 'bm25_model.pkl'
TOP_K = 5
EXIT_COMMAND = 'exit'

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_bm25_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def chatbot_loop(rag_chain):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")

        try:
            response = rag_chain.invoke({"query": user_input})
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue

        if response and 'source_documents' in response and response['source_documents']:
            print(f"\n\nBot: \n{response['result']}\n")
            print("\nRetrieved Documents:\n")
            for count, doc in enumerate(response['source_documents'], start=1):
                filename = doc.metadata.get("filename", "Unknown")
                print(f"{count}. {filename}")
            print("\n--------------------------\n")
        else:
            print("Bot: I don't know. No relevant documents were retrieved.")
            logging.warning("No source documents retrieved.")


def main():
    logging.info("Starting the chatbot...")

    output = setup_custom_rag_chain(OPENAI_API_KEY, TOP_K)
    logging.info(f"Output: {output}")

    # if rag_chain:
    #     chatbot_loop(rag_chain)
    # else:
    #     logging.error("Failed to set up the RAG chain.")


if __name__ == "__main__":
    main()
