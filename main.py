# main.py

import os
import logging
from typing import List
from tqdm import tqdm
from agents import AgentFactory
from langchain_openai import OpenAIEmbeddings
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import Collection, connections
from constants import COLLECTION_NAME, CONNECTION_ARGS
from langchain.schema import HumanMessage, AIMessage
EXIT_COMMAND = 'exit'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Instantiate the collection
collection = Collection(COLLECTION_NAME)

# Get corpus


def get_corpus(collection: Collection):
    results = collection.query(expr="pk != ''", output_fields=["text"])
    corpus = [doc["text"] for doc in tqdm(results, desc="fitting bm25 model")]
    return corpus


corpus = get_corpus(collection)

# Initialize embedding functions
sparse_embedding_func = BM25SparseEmbedding(corpus)
dense_embedding_func = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

# Define field names
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"

# Instantiate the AgentFactory
agent_factory = AgentFactory(
    collection=collection,
    dense_embedding_func=dense_embedding_func,
    sparse_embedding_func=sparse_embedding_func,
    dense_field=dense_field,
    sparse_field=sparse_field,
    text_field=text_field,
    OPENAI_API_KEY=OPENAI_API_KEY
)

# Build the desired agent
# For a knowledgebase routing agent
graph = agent_factory.create_knowledgebase_routing_agent()


def chatbot_loop():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")
    messages = []  # Initialize conversation history

    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")

        # Append user's message to conversation history
        messages.append(HumanMessage(content=user_input))

        # Run the agent graph with the current conversation history
        try:
            state = {"messages": messages.copy()}
            events = graph.stream(state, stream_mode="values")
            for event in events:
                if "messages" in event:
                    # Get the last AI message
                    ai_message = event["messages"][-1]
                    # Append AI message to conversation history
                    if isinstance(ai_message, AIMessage):
                        messages.append(ai_message)
                        # Print the AI's response
                        print(f"Bot:\n{ai_message.content}\n")
                    else:
                        logging.error(
                            "Received an unexpected message type from the agent.")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue


if __name__ == "__main__":
    chatbot_loop()
