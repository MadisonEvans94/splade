# main.py

import logging

from pymilvus import Collection, connections
from agents.agent_factory import AgentFactory
from constants import COLLECTION_NAME, CONNECTION_ARGS
from langchain.schema import HumanMessage, AIMessage
import os
from tqdm import tqdm
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import OpenAIEmbeddings

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
factory_kwargs = {
    'collection': collection,
    'dense_embedding_func': dense_embedding_func,
    'sparse_embedding_func': sparse_embedding_func,
    'dense_field': dense_field,
    'sparse_field': sparse_field,
    'text_field': text_field,
    'OPENAI_API_KEY': OPENAI_API_KEY
}

agent_factory = AgentFactory(**factory_kwargs)

# Build the desired agent using the factory method
agent_type = 'knowledgebase_router'  # or 'simple_llm', etc.
agent = agent_factory.factory(agent_type)
graph = agent.build_graph()


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
            events = list(graph.stream(state, stream_mode="values"))
            if events:
                final_event = events[-1]
                if "messages" in final_event:
                    # Get the last AI message
                    ai_message = final_event["messages"][-1]
                    # Append AI message to conversation history
                    if isinstance(ai_message, AIMessage):
                        messages.append(ai_message)
                        # Print the AI's response
                        print(f"Bot:\n{ai_message.content}\n")
                    else:
                        logging.error(
                            "Received an unexpected message type from the agent.")
                else:
                    logging.error("No messages returned from the agent.")
            else:
                logging.error("No events returned from the agent.")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue


if __name__ == "__main__":
    chatbot_loop()
