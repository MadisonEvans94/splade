import json
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from agents.agent_factory import AgentFactory
from agents.tools.tool_registry import ToolRegistry  
from utils import RAGChainSetup
from constants import (
    COLLECTION_NAME,
    CONNECTION_ARGS,
    EXIT_COMMAND,
)


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
#TODO: make retrieval chain more streamlined 
# Retrieval Chain Object
rag_chain_setup = RAGChainSetup(
    collection_name=COLLECTION_NAME,
    connection_args=CONNECTION_ARGS,
    openai_api_key=OPENAI_API_KEY,
    llm=llm
)

# RAG chain instantiation
retrieval_chain = rag_chain_setup.setup_chain(hybrid=True)

# Initialize ToolRegistry
tool_registry = ToolRegistry()

# Get RAGTool from ToolRegistry with required parameters
rag_tool = tool_registry.get_tool(
    'rag_tool', chain=retrieval_chain, memory=memory)
tools = [rag_tool]

# Initialize the AgentFactory with dependencies
agent_factory = AgentFactory(llm=llm, tools=tools, memory=memory)

# Create the agent using the factory
agent = agent_factory.factory('rag_agent')


def serialize_message(message):
    """Helper function to serialize HumanMessage and AIMessage objects."""
    return {
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
        "response_metadata": getattr(message, "response_metadata", {})
    }


def serialize_chat_history(chat_history):
    """Helper function to serialize chat history containing messages."""
    return [serialize_message(msg) for msg in chat_history]


def chatbot_loop(agent):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            logging.info(f"User input: {user_input}")
            user_message = HumanMessage(content=user_input)
            ai_message = agent.run(user_message)

            logging.info(f"Agent response: {ai_message.content}")
            print(f"\n\nBot: \n{ai_message.content}\n")

        except Exception as e:
            logging.error("Error generating response", exc_info=True)
            continue


if __name__ == "__main__":
    chatbot_loop(agent)
