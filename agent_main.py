import json
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from agents.tools.rag_tool import RAGTool
from utils import ChainSetup
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

# Initialize the chain setup
chain_setup = ChainSetup(
    collection_name=COLLECTION_NAME,
    connection_args=CONNECTION_ARGS,
    openai_api_key=OPENAI_API_KEY,
    llm=llm
)

# Prepare the RAG chain
retrieval_chain = chain_setup.setup_chain(hybrid=True)

# Create the RAG tool
rag_tool = RAGTool(chain=retrieval_chain, memory=memory)

# Add the tool to the agent
tools = [rag_tool]


def initialize_rag_agent(llm, tools, memory):
    """Function to initialize the RAG agent."""
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )
    return agent_executor


# Initialize the agent
agent_executor = initialize_rag_agent(llm, tools, memory)


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


def chatbot_loop(agent_executor):
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
            response = agent_executor.invoke(input=user_input)

            # Pretty-print the response if it's a dictionary
            if isinstance(response, dict):
                if "chat_history" in response:
                    response["chat_history"] = serialize_chat_history(
                        response["chat_history"])
                logging.info(
                    f"Agent response:\n{json.dumps(response, indent=4)}")
                print(f"\n\nBot: \n{response['output']}\n")
            else:
                logging.info(f"Agent response: {response}")
                print(f"\n\nBot: \n{response}\n")

        except Exception as e:
            logging.error("Error generating response", exc_info=True)
            continue


if __name__ == "__main__":
    chatbot_loop(agent_executor)
