import json
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import initialize_agent, AgentType
from agents.rag_agent import RAGAgent
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

# Initialize the RAG agent
agent = RAGAgent(llm=llm, tools=tools, memory=memory)


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
