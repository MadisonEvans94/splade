import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from agents.agent_factory import AgentFactory
from agents.base_agent import Agent
from agents.tools.tool_registry import ToolRegistry
from langchain.memory import ConversationBufferMemory
from utils import RAGChainSetup
from constants import COLLECTION_NAME, CONNECTION_ARGS, EXIT_COMMAND

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

# Initialize shared memory
shared_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize AgentFactory with shared dependencies
agent_factory = AgentFactory(llm=llm, memory=shared_memory)

# Choose agent type ('rag_agent' or 'web_search_agent') and get agent
agent = agent_factory.factory('web_search_agent')
# To instantiate WebSearchAgent, use:
# agent = agent_factory.factory('web_search_agent')


def chatbot_loop(agent: Agent):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            print("Goodbye!")
            break

        try:
            ai_message = agent.run(HumanMessage(content=user_input))
            print(f"\n-----\n\nBot: {ai_message.content}\n")
        except Exception as e:
            logging.error("Error generating response", exc_info=True)


if __name__ == "__main__":
    chatbot_loop(agent)
