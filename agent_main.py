import logging
import os
from typing import Dict, Type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from agents.agent_factory import AgentFactory
from agents.base_agent import Agent
from utils import get_available_agents, prompt_user_for_agent
from constants import COLLECTION_NAME, CONNECTION_ARGS, EXIT_COMMAND
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")



def main():

    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

    # Initialize shared memory using LangGraph's MemorySaver for persistence
    shared_memory = MemorySaver()

    # Initialize AgentFactory with shared dependencies
    agent_factory = AgentFactory(llm=llm, memory=shared_memory)

    # Retrieve available agents
    available_agents = get_available_agents(agent_factory)
    if not available_agents:
        logger.error("No agents available in the AgentFactory.")
        print("No agents available to visualize.")
        return
    
    # Prompt user to select an agent
    selected_agent_type = prompt_user_for_agent(available_agents)
    
    try: 
        agent = agent_factory.factory(selected_agent_type)
        logger.info(f"Instantiated agent: {selected_agent_type}")
    except Exception as e:
        logger.error(
            f"Failed to instantiate agent '{selected_agent_type}': {e}", exc_info=True)
        print(
            f"Error: Could not instantiate agent '{selected_agent_type}'. Check logs for details.")
        return
    
    print(f"\n\n-----\n\nWelcome to the Chatbot! \nYou've selected the {selected_agent_type} agent. Type 'exit' to end the conversation.\n")
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
    main()
