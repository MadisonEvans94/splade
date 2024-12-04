
# Configure logging
import logging
import os
from typing import Dict, Type
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from agent_resources.agent_factory import AgentFactory
from agent_resources.base_agent import Agent
from utils import get_available_agents, prompt_user_for_agent
logger = logging.getLogger(__name__)

# Initialize LLM
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
    
    # Define the path where you want to save the visualization
    save_directory = os.path.dirname(__file__)  # Directory of the current script
    save_path = os.path.join(
        save_directory, f"agent_resources/agents/{selected_agent_type}/{selected_agent_type}_workflow.png")

    # Visualize and save the workflow
    agent.visualize_workflow(save_path=save_path)

if __name__ == "__main__": 
    main()