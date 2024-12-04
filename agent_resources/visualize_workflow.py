
# Configure logging
import logging
import os
from typing import Dict, Type
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from agent_factory import AgentFactory
from base_agent import Agent
# from utils import get_available_agents, prompt_user_for_agent


logger = logging.getLogger(__name__)

# Initialize LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


def get_available_agents(agent_factory: AgentFactory) -> Dict[str, Type[Agent]]:
    """
    Retrieve the available agent types from the AgentFactory.
    
    :param agent_factory: Instance of AgentFactory.
    :return: Dictionary of agent types and their corresponding classes.
    """
    return agent_factory.agent_registry


def prompt_user_for_agent(agent_types: Dict[str, Type[Agent]]) -> str:
    """
    Prompt the user to select an agent type from the available options.
    
    :param agent_types: Dictionary of available agent types.
    :return: Selected agent type as a string.
    """
    print("Available Agent Types:")
    for idx, agent_name in enumerate(agent_types.keys(), start=1):
        print(f"{idx}. {agent_name}")

    while True:
        try:
            choice = int(
                input("Enter the number corresponding to the agent you want to visualize: "))
            if 1 <= choice <= len(agent_types):
                selected_agent = list(agent_types.keys())[choice - 1]
                print(f"User selected agent: {selected_agent}")
                return selected_agent
            else:
                print(
                    f"Please enter a number between 1 and {len(agent_types)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


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
        save_directory, f"agents/{selected_agent_type}/{selected_agent_type}_workflow.png")

    # Visualize and save the workflow
    agent.visualize_workflow(save_path=save_path)

if __name__ == "__main__": 
    main()