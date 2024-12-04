from typing import Dict, Type
from agent_resources.agent_factory import AgentFactory
from agent_resources.base_agent import Agent


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
