# agent_graph_plot_generator.py

import os
import logging
from typing import Type, Dict
from agents.agent_factory import AgentFactory
from agents.base_agent import Agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_core.runnables.graph import MermaidDrawMethod
# Import necessary modules for visualization
from langgraph.graph import Graph  # Adjust based on actual methods
from langgraph.graph.state import StateGraph

# Configure logging


class CustomFormatter(logging.Formatter):
    """Custom logging formatter to enhance readability."""

    def format(self, record):
        if record.levelno == logging.DEBUG:
            format_string = "[DEBUG] {asctime} - {name} - {message}"
        elif record.levelno == logging.INFO:
            format_string = "[INFO] {asctime} - {name} - {message}"
        elif record.levelno == logging.WARNING:
            format_string = "[WARNING] {asctime} - {name} - {message}"
        elif record.levelno == logging.ERROR:
            format_string = "[ERROR] {asctime} - {name} - {message}"
        else:
            format_string = "{asctime} - {levelname} - {name} - {message}"

        formatter = logging.Formatter(
            format_string, style='{', datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Create console handler with the custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

# Optionally, add a file handler to persist logs
file_handler = logging.FileHandler("agent_graph_plot_generator.log")
file_handler.setFormatter(CustomFormatter())
logger.addHandler(file_handler)


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
                logger.info(f"User selected agent: {selected_agent}")
                return selected_agent
            else:
                print(
                    f"Please enter a number between 1 and {len(agent_types)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def generate_and_save_plot(agent: Agent, agent_type: str, output_dir: str = "agents/visualizations"):
    """
    Generate the graph visualization for the given agent and save it as a PNG image.
    
    :param agent: Instance of the agent.
    :param agent_type: Type/name of the agent.
    :param output_dir: Directory where the plot image will be saved.
    """
    logger.info(f"Generating graph plot for agent: {agent_type}")

    # Ensure the visualization directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Ensured that visualization directory exists: {output_dir}")

    # Access the compiled workflow from the agent
    if hasattr(agent, 'app'):
        compiled_workflow = agent.app
        logger.debug(
            "Retrieved compiled workflow from the agent's 'app' attribute.")
    else:
        logger.error(
            "The agent does not have an 'app' attribute containing the compiled workflow.")
        raise AttributeError("Invalid agent workflow.")

    # Retrieve the graph from the compiled workflow
    try:
        graph = compiled_workflow.get_graph()
        logger.debug("Retrieved graph from the compiled workflow.")
    except AttributeError:
        logger.error(
            "The compiled workflow does not have a 'get_graph' method.")
        raise

    # Generate Mermaid diagram
    try:
        mermaid_diagram = graph.draw_mermaid()
        logger.debug("Generated Mermaid diagram from the agent's workflow.")
    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}", exc_info=True)
        raise

    # Save Mermaid diagram as a .mmd file
    mermaid_file_path = os.path.join(output_dir, f"{agent_type}_diagram.mmd")
    try:
        with open(mermaid_file_path, 'w') as mermaid_file:
            mermaid_file.write(mermaid_diagram)
        logger.info(f"Saved Mermaid diagram to {mermaid_file_path}")
    except Exception as e:
        logger.error(f"Failed to save Mermaid diagram: {e}", exc_info=True)
        raise

    # Convert Mermaid diagram to PNG using Mermaid.Ink API
    try:
        # Use the proper enumerated constant for the draw method
        mermaid_png = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        image_path = os.path.join(output_dir, f"{agent_type}_plot.png")
        with open(image_path, 'wb') as img_file:
            img_file.write(mermaid_png)
        logger.info(f"Saved graph visualization as PNG to {image_path}")
        print(f"Graph visualization saved to: {image_path}")
    except Exception as e:
        logger.error(
            f"Failed to generate PNG from Mermaid diagram: {e}", exc_info=True)
        raise


def main():
    """
    Main function to handle the visualization process.
    """
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Initialize LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    logger.info("Initialized ChatOpenAI LLM.")

    # Initialize shared memory
    shared_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    logger.info("Initialized shared conversation memory.")

    # Initialize AgentFactory with shared dependencies
    agent_factory = AgentFactory(llm=llm, memory=shared_memory)
    logger.info("Initialized AgentFactory with shared dependencies.")

    # Retrieve available agents
    available_agents = get_available_agents(agent_factory)
    if not available_agents:
        logger.error("No agents available in the AgentFactory.")
        print("No agents available to visualize.")
        return

    # Prompt user to select an agent
    selected_agent_type = prompt_user_for_agent(available_agents)

    # Instantiate the selected agent
    try:
        agent = agent_factory.factory(selected_agent_type)
        logger.info(f"Instantiated agent: {selected_agent_type}")
    except Exception as e:
        logger.error(
            f"Failed to instantiate agent '{selected_agent_type}': {e}", exc_info=True)
        print(
            f"Error: Could not instantiate agent '{selected_agent_type}'. Check logs for details.")
        return

    # Generate and save the graph plot
    try:
        generate_and_save_plot(agent, selected_agent_type)
    except Exception as e:
        logger.error(
            f"An error occurred while generating the graph plot: {e}", exc_info=True)
        print("An error occurred while generating the graph plot. Check logs for details.")


if __name__ == "__main__":
    main()
