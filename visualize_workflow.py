import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agents.task_planner_agent.task_planner_agent import TaskPlannerAgent
from agents.web_search_agent.web_search_agent import WebSearchAgent

# Initialize LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Initialize Memory
memory = MemorySaver()

# Instantiate the TaskPlannerAgent
agent = WebSearchAgent(llm=llm, memory=memory)

# Define the path where you want to save the visualization
save_directory = os.path.dirname(__file__)  # Directory of the current script
save_path = os.path.join(save_directory, "workflow.png")

# Visualize and save the workflow
agent.visualize_workflow(save_path=save_path)
