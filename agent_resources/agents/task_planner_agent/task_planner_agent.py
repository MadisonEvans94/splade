import logging
from typing import List, TypedDict
from langgraph.graph import StateGraph
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from agent_resources.base_agent import Agent
from langchain.globals import set_verbose
set_verbose(True)

class State(TypedDict):
    text: str
    tasks: List[str]


logger = logging.getLogger(__name__)


class TaskPlannerAgent(Agent):
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

        # Compile the graph
        self.agent = self.compile_graph()

    def compile_graph(self):
        """
        Compiles the workflow graph for the agent.
        """
        workflow = StateGraph(State)

        # Add the single node
        workflow.add_node("split_tasks_node", self.split_tasks_node)

        # Define start and end nodes
        workflow.set_entry_point("split_tasks_node")
        workflow.set_finish_point("split_tasks_node")

        agent = workflow.compile(debug=False)

        return agent

    def split_tasks_node(self, state: State) -> State:
        """
        Splits the input text into high-level tasks.
        """
        prompt = (
            f"Split the following text into clear and distinct high-level tasks. "
            f"Each task should represent a complete action or goal.\n\n"
            f"Text: {state['text']}\n\nTasks:"
        )
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message]).content.strip()
        tasks = [
            line.strip("- ").strip().lstrip('0123456789.').strip()
            for line in response.split("\n") if line.strip()
        ]
        state["tasks"] = tasks
        return state

    def run(self, message: BaseMessage) -> AIMessage:
        """
        Processes the input message through the workflow and returns the response.
        """
        try:
            state_input = {"text": message.content}
            result_state = self.agent.invoke(state_input)
            tasks = result_state["tasks"]

            response_content = "Identified tasks:\n" + "\n".join(
                [f"- {task}" for task in tasks]
            )
            return AIMessage(content=response_content)

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(
                content="Sorry, I encountered an error while processing your request."
            )