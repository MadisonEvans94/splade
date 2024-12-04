import logging
from typing import List, TypedDict, Union
from langgraph.graph import StateGraph
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from agents.base_agent import Agent


class State(TypedDict):
    text: str
    needs_splitting: bool
    tasks: List[str]


logger = logging.getLogger(__name__)


class TaskPlannerAgent(Agent):
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

        # Compile the graph
        self.agent = self.compile_graph()

    def compile_graph(self):
        workflow = StateGraph(State)

        # Add nodes
        determine_split = workflow.add_node(
            "determine_split_node", self.determine_split_node)
        split_task = workflow.add_node("split_task_node", self.split_task_node)
        no_split = workflow.add_node("no_split_node", self.no_split_node)
        collect_tasks = workflow.add_node(
            "collect_tasks_node", self.collect_tasks_node)

        # Add conditional edges
        def determine_next_node(state: State) -> Union[str, List[str]]:
            if state["needs_splitting"]:
                return "split_task_node"
            else:
                return "no_split_node"

        workflow.add_conditional_edges(
            source="determine_split_node",
            path=determine_next_node
        )

        # Both split_task_node and no_split_node lead to collect_tasks_node
        workflow.add_edge("split_task_node", "collect_tasks_node")
        workflow.add_edge("no_split_node", "collect_tasks_node")

        # Define start and end nodes
        workflow.set_entry_point("determine_split_node")
        workflow.set_finish_point("collect_tasks_node")

        # Compile the agent
        agent = workflow.compile(debug=True)

        return agent

    def determine_split_node(self, state: State) -> State:
        """
        Determines whether the task needs to be split into subtasks.
        """
        prompt = (
            f"Does the following task need to be split into smaller subtasks? "
            f"Answer 'Yes' or 'No'.\n\nTask: {state['text']}\n\nAnswer:"
        )
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message]).content.strip().lower()
        state["needs_splitting"] = "yes" in response
        return state

    def split_task_node(self, state: State) -> State:
        """
        Splits the task into subtasks using the LLM.
        """
        prompt = (
            f"Please split the following task into a list of subtasks:\n\n"
            f"{state['text']}\n\nSubtasks:"
        )
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message]).content.strip()
        # Parse the LLM's response into a list
        subtasks = [
            line.strip("- ").strip()
            for line in response.split("\n") if line.strip()
        ]
        state["tasks"] = subtasks
        return state

    def no_split_node(self, state: State) -> State:
        """
        Handles tasks that do not need splitting by adding the original task to the tasks list.
        """
        state["tasks"] = [state["text"]]
        return state

    def collect_tasks_node(self, state: State) -> State:
        """
        Final node that prepares the state for output.
        """
        # Tasks are already in state["tasks"]
        return state

    def run(self, message: BaseMessage) -> AIMessage:
        try:
            state_input = {"text": message.content}
            result_state = self.agent.invoke(state_input)
            tasks = result_state["tasks"]
            # Format the tasks into a string response
            response_content = "\n".join(f"- {task}" for task in tasks)
            return AIMessage(content=response_content)
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(
                content="Sorry, I encountered an error while processing your request."
            )
