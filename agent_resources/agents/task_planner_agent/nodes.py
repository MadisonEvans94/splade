
from typing import List, TypedDict
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    text: str
    needs_splitting: bool
    tasks: List[str]
    
def determine_split_node(state: State, llm) -> State:
    """
    Determines whether the task needs to be split into subtasks.
    """
    prompt = f"Does the following task need to be split into smaller subtasks? Answer 'Yes' or 'No'.\n\nTask: {state['text']}\n\nAnswer:"
    message = HumanMessage(content=prompt)
    response = llm.invoke([message]).content.strip().lower()
    state['needs_splitting'] = 'yes' in response
    return state


def split_task_node(state: State, llm) -> State:
    """
    Splits the task into subtasks using the LLM.
    """
    prompt = f"Please split the following task into a list of subtasks:\n\n{state['text']}\n\nSubtasks:"
    message = HumanMessage(content=prompt)
    response = llm.invoke([message]).content.strip()
    # Parse the LLM's response into a list
    subtasks = [line.strip('- ').strip()
                for line in response.split('\n') if line.strip()]
    state['tasks'] = subtasks
    return state


def no_split_node(state: State, llm) -> State:
    """
    Handles tasks that do not need splitting by adding the original task to the tasks list.
    """
    state['tasks'] = [state['text']]
    return state


def collect_tasks_node(state: State, llm) -> State:
    """
    Final node that prepares the state for output.
    """
    # Tasks are already in state['tasks']
    return state