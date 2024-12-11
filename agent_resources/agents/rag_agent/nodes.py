from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from typing import Annotated, TypedDict
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def retrieve_node(state: State, config: dict) -> State:
    """
    Node function to retrieve documents based on the user's last query.
    Assumes 'retriever_tool' is provided in config.
    """
    # Extract the last user message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if not user_message:
        # No user message, just return state
        return state

    query = user_message.content
    retriever_tool = config.get("retriever_tool")
    if retriever_tool is None:
        # If we don't have a retriever_tool, no docs retrieved
        return state

    docs = retriever_tool.run(query)
    retrieved_text = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    # We can append these as an AIMessage for demonstration.
    # In practice, you might want a more structured way of passing retrieved docs to the LLM.
    state["messages"].append(
        AIMessage(content=f"Relevant documents:\n{retrieved_text}")
    )
    return state


def llm_node(state: State, config: dict) -> State:
    """
    Node function to generate a final answer using the LLM.
    Assumes 'llm' is provided in config.
    """
    llm = config.get("llm")
    if llm is None:
        # If we don't have an LLM, just return state
        return state

    # Construct the prompt from the conversation history
    full_context = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            full_context += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            full_context += f"AI: {msg.content}\n"

    response = llm(full_context)
    state["messages"].append(AIMessage(content=response))
    return state
