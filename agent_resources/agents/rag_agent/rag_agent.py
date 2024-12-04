from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agent_resources.base_agent import Agent


class RAGAgent(Agent):
    """
    LangGraph-based RAG agent implementation.
    """

    def __init__(self, llm, memory, tools):
        """
        Initialize the RAG agent using LangGraph.

        :param llm: The language model.
        :param memory: Shared conversation buffer memory.
        :param tools: List of tools available to the agent.
        """
        self.memory = memory
        self.tools = tools
        self.llm = llm

        # TODO

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response using LangGraph.

        :param message: User's input message.
        :return: AIMessage response.
        """
        # TODO
        pass
