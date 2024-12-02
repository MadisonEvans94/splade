from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agents.base_agent import Agent


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

        # Create LangGraph nodes
        tool_node = ToolNode(tools=self.tools)
        graph_agent = create_react_agent(self.llm, tools=self.tools)

        # Define state graph
        self.workflow = StateGraph(MessagesState)
        self.workflow.add_node("agent", graph_agent)
        self.workflow.add_node("tools", tool_node)

        # Add edges
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent", lambda state: "tools" if state['messages'][-1].tool_calls else END
        )
        self.workflow.add_edge("tools", "agent")

        # Compile graph with persistence
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response using LangGraph.

        :param message: User's input message.
        :return: AIMessage response.
        """
        inputs = {"messages": [message]}
        final_state = self.app.invoke(
            inputs, config={"configurable": {
                "thread_id": 1}}  # Unique thread ID
        )
        return final_state["messages"][-1]
