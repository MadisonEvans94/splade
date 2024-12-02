# from langgraph.graph import END, START, StateGraph, MessagesState
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.prebuilt import create_react_agent, ToolNode
# from langgraph.checkpoint.memory import MemorySaver
# from agents.base_agent import Agent
# from agents.tools.tool_registry import ToolRegistry


# class WebSearchAgent(Agent):
#     """
#     LangGraph-based WebSearch agent implementation.
#     """

#     def __init__(self, llm, memory):
#         """
#         Initialize the WebSearch agent using LangGraph.

#         :param llm: The language model.
#         :param memory: Shared conversation buffer memory.
#         :param tools: List of tools available to the agent.
#         """
#         tavily_search_tool = ToolRegistry.get_tool(
#             'tavily_search', max_results=1)

#         self.memory = memory
#         self.tools = [tavily_search_tool]
#         self.llm = llm

#         # Create LangGraph nodes
#         tool_node = ToolNode(tools=self.tools)
#         graph_agent = create_react_agent(self.llm, tools=self.tools)

#         # Define state graph
#         self.workflow = StateGraph(MessagesState)
#         self.workflow.add_node("agent", graph_agent)
#         self.workflow.add_node("tools", tool_node)

#         # Add edges
#         self.workflow.add_edge(START, "agent")
#         self.workflow.add_conditional_edges(
#             "agent", lambda state: "tools" if state['messages'][-1].tool_calls else END
#         )
#         self.workflow.add_edge("tools", "agent")

#         # Compile graph with persistence
#         self.checkpointer = MemorySaver()
#         self.app = self.workflow.compile(checkpointer=self.checkpointer)

#     def run(self, message: HumanMessage) -> AIMessage:
#         """
#         Process a HumanMessage and return an AIMessage response using LangGraph.

#         :param message: User's input message.
#         :return: AIMessage response.
#         """
#         inputs = {"messages": [message]}
#         final_state = self.app.invoke(
#             inputs, config={"configurable": {
#                 "thread_id": 2}}  # Unique thread ID
#         )
#         return final_state["messages"][-1]

# agents/web_search_agent.py

import logging
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agents.base_agent import Agent
from agents.tools.tool_registry import ToolRegistry

# Initialize a logger specific to this module
logger = logging.getLogger(__name__)


class WebSearchAgent(Agent):
    """
    LangGraph-based WebSearch agent implementation.
    """

    def __init__(self, llm, memory):
        """
        Initialize the WebSearch agent using LangGraph.

        :param llm: The language model.
        :param memory: Shared conversation buffer memory.
        """
        logger.info("Initializing WebSearchAgent...")

        # Retrieve the tavily_search tool from the registry
        tavily_search_tool = ToolRegistry.get_tool(
            'tavily_search', max_results=1)
        logger.debug(f"Retrieved tool: {tavily_search_tool.name}")

        self.memory = memory
        self.tools = [tavily_search_tool]
        self.llm = llm

        # Create LangGraph nodes
        tool_node = ToolNode(tools=self.tools)
        logger.debug("Created ToolNode with tools.")

        graph_agent = create_react_agent(self.llm, tools=self.tools)
        logger.debug("Created React Agent using LangGraph.")

        # Define state graph
        self.workflow = StateGraph(MessagesState)
        self.workflow.add_node("agent", graph_agent)
        self.workflow.add_node("tools", tool_node)
        logger.info("Added 'agent' and 'tools' nodes to the workflow.")

        # Add edges
        self.workflow.add_edge(START, "agent")
        logger.debug("Added edge from START to 'agent'.")

        self.workflow.add_conditional_edges(
            "agent", lambda state: "tools" if state['messages'][-1].tool_calls else END
        )
        logger.debug(
            "Added conditional edge from 'agent' based on tool calls.")

        self.workflow.add_edge("tools", "agent")
        logger.debug("Added edge from 'tools' back to 'agent'.")

        # Compile graph with persistence
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        logger.info("Compiled the workflow with MemorySaver checkpointer.")

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response using LangGraph.

        :param message: User's input message.
        :return: AIMessage response.
        """
        logger.info(f"Received user message: {message.content}")
        inputs = {"messages": [message]}
        logger.debug(f"Invoking LangGraph workflow with inputs: {inputs}")

        try:
            final_state = self.app.invoke(
                inputs, config={"configurable": {
                    "thread_id": 2}}  # Unique thread ID
            )
            logger.debug(f"Workflow invoked. Final state: {final_state}")

            response = final_state["messages"][-1]
            logger.info(f"Generated AI response: {response.content}")
            return response

        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
