import logging
import json
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agents.base_agent import Agent
from agents.tools.tool_registry import ToolRegistry

# Initialize a logger specific to this module
logger = logging.getLogger(__name__)

# Avoid multiple handler additions
if not logger.hasHandlers():
    # Custom formatter for color and indentation
    class CustomFormatter(logging.Formatter):
        GREEN = "\033[92m"
        RESET = "\033[0m"

        def format(self, record):
            if "Tool Call" in record.msg:
                record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
            if isinstance(record.args, dict):
                # Pretty-print JSON arguments
                record.msg += "\n" + json.dumps(record.args, indent=4)
                record.args = ()  # Clear args to avoid default string formatting
            return super().format(record)

    handler = logging.StreamHandler()
    formatter = CustomFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class WebSearchAgent(Agent):
    """
    LangGraph-based WebSearch agent implementation with MemorySaver for persistence.
    """

    def __init__(self, llm, memory):
        """
        Initialize the WebSearch agent using LangGraph.

        :param llm: The language model.
        :param memory: Persistent memory for managing conversation history.
        """
        logger.info("Initializing WebSearchAgent...")

        # Retrieve the tavily_search tool from the registry
        web_search_tool = ToolRegistry.get_tool('tavily_search', max_results=1)

        self.tools = [web_search_tool]
        self.llm = llm
        self.memory = memory

        # Create a prebuilt React agent with tools and memory
        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,  # Add persistence
        )

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response.

        :param message: User's input message.
        :return: AIMessage response.
        """
        try:
            thread_id = "default"  # You can customize this for multi-user support
            config = {"configurable": {"thread_id": thread_id}}

            # Stream responses from the agent
            for event in self.agent.stream(
                {"messages": [message]},
                config,
                stream_mode="values",
            ):
                # Check if the event contains tool_calls
                if "tool_calls" in event.get("messages", [])[-1].additional_kwargs:
                    tool_calls = event["messages"][-1].additional_kwargs["tool_calls"]
                    for tool_call in tool_calls:
                        logger.info(
                            f"Tool Call: {tool_call['function']['name']}\n")
                        # logger.info("Function Call JSON:\n%s",
                        #             json.dumps(tool_call, indent=4))

                response = event["messages"][-1]
            return response
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
