import logging
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from agent_resources.tools.tool_registry import ToolRegistry
from agent_resources.base_agent import Agent


# Initialize a logger specific to this module
logger = logging.getLogger(__name__)

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

        web_search_tool = ToolRegistry.get_tool('tavily_search', max_results=1)
        self.tools = [web_search_tool]
        self.llm = llm
        self.memory = memory
        self.agent = self.compile_graph()
        
    def compile_graph(self):
        agent = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,  
        )
        return agent

    def run(self, message: HumanMessage) -> AIMessage:
        """
        Process a HumanMessage and return an AIMessage response.

        :param message: User's input message.
        :return: AIMessage response.
        """

        try:
            thread_id = "default"  
            config = {"configurable": {"thread_id": thread_id}}
            response = self.agent.invoke(
                {"messages": [message]}, config=config)

            ai_message = response["messages"][-1]
            if isinstance(ai_message, AIMessage):
                return ai_message
            else:
                logger.error("Unexpected message type in response.")
                raise ValueError("Expected AIMessage in the response.")
            
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")