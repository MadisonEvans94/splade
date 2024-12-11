import logging
from langchain_core.messages import BaseMessage, AIMessage
from agent_resources.base_agent import Agent
from agent_resources.tools.tool_registry import ToolRegistry
from langgraph.prebuilt import create_react_agent
logger = logging.getLogger(__name__)


class RAGAgent(Agent):

    def __init__(self, llm, memory):
        retrieve_documents_tool = ToolRegistry.get_tool('retrieve_documents')
        self.tools = [retrieve_documents_tool]
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


    def run(self, message: BaseMessage):
        """
        Process a message and return the AI's final response.
        """
        try:
            thread_id = "default"
            # Pass llm and retriever_tool via config so nodes can access them
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }

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
