import logging
from langchain_core.messages import BaseMessage, AIMessage
from agent_resources.base_agent import Agent
from agent_resources.tools.tool_registry import ToolRegistry
from langgraph.prebuilt import create_react_agent
logger = logging.getLogger(__name__)


class RAGAgent(Agent):

    def __init__(self, llm, memory, domain_knowledge):
        retrieve_documents_tool = ToolRegistry.get_tool('retrieve_documents')
        self.tools = [retrieve_documents_tool]
        self.llm = llm
        self.memory = memory
        # TODO: Make this a settable attribute
        self.domain_knowledge = domain_knowledge
        self.agent = self.compile_graph()
        
        
    def compile_graph(self):
        try:
            if not self.domain_knowledge or not isinstance(self.domain_knowledge, str):
                raise ValueError(
                    "For RAG agents, the 'domain_knowledge' must be a non-empty string representing the subject matter of the knowledge base."
                )

            system_prompt = f"""
            You have the following tool available:

            1. retrieve_documents: Use this tool only for queries directly related to {self.domain_knowledge}.
            Do not invoke this tool for unrelated queries such as weather, general trivia, or personal topics.

            When answering user queries:
            - If the query is directly related to {self.domain_knowledge}, use the retrieve_documents tool to provide an answer.
            - If the query is unrelated to {self.domain_knowledge}, and you do not have a tool available to answer it, respond by stating that you do not have the capabilities to answer the query.
            - Only use the tool when necessary, and otherwise rely on your own knowledge if appropriate.
            """

            # Pass the system prompt as the `state_modifier`
            agent = create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.memory,
                state_modifier=system_prompt,  # Add the system prompt here
            )
            return agent

        except ValueError as ve:
            logger.error(f"Validation error during agent compilation: {ve}")
            raise

        except Exception as e:
            logger.error(
                "Unexpected error during agent compilation", exc_info=True)
            raise



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
