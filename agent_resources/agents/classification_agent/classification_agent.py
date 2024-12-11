import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage

from agent_resources.base_agent import Agent
from .nodes import State, classification_node, entity_extraction_node, summarization_node

logger = logging.getLogger(__name__)


class ClassificationAgent(Agent):

    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

        # Compile the graph
        self.agent = self.compile_graph()

    def compile_graph(self):
        workflow = StateGraph(State)

        # Add nodes to the graph
        workflow.add_node("start_node", lambda state: state)
        workflow.add_node("classification_node", classification_node)
        workflow.add_node("entity_extraction", entity_extraction_node)
        workflow.add_node("summarization", summarization_node)
        workflow.add_node("combine_node", lambda state: state)

        # Set entry point
        workflow.set_entry_point("start_node")

        # Add edges to create fan-out from start_node
        workflow.add_edge("start_node", "classification_node")
        workflow.add_edge("start_node", "entity_extraction")
        workflow.add_edge("start_node", "summarization")

        # Add edges to combine_node to create fan-in
        workflow.add_edge("classification_node", "combine_node")
        workflow.add_edge("entity_extraction", "combine_node")
        workflow.add_edge("summarization", "combine_node")

        # Add edge to END
        workflow.add_edge("combine_node", END)

        # Compile into agent runnable
        agent = workflow.compile()

        return agent

    def run(self, message: BaseMessage) -> AIMessage:
        logger.info(f"message: {message}")
        try:
            state_input = {"text": message.content}
            logger.info(f"state_input: {state_input}")
            result = self.agent.invoke(state_input)
            logger.info(f"result: {result}")
            return AIMessage(content=result['classification'])
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")