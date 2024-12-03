import logging
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from agents.base_agent import Agent
from agents.nodes.classification_node import classification_node, entity_extraction_node, summarization_node

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

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
        workflow.add_node("classification_node", classification_node)
        workflow.add_node("entity_extraction", entity_extraction_node)
        workflow.add_node("summarization", summarization_node)

        # Add edges to the graph
        workflow.set_entry_point("classification_node")
        workflow.add_edge("classification_node", "entity_extraction")
        workflow.add_edge("entity_extraction", "summarization")
        workflow.add_edge("summarization", END)
        
        # compile into agent runnable
        agent = workflow.compile()
        
        return agent
        
    def run(self, message: HumanMessage) -> AIMessage:
        try:
            state_input = {"text": message.content}
            result = self.agent.invoke(state_input)
            return AIMessage(content=result['classification'])
        except Exception as e:
            logger.error("Error generating response", exc_info=True)
            return AIMessage(content="Sorry, I encountered an error while processing your request.")
