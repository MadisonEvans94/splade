from typing import Dict, Type
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .agents.rag_agent.rag_agent import RAGAgent
from .agents.classification_agent.classification_agent import ClassificationAgent
from .agents.task_planner_agent.task_planner_agent import TaskPlannerAgent
from .agents.web_search_agent.web_search_agent import WebSearchAgent
from .base_agent import Agent
class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    """

    def __init__(self, llm: ChatOpenAI, memory: MemorySaver):
        """
        Initialize the factory with shared dependencies.

        :param llm: Language model instance.
        :param memory: Shared persistent memory.
        """
        self.llm = llm
        self.memory = memory

        self.agent_registry: Dict[str, Type[Agent]] = {
            'web_search_agent': WebSearchAgent,
            'classification_agent': ClassificationAgent,
            'task_planner_agent': TaskPlannerAgent,
            
            'rag_agent': RAGAgent
            # etc...
        }

    def factory(self, agent_type: str) -> Agent:
        """
        Create an agent instance.

        :param agent_type: Type of agent to create.
        :return: Initialized agent instance.
        """
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        elif agent_type == "rag_agent":
            return agent_class(llm=self.llm, memory=self.memory, domain_knowledge="Intel, its products, and services")
        return agent_class(llm=self.llm, memory=self.memory)