from typing import Dict, Type
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from agents.conversation_agent import ConversationAgent
from agents.graph_agent import GraphAgent
from agents.web_search_agent import WebSearchAgent
from agents.rag_agent import RAGAgent

from agents.base_agent import Agent


class AgentFactory:
    """
    Factory class for creating agents with shared configurations.
    """

    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        """
        Initialize the factory with shared dependencies.

        :param llm: Language model instance.
        :param memory: Shared conversation memory.
        :param tools: List of tools available to agents.
        """
        self.llm = llm
        self.memory = memory

        self.agent_registry: Dict[str, Type[Agent]] = {
            # 'conversation_agent': ConversationAgent,
            'web_search_agent': WebSearchAgent,
            # 'graph_agent': GraphAgent,
            # 'rag_agent': RAGAgent,
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
        return agent_class(llm=self.llm, memory=self.memory)
