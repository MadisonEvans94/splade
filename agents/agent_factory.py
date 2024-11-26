# agents/agent_factory.py

from typing import Dict, Type
from agents.conversation_agent import ConversationAgent
from agents.graph_agent import GraphAgent
from agents.web_search_agent import WebSearchAgent

from agents.base_agent import Agent


class AgentFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agent_registry: Dict[str, Type[Agent]] = {
            'conversation_agent': ConversationAgent,
            'web_search_agent': WebSearchAgent,
            'graph_agent': GraphAgent,  # Register the new agent
        }

    def factory(self, agent_type: str) -> Agent:
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(**self.kwargs)
