# agents.py

import logging
from typing import Dict, Type
from agents.knowledgebase_routing_agent import KnowledgebaseRoutingAgent
from agents.conversation_agent import ConversationAgent
from agents.base_agent import Agent


class AgentFactory:
    """
    A factory class responsible for creating agent instances based on the provided agent type.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.agent_registry: Dict[str, Type[Agent]] = {
            'conversation_agent': ConversationAgent,
            # Add new agents here
        }

    def factory(self, agent_type: str) -> Agent:
        """Factory method to create agents based on the agent_type string."""
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(**self.kwargs)
