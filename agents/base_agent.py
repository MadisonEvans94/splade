from abc import ABC, abstractmethod
from typing import List
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory


class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    @abstractmethod
    def run(self, messages: List[HumanMessage]) -> AIMessage:
        """
        Abstract method that all agents must implement.
        """
        pass

    def get_session_history(self) -> BaseChatMessageHistory:
        """
        Method to return chat message history.
        Can be overridden by agents that require session history.
        """
        return None
