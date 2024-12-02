from abc import ABC, abstractmethod
from typing import List
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory


class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    @abstractmethod
    def run(self, message: HumanMessage) -> AIMessage:
        """
        Abstract method that all agents must implement.
        Takes a HumanMessage as input and returns an AIMessage as the response.
        """
        pass

    def get_session_history(self):
        """
        Optional method to return chat session history.
        Can be overridden by agents requiring history tracking.
        """
        return None
