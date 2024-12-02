from abc import ABC, abstractmethod
from typing import List
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory


class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    # @abstractmethod
    # def run(self, message: HumanMessage) -> AIMessage:
    #     """
    #     Abstract method that all agents must implement.
    #     Takes a HumanMessage as input and returns an AIMessage as the response.
    #     """
    #     pass

    # @abstractmethod
    # def get_session_history(self) -> BaseChatMessageHistory:
    #     """
    #     Method to return chat session history.
    #     Must be implemented by all agents that support session history.
    #     """
    #     pass
