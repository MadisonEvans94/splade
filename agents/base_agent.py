from abc import ABC, abstractmethod
from typing import List
from langchain.schema import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory


class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    
    @abstractmethod
    def compile_graph(self):
        """
        method for compiling graph and creating executable agent
        """
        pass

    @abstractmethod
    def run(self, message) -> AIMessage:
        """
        Abstract method that all agents must implement.
        Takes a HumanMessage as input and returns an AIMessage as the response.
        """
        pass

    
