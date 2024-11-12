from abc import ABC, abstractmethod
from langgraph.graph import StateGraph


class Agent(ABC):
    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass
