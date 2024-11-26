from langchain_core.runnables import Runnable
from langchain.schema import BaseMessage
from typing import List, Optional


class LLMRunnable(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input: List[BaseMessage], config: Optional[dict] = None, **kwargs) -> BaseMessage:
        response = self.llm(input)
        return response
