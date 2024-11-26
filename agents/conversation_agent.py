from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import List, Optional
import logging
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable
from langchain.schema import BaseMessage


class LLMRunnable(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input: List[BaseMessage], config: Optional[dict] = None, **kwargs) -> BaseMessage:
        response = self.llm(input)
        return response


def get_session_history() -> BaseChatMessageHistory:
    return ConversationBufferMemory().chat_memory


class ConversationAgent:
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo"
        )

        self.conversation = RunnableWithMessageHistory(
            runnable=LLMRunnable(self.llm),
            get_session_history=get_session_history,
            verbose=True
        )

    def run(self, messages: List[HumanMessage]) -> AIMessage:
        logging.info(
            "Executing ConversationAgent with RunnableWithMessageHistory.")

        # Invoke the conversation with the input messages
        response = self.conversation.invoke(messages)

        logging.info("LLM response completed.")

        # Return the final response as an AIMessage
        return AIMessage(content=response.content)
