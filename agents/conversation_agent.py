from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from typing import List, Optional
import logging
from langchain_core.chat_history import BaseChatMessageHistory
from .base_agent import Agent  # Import the base Agent class
from .llm_runnable import LLMRunnable  # Import LLMRunnable


class ConversationAgent(Agent):
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo"
        )

        # Initialize ConversationBufferMemory
        self.memory = ConversationBufferMemory()

        self.conversation = RunnableWithMessageHistory(
            runnable=LLMRunnable(self.llm),
            get_session_history=self.get_session_history,
            verbose=True
        )

    def get_session_history(self) -> BaseChatMessageHistory:
        """
        Returns the chat message history for the agent.
        """
        return self.memory.chat_memory

    def run(self, messages: List[HumanMessage]) -> AIMessage:
        logging.info(
            "Executing ConversationAgent with RunnableWithMessageHistory.")

        # Invoke the conversation with the input messages
        response = self.conversation.invoke(messages)

        logging.info("LLM response completed.")

        # Return the final response as an AIMessage
        return AIMessage(content=response.content)
