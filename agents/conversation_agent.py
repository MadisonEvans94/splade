from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import List
import logging


class SimpleLLMAgent:
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo"
        )
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def run(self, messages: List[HumanMessage]) -> AIMessage:
        logging.info("Executing simple LLM agent with ConversationChain.")

        # Combine the messages into a single input string
        input_text = "\n".join([msg.content for msg in messages])

        # Get the response from the conversation chain
        response = self.conversation.predict(input=input_text)

        logging.info("LLM response completed.")

        # Wrap the response in an AIMessage
        return AIMessage(content=response)
