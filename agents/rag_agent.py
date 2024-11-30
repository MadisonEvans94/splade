# agents/rag_agent.py

from typing import List
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from agents.base_agent import Agent


class RAGAgent(Agent):
    def __init__(self, retrieval_chain, memory: ConversationBufferMemory):
        self.retrieval_chain = retrieval_chain
        self.memory = memory

    def run(self, message: HumanMessage) -> AIMessage:
        # Append the user's message to memory
        self.memory.chat_memory.add_user_message(message.content)

        # Retrieve chat history from memory
        chat_history = self.memory.chat_memory.messages

        # Pass the correct input keys to the retrieval chain
        response = self.retrieval_chain.invoke(
            {"input": message.content, "chat_history": chat_history}
        )

        # Safely extract the answer
        answer_text = response.get("answer", "No answer found.")

        # Append the assistant's response to memory
        self.memory.chat_memory.add_ai_message(answer_text)

        # Return an AIMessage
        return AIMessage(content=answer_text)
