from typing import Any
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory


class RAGTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = "Use this tool for questions requiring knowledge base retrieval."
    chain: Any  # The retrieval_chain instance
    memory: ConversationBufferMemory

    def _run(self, input_text: str) -> str:
        # Retrieve chat history from memory
        chat_history = self.memory.chat_memory.messages

        # Pass the correct input keys to the retrieval chain
        response = self.chain.invoke(
            {"input": input_text, "chat_history": chat_history}
        )
        # Safely extract the answer
        return response.get("answer", "No answer found.")

    async def _arun(self, input_text: str) -> str:
        # Asynchronous version
        chat_history = self.memory.chat_memory.messages
        response = await self.chain.ainvoke(
            {"input": input_text, "chat_history": chat_history}
        )
        return response.get("answer", "No answer found.")
