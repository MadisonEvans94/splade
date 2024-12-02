from typing import Any
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from pydantic import Field


class RAGTool(BaseTool):
    """
    RAGTool is used for questions requiring knowledge base retrieval.
    """

    # Required field
    chain: Any = Field(..., description="The retrieval chain instance")
    # Required field
    memory: ConversationBufferMemory = Field(...,
                                             description="The conversation buffer memory")

    def __init__(self, chain: Any, memory: ConversationBufferMemory, **kwargs):
        """
        Initialize the RAGTool with the retrieval chain and memory.
        """
        super().__init__(
            name="knowledge_base_search",
            description="Use this tool for questions requiring knowledge base retrieval.",
            chain=chain,
            memory=memory,
            **kwargs,  
        )

    def _run(self, input_text: str) -> str:
        """
        Run the tool with the given input text.

        :param input_text: The user's input text.
        :return: The response from the retrieval chain.
        """

        memory_variables = self.memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])

        response = self.chain.invoke(
            {"input": input_text, "chat_history": chat_history}
        )

        return response.get("answer", "No answer found.")

    async def _arun(self, input_text: str) -> str:
        """
        Asynchronous version of the run method.

        :param input_text: The user's input text.
        :return: The response from the retrieval chain.
        """
        memory_variables = self.memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])

        response = await self.chain.ainvoke(
            {"input": input_text, "chat_history": chat_history}
        )
        return response.get("answer", "No answer found.")
