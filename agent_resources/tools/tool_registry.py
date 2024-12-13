import os
from typing import Dict, Type, List
from langchain.tools import BaseTool
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from .retrieve_documents import RetrieveDocuments  


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If this file is in agent_resources/tools/tool_registry.py,
# and you need to go up directories, you can do:
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "rag_agent"))

PERSIST_DIR = os.path.join(BASE_DIR, "../../chroma_langchain_db")
PERSIST_DIR = os.path.abspath(PERSIST_DIR)
print(f"PERSIST_DIR: {PERSIST_DIR}")
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

class ToolRegistry:
    """
    ToolRegistry manages the registration and retrieval of tools.
    """

    tool_registry: Dict[str, BaseTool] = {
        'tavily_search': TavilySearchResults(),
        'retrieve_documents': RetrieveDocuments(
            embeddings=embeddings,
            vector_store=vector_store
        )
    }

    @classmethod
    def get_tool(cls, tool_name: str, **kwargs) -> BaseTool:
        """
        Retrieve a single tool by name.
        """
        tool = cls.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        # If additional kwargs are provided, they are used to reinitialize the tool
        return tool if not kwargs else tool.__class__(**kwargs)

    @classmethod
    def get_tools(cls, tool_names: List[str], **kwargs) -> List[BaseTool]:
        """
        Retrieve multiple tools by their names.
        """
        return [cls.get_tool(name, **kwargs) for name in tool_names]
