from typing import Dict, Type, List
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from .rag_tool import RAGTool


class ToolRegistry:
    """
    ToolRegistry manages the registration and retrieval of tools.
    """

    tool_registry: Dict[str, Type[BaseTool]] = {
        'tavily_search': TavilySearchResults,
        'rag_tool': RAGTool,
    }

    @classmethod
    def get_tool(cls, tool_name: str, **kwargs) -> BaseTool:
        """
        Retrieve a single tool by name.
        """
        tool_class = cls.tool_registry.get(tool_name)
        if tool_class is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_class(**kwargs)

    @classmethod
    def get_tools(cls, tool_names: List[str], **kwargs) -> List[BaseTool]:
        """
        Retrieve multiple tools by their names.
        """
        return [cls.get_tool(name, **kwargs) for name in tool_names]
