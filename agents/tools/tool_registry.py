from typing import Dict, Type, List
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from agents.tools.rag_tool import RAGTool


class ToolRegistry:
    """
    ToolRegistry manages the registration and retrieval of tools.
    """

    def __init__(self):
        self.tool_registry: Dict[str, Type[BaseTool]] = {
            'tavily_search': TavilySearchResults,
            'rag_tool': RAGTool,  # Register the RAGTool
            # Add other tools here as needed
        }

    def get_tool(self, tool_name: str, **kwargs) -> BaseTool:
        """
        Retrieve a single tool by name.

        :param tool_name: The name of the tool to retrieve.
        :param kwargs: Additional keyword arguments for tool initialization.
        :return: An instance of the requested tool.
        """
        tool_class = self.tool_registry.get(tool_name)
        if tool_class is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_class(**kwargs)

    def get_tools(self, tool_names: List[str], **kwargs) -> List[BaseTool]:
        """
        Retrieve multiple tools by their names.

        :param tool_names: A list of tool names to retrieve.
        :param kwargs: Additional keyword arguments for tool initialization.
        :return: A list of tool instances.
        """
        return [self.get_tool(name, **kwargs) for name in tool_names]
