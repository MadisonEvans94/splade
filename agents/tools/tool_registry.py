from typing import Dict, Type, List
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults


class ToolRegistry:
    def __init__(self):
        self.tool_registry: Dict[str, Type[BaseTool]] = {
            'tavily_search': TavilySearchResults,
            # Add other tools here
        }

    def get_tool(self, tool_name: str, **kwargs) -> BaseTool:
        tool_class = self.tool_registry.get(tool_name)
        if tool_class is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_class(**kwargs)

    def get_tools(self, tool_names: List[str], **kwargs) -> List[BaseTool]:
        return [self.get_tool(name, **kwargs) for name in tool_names]
