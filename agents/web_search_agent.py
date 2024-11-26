from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage

from agents.tools.tool_registry import ToolRegistry


class WebSearchAgent:
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                              model="gpt-3.5-turbo-0125")

        tool_registry = ToolRegistry()
        self.tools = tool_registry.get_tools(['tavily_search'], max_results=1)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the Tavily search tool for queries that require external or up-to-date information, "
                    "especially for topics not commonly found in standard encyclopedias. For common knowledge, respond directly without using tools."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True)

    def run(self, input_text: str, chat_history: list = None) -> AIMessage:
        """
        Run the agent with the given input text and optional chat history.
        """
        input_data = {"input": input_text}
        if chat_history:
            input_data["chat_history"] = chat_history

        # Execute the agent
        result = self.executor.invoke(input_data)

        # Extract the output from the result dictionary
        if isinstance(result, dict) and "output" in result:
            response_content = result["output"]
        else:
            response_content = str(result)

        # Wrap the output in an AIMessage
        return AIMessage(content=response_content)
