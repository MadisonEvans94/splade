# agents/tool_calling_agent.py

from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage, AIMessage
from typing import List
import logging
from .base_agent import Agent
from .tools import TOOLS  # Import the tools we've defined


class ToolCallingAgent(Agent):
    def __init__(self, OPENAI_API_KEY: str):
        self.llm = OpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="text-davinci-003",
            temperature=0,
        )

        # Initialize the agent with tools
        self.agent = initialize_agent(
            tools=TOOLS,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, messages: List[HumanMessage]) -> AIMessage:
        logging.info("Executing ToolCallingAgent with tools.")

        # Combine messages into a single prompt
        input_text = "\n".join([msg.content for msg in messages])

        # Run the agent with the input text
        response = self.agent.run(input_text)

        logging.info("ToolCallingAgent response completed.")

        # Return the response as an AIMessage
        return AIMessage(content=response)
