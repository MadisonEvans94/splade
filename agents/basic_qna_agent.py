# simple_llm_agent.py

import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from .base_agent import Agent


class SimpleLLMAgent(Agent):
    def __init__(self, OPENAI_API_KEY):
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def build_graph(self) -> StateGraph:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def llm_node(state: State):
            logging.info("Executing simple LLM agent.")
            llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY,
                             model="gpt-3.5-turbo")
            messages = state["messages"]
            if not all(isinstance(msg, HumanMessage) for msg in messages):
                messages = [HumanMessage(content=str(msg)) for msg in messages]
            response = llm.invoke(messages)
            logging.info("LLM response completed.")
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response.content)
            return {"messages": [response]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("llm", llm_node)
        graph_builder.add_edge(START, "llm")
        graph_builder.add_edge("llm", END)
        return graph_builder.compile()
