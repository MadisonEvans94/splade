# knowledgebase_agent.py

import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from .base_agent import Agent
from retrievers import HybridRetriever


class KnowledgebaseRoutingAgent(Agent):
    def __init__(self, collection, dense_embedding_func, sparse_embedding_func,
                 dense_field, sparse_field, text_field, OPENAI_API_KEY):
        self.collection = collection
        self.dense_embedding_func = dense_embedding_func
        self.sparse_embedding_func = sparse_embedding_func
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.text_field = text_field
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def build_graph(self) -> StateGraph:
        # Define the State for the agent
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            use_rag: bool

        # Define the check_for_rag function
        def check_for_rag(state: State):
            last_message = state["messages"][-1]
            content = last_message.content if isinstance(
                last_message, HumanMessage) else last_message

            # Implement semantic routing logic here
            use_rag = "cpu" in content.lower()
            if use_rag:
                logging.info("RAG is triggered based on the query.")
            else:
                logging.info("Normal LLM invocation will be used.")
            return {"use_rag": use_rag}

        # Define the LLM node
        def llm_node(state: State):
            logging.info("Executing LLM node for direct response.")
            llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY,
                             model="gpt-3.5-turbo")
            messages = state["messages"]
            # Ensure messages are properly formatted
            if not all(isinstance(msg, HumanMessage) for msg in messages):
                messages = [HumanMessage(content=str(msg)) for msg in messages]
            response = llm.invoke(messages)
            logging.info("LLM response completed.")
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response.content)
            return {"messages": [response]}

        # Define the RAG node
        def rag_node(state: State):
            logging.info("Executing RAG node with document retrieval.")
            retriever = HybridRetriever(
                collection=self.collection,
                dense_field=self.dense_field,
                sparse_field=self.sparse_field,
                top_k=3,
                embeddings_model=self.dense_embedding_func,
                sparse_embeddings_model=self.sparse_embedding_func,
            )
            last_message = state["messages"][-1]
            query = last_message.content if isinstance(
                last_message, HumanMessage) else str(last_message)
            if not isinstance(query, str):
                query = str(query)

            docs = retriever.retrieve(query)
            logging.info(
                f"Retrieved {len(docs)} documents for RAG processing.")

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

            llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY,
                             model="gpt-3.5-turbo")
            response = llm.invoke([HumanMessage(content=prompt)])

            logging.info("RAG response completed.")
            if not isinstance(response, AIMessage):
                response = AIMessage(content=response.content)
            return {"messages": [response]}

        # Build the graph
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("check_rag", check_for_rag)
        graph_builder.add_node("llm", llm_node)
        graph_builder.add_node("rag", rag_node)

        # Conditional routing
        def route_to_rag_or_llm(state: State):
            return "rag" if state["use_rag"] else "llm"

        graph_builder.add_conditional_edges(
            "check_rag",
            route_to_rag_or_llm,
            {"rag": "rag", "llm": "llm"}
        )

        # Connect the nodes
        graph_builder.add_edge(START, "check_rag")
        graph_builder.add_edge("llm", END)
        graph_builder.add_edge("rag", END)

        # Compile and return the graph
        return graph_builder.compile()
