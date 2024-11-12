# agents.py

import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from pymilvus import connections, Collection
import os
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from constants import CONNECTION_ARGS, COLLECTION_NAME
from retrievers import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Instantiate the collection
collection = Collection(COLLECTION_NAME)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the dense embedding function
dense_embedding_func = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

# Function to get corpus from the collection


def get_corpus(collection: Collection):
    results = collection.query(expr="pk != ''", output_fields=["text"])
    corpus = [doc["text"] for doc in results]
    return corpus


# Get the corpus
corpus = get_corpus(collection)

# Initialize the sparse embedding function
sparse_embedding_func = BM25SparseEmbedding(corpus)

# Define field names
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"

# Define the State for the agent


class State(TypedDict):
    messages: Annotated[list, add_messages]
    use_rag: bool

# Function to check if RAG should be used


def check_for_rag(state: State):
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(
        last_message, HumanMessage) else last_message
    use_rag = "cpu" in content.lower()
    if use_rag:
        logging.info("RAG is triggered based on the query.")
    else:
        logging.info("Normal LLM invocation will be used.")
    return {"use_rag": use_rag}

# LLM node for direct LLM invocation

# RAG retrieval node


def rag_node(state: State):
    logging.info("Executing RAG node with document retrieval.")
    retriever = HybridRetriever(
        collection=collection,
        dense_field=dense_field,
        sparse_field=sparse_field,
        top_k=3,
        embeddings_model=dense_embedding_func,
        sparse_embeddings_model=sparse_embedding_func,
    )
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(
        last_message, HumanMessage) else str(last_message)
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)
    # Retrieve documents using retrieve method
    docs = retriever.retrieve(query)
    logging.info(f"Retrieved {len(docs)} documents for RAG processing.")
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])
    # Create a prompt with the retrieved context
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    response = llm.invoke([HumanMessage(content=prompt)])
    logging.info("RAG response completed.")
    if not isinstance(response, AIMessage):
        response = AIMessage(content=response.content)
    return {"messages": [response]}

# LLM node for direct LLM invocation


def llm_node(state: State):
    logging.info("Executing LLM node for direct response.")
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    messages = state["messages"]
    # Ensure messages are properly formatted as a list of HumanMessage instances
    if not all(isinstance(msg, HumanMessage) for msg in messages):
        messages = [HumanMessage(content=str(msg)) for msg in messages]
    response = llm.invoke(messages)
    logging.info("LLM response completed.")
    if not isinstance(response, AIMessage):
        response = AIMessage(content=response.content)
    return {"messages": [response]}



# Create the graph


def build_agent_graph():
    graph_builder = StateGraph(State)

    # Add nodes for checking, LLM, and RAG processing
    graph_builder.add_node("check_rag", check_for_rag)
    graph_builder.add_node("llm", llm_node)
    graph_builder.add_node("rag", rag_node)

    # Conditional routing based on the RAG flag
    def route_to_rag_or_llm(state: State):
        return "rag" if state["use_rag"] else "llm"

    # Conditional edge from check node to either RAG or LLM
    graph_builder.add_conditional_edges(
        "check_rag",
        route_to_rag_or_llm,
        {"rag": "rag", "llm": "llm"}
    )

    # Direct edges to start and end points
    graph_builder.add_edge(START, "check_rag")
    graph_builder.add_edge("llm", END)
    graph_builder.add_edge("rag", END)

    # Compile and return the graph
    return graph_builder.compile()
