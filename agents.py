# agents.py

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import connections, Collection
import os
from langchain_milvus.utils.sparse import BM25SparseEmbedding
# Import constants
from constants import CONNECTION_ARGS, COLLECTION_NAME
# Import from your retrievers.py
from retrievers import HybridRetriever

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
    # Fetch all documents with a query expression matching any valid pk
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
    content = last_message if isinstance(
        last_message, str) else last_message.content
    return {"use_rag": "cpu" in content.lower()}

# LLM node for direct LLM invocation


def llm_node(state: State):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# RAG retrieval node


def rag_node(state: State):
    retriever = HybridRetriever(
        collection=collection,
        dense_field=dense_field,
        sparse_field=sparse_field,
        top_k=3,
        embeddings_model=dense_embedding_func,
        sparse_embeddings_model=sparse_embedding_func,
    )
    last_message = state["messages"][-1]
    query = last_message if isinstance(
        last_message, str) else last_message.content
    # Retrieve documents
    docs = retriever.get_relevant_documents(query)
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])
    # Create a prompt with the retrieved context
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
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
