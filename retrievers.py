import os
import pickle
import logging
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import (
    Collection,
    WeightedRanker,
    connections,
)
from langchain_core.runnables.base import RunnableSerializable
from constants import COLLECTION_NAME, CONNECTION_ARGS

TOP_K = 5
EXIT_COMMAND = 'exit'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embeddings
with open('bm25_embeddings.pkl', 'rb') as f:
    sparse_embeddings: BM25SparseEmbedding = pickle.load(f)

dense_embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Define fields and collection
pk_field = "pk"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"
collection = Collection(COLLECTION_NAME)

# Define search parameters
sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}

# Initialize retriever
retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    rerank=WeightedRanker(0.5, 0.5),
    anns_fields=[dense_field, sparse_field],
    field_embeddings=[dense_embeddings, sparse_embeddings],
    field_search_params=[dense_search_params, sparse_search_params],
    top_k=3,
    text_field=text_field,
)

# Extend the retriever to add retrieval type metadata


def retrieve_with_metadata(query):
    dense_results = retriever.invoke(query, method='dense')
    sparse_results = retriever.invoke(query, method='sparse')

    # Add metadata to each document to indicate retrieval method
    for doc in dense_results:
        doc.metadata['retrieval_method'] = 'dense'
    for doc in sparse_results:
        doc.metadata['retrieval_method'] = 'sparse'

    # Combine results, maintaining the retrieval method metadata
    combined_results = dense_results + sparse_results
    return combined_results


# Define prompt template
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["context", "question"])

# Utility function to format documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the Q&A chain with sources
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retrieve_with_metadata, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# Define the chatbot loop to include sources in the response


def format_sources(sources):
    """
    Formats the source documents for clearer output.

    Args:
        sources (list): A list of Document objects retrieved as sources.

    Returns:
        str: A formatted string representation of the sources.
    """
    formatted_sources = []
    for i, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_info = f"Source {i}:"
        source_info += f"\n- Filename: {metadata.get('filename', 'Unknown')}"
        source_info += f"\n- Document ID: {metadata.get('pk', 'Unknown')}"
        # Show a preview of the content
        source_info += f"\n- Content Preview: {doc.page_content[:50]}..."
        source_info += f"\n- Retrieval Type: {metadata.get('retrieval_method', 'Unknown')}"
        formatted_sources.append(source_info)
    return "\n\n".join(formatted_sources)


def chatbot_loop(rag_chain: RunnableSerializable[Any, str]):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            # Pass the user input as the "question" in the dictionary
            response = rag_chain.invoke(user_input)
            print(f"\n\nBot: \n{response['answer']}\n")
            # Print the sources used in a formatted manner
            formatted_sources = format_sources(response['context'])
            print(f"Sources:\n{formatted_sources}\n")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue

def main():
    try:
        chatbot_loop(rag_chain_with_source)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()
