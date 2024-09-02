import os
import pickle
import logging
from typing import Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymilvus import (
    Collection,
    connections,
)
from langchain_core.runnables.base import RunnableSerializable
from constants import COLLECTION_NAME, CONNECTION_ARGS
import click
from pymilvus import Collection
from retrievers import CustomHybridRetriever, StandardRetriever

TOP_K = 2
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

# Function to set up the retriever and chains


def setup_chain(hybrid: bool):
    if hybrid:
        ratio = [0.5, 0.5]
        logging.info("Running in hybrid retrieval mode.")
        # Use the custom hybrid retriever
        retriever = CustomHybridRetriever(
            collection=collection,
            dense_field=dense_field,
            sparse_field=sparse_field,
            top_k=TOP_K,
            embeddings_model=dense_embeddings,
            sparse_embeddings_model=sparse_embeddings,
            ratio=ratio
        )
    else:
        logging.info("Running in dense-only retrieval mode.")
        # Use the standard retriever for dense-only search
        retriever = StandardRetriever(
            collection=collection,
            dense_field=dense_field,
            top_k=TOP_K,
            embeddings_model=dense_embeddings
        )

    # Define the chain using the configured retriever
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x: {"context": format_docs(x["context"]), "sources": x["context"]}))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Adjust the final output to include source document information
    return RunnableParallel(
        {
            "context": retriever.invoke,
            "question": RunnablePassthrough()
        }
    ).assign(
        answer=rag_chain_from_docs,
        # Include source documents in the output
        sources=(lambda x: x["context"])
    )


# Define prompt template
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provide answers to questions by using fact-based and statistical information when possible.
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

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the chatbot loop to include sources in the response


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_sources(sources):
    """Formats the source information for output."""
    formatted_sources = []
    for i, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_info = f"Source {i}:"
        source_info += f"\n- Document ID: {metadata.get('pk', 'Unknown')}"
        source_info += "----------------------------------"
        source_info += f"\n\n{doc.page_content[:200]}...\n\n"
        source_info += "----------------------------------"
        source_info += f"\n- Retrieved by: {metadata.get('retriever', 'Unknown')}"
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
            response = rag_chain.invoke(user_input)
            print(f"\n\nBot: \n{response['answer']}\n")

            # Print the sources
            if 'sources' in response:
                formatted_sources = format_sources(response['sources'])
                print(f"Sources:\n{formatted_sources}\n")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue


@click.command()
@click.option('--hybrid', is_flag=True, help="Enable hybrid retrieval mode.")
def main(hybrid):
    rag_chain = setup_chain(hybrid)
    try:
        chatbot_loop(rag_chain)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()
