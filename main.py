import os
import logging
from typing import Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from tqdm import tqdm
from retrievers import SpladeSparseEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from retrievers import CustomHybridRetriever, StandardRetriever
from langchain_milvus.utils.sparse import BaseSparseEmbedding, BM25SparseEmbedding
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)
from langchain.memory import ConversationBufferWindowMemory
from constants import COLLECTION_NAME, CONNECTION_ARGS
import click
from retrievers import StandardRetriever
from langchain.chains import RetrievalQA

TOP_K = 5
EXIT_COMMAND = 'exit'
CONV_HISTORY_SIZE = 5  # Example size of conversation memory buffer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# instantiate the collection
collection = Collection(COLLECTION_NAME)

# get corpus


def get_corpus(collection: Collection) -> List[str]:
    # Fetch all documents with a query expression matching any valid pk
    results = collection.query(expr="pk != ''", output_fields=["text"])

    # Use tqdm to show progress as documents are processed
    corpus = [doc["text"] for doc in tqdm(results, desc="fitting bm25 model")]

    return corpus



corpus = get_corpus(collection)
sparse_embedding_type = "BM25"
if sparse_embedding_type == "SPLADE": 
    logging.info("Using SPLADE sparse embeddings.")
    sparse_embedding_func = SpladeSparseEmbedding()
else: 
    logging.info("Using BM25 sparse embeddings.")
    sparse_embedding_func = BM25SparseEmbedding(corpus)
dense_embedding_func = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Define fields and collection
pk_field = "pk"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"


# Define search parameters for dense and sparse fields
dense_search_params = {"metric_type": "IP", "params": {}}
sparse_search_params = {"metric_type": "IP"}

# Define prompt template
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

{history}
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=PROMPT_TEMPLATE
)

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Initialize memory
memory = ConversationBufferWindowMemory(
    input_key="question",
    memory_key="history",
    k=CONV_HISTORY_SIZE
)

# Function to set up the retriever and RetrievalQA chain


def setup_chain(hybrid: bool):
    if hybrid:
        logging.info("Running in hybrid retrieval mode.")
        # Use the custom hybrid retriever
        retriever = CustomHybridRetriever(
            collection=collection,
            dense_field=dense_field,
            sparse_field=sparse_field,
            top_k=TOP_K,
            embeddings_model=dense_embedding_func,
            sparse_embeddings_model=sparse_embedding_func,
            ratio=[0.5, 0.5]  # Adjust this ratio as needed
        )
    else:
        logging.info("Running in dense-only retrieval mode.")
        # Use the standard retriever for dense-only search
        retriever = StandardRetriever(
            collection=collection,
            dense_field=dense_field,
            top_k=TOP_K,
            embeddings_model=dense_embedding_func
        )

    # Create the RetrievalQA chain with memory and custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory}
    )

    return qa_chain


# Define the chatbot loop to include sources in the response


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_sources(sources):
    """Formats the source information for output."""
    formatted_sources = []
    for i, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_info = f"Source {i}:"
        source_info += f"\n- Document ID: {metadata.get('pk', 'Unknown')}\n"
        source_info += "----------------------------------"
        source_info += f"\n\n{doc.page_content}...\n\n"
        source_info += "----------------------------------"
        source_info += f"\n- Retrieved by: {metadata.get('retriever', 'Unknown')}"
        formatted_sources.append(source_info)
    return "\n\n".join(formatted_sources)


def chatbot_loop(qa_chain: RetrievalQA):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            # Invoke the RetrievalQA to get the answer
            response = qa_chain.invoke({"query": user_input})
            print(f"\n\nBot: \n{response['result']}\n")
            if 'source_documents' in response:
                formatted_sources = format_sources(
                    response['source_documents'])
                print(f"Sources:\n{formatted_sources}\n")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            continue


@click.command()
@click.option('--hybrid', is_flag=True, help="Enable hybrid retrieval mode.")
def main(hybrid):
    qa_chain = setup_chain(hybrid)
    try:
        chatbot_loop(qa_chain)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()
