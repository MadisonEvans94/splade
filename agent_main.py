# agent_main.py
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from pymilvus import WeightedRanker, connections, Collection
from constants import (
    COLLECTION_NAME,
    CONNECTION_ARGS,
    TOP_K,
    EXIT_COMMAND,
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from agents.tools.rag_tool import RAGTool
from retrievers import StandardRetriever  # Ensure this is imported if needed

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

# Instantiate the collection
collection = Collection(COLLECTION_NAME)


def get_corpus(collection: Collection) -> List[str]:
    # Fetch all documents with a query expression matching any valid pk
    results = collection.query(expr="pk != ''", output_fields=["text"])

    # Use tqdm to show progress as documents are processed
    corpus = [doc["text"] for doc in tqdm(results, desc="fitting bm25 model")]

    return corpus


corpus = get_corpus(collection)
sparse_embedding_type = "BM25"


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

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    # Ensures messages are returned as a list of BaseMessage objects.
    return_messages=True
)


PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know; don't try to make up an answer.

{context}

Chat History:
{chat_history}

Question: {input}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=PROMPT_TEMPLATE
)


def setup_chain(hybrid: bool):
    if hybrid:
        logging.info("Running in hybrid retrieval mode.")
        retriever = MilvusCollectionHybridSearchRetriever(
            collection=collection,
            rerank=WeightedRanker(0.5, 0.5),
            anns_fields=[dense_field, sparse_field],
            field_embeddings=[dense_embedding_func, sparse_embedding_func],
            field_search_params=[dense_search_params, sparse_search_params],
            top_k=TOP_K,
            text_field=text_field,
        )
    else:
        logging.info("Running in dense-only retrieval mode.")
        retriever = StandardRetriever(
            collection=collection,
            dense_field=dense_field,
            top_k=TOP_K,
            embeddings_model=dense_embedding_func,
        )

    # Define the contextualization prompt for history-aware retrieval
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Reformulate the user's query to be independent of prior context."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,  # This could be your Milvus hybrid retriever
        prompt=contextualize_q_prompt,
    )



    # Define the QA prompt for combining documents
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant for answering questions."),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}\n\nQuestion: {input}\nAnswer:"),
        ]
    )


    # Use create_stuff_documents_chain to combine documents
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    # Correctly use `combine_docs_chain` as the argument

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    return retrieval_chain


# Prepare the RAG chain
retrieval_chain = setup_chain(hybrid=True)  # Use the function defined earlier

# Create the RAG tool
rag_tool = RAGTool(chain=retrieval_chain, memory=memory)

# Add the tool to the agent
tools = [rag_tool]

# Initialize the agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

# Define the chatbot loop
EXIT_COMMAND = 'exit'


def chatbot_loop(agent_executor):
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == EXIT_COMMAND:
            logging.info("User exited the conversation.")
            print("Goodbye!")
            break
        print("\n--------------------------\n")
        try:
            logging.info(f"User input: {user_input}")

            # Run the agent executor with the user input
            response = agent_executor.invoke(input=user_input)

            logging.info(f"Agent response: {response}")
            print(f"\n\nBot: \n{response}\n")
        except Exception as e:
            logging.error("Error generating response", exc_info=True)
            continue



if __name__ == "__main__":
    chatbot_loop(agent_executor)
