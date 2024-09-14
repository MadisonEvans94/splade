import os
import pprint
from typing import Tuple
import click
from pymilvus import (
    Collection,
    connections,
)
from tqdm import tqdm
from constants import COLLECTION_NAME, CONNECTION_ARGS
from retrievers import CustomHybridRetriever, SpladeSparseEmbedding, StandardRetriever
from langchain_milvus.utils.sparse import BaseSparseEmbedding, BM25SparseEmbedding
import logging
from langchain.chains import RetrievalQA
from typing import List
from datasets import Dataset
from langchain_core.prompts import PromptTemplate
import pandas as pd
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
# Connect to Milvus
connections.connect(**CONNECTION_ARGS)

df = pd.read_csv('testset.csv')
questions: List[str] = df['question'].to_list()
ground_truths: List[str] = df['ground_truth'].to_list()
TOP_K = 3
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
TEXT_FIELD = "text"
PK_FIELD = "pk"
# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define prompt template
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep your answers concise and to the point. 

{context}

Question: {question}
Answer:
"""

def get_corpus(collection: Collection) -> List[str]:
    # Fetch all documents with a query expression matching any valid pk
    results = collection.query(expr="pk != ''", output_fields=["text"])

    # Use tqdm to show progress as documents are processed
    corpus = [doc["text"] for doc in tqdm(results, desc="fitting bm25 model")]

    return corpus


def connect_to_milvus(connection_args: dict):
    """Connect to the Milvus database."""
    connections.connect(**connection_args)
    logging.info("Connected to Milvus.")


def load_test_data(csv_file: str) -> Tuple[List[str], List[str]]:
    """Load test data from a CSV file."""
    df = pd.read_csv(csv_file)
    questions = df['question'].tolist()
    ground_truths = df['ground_truth'].tolist()
    logging.info(f"Loaded {len(questions)} questions from {csv_file}.")
    return questions, ground_truths


def initialize_llm(api_key: str, model_name: str) -> ChatOpenAI:
    """Initialize the language model."""
    llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)
    logging.info(f"Initialized LLM with model {model_name}.")
    return llm


def get_collection(collection_name: str) -> Collection:
    """Retrieve the collection from Milvus."""
    collection = Collection(collection_name)
    logging.info(f"Retrieved collection '{collection_name}'.")
    return collection


def get_corpus(collection: Collection) -> List[str]:
    """Fetch all documents from the collection to build the corpus."""
    logging.info("Fetching documents from collection to build corpus.")
    results = collection.query(
        expr=f"{PK_FIELD} != ''", output_fields=[TEXT_FIELD])
    corpus = [doc[TEXT_FIELD] for doc in tqdm(results, desc="Building corpus")]
    logging.info(f"Corpus built with {len(corpus)} documents.")
    return corpus


def setup_embeddings(corpus: List[str], api_key: str) -> Tuple[BM25SparseEmbedding, OpenAIEmbeddings]:
    """Set up sparse and dense embeddings."""
    logging.info("Setting up embeddings.")
    sparse_embedding_func = BM25SparseEmbedding(corpus)
    dense_embedding_func = OpenAIEmbeddings(
        openai_api_key=api_key, model="text-embedding-ada-002")
    logging.info("Embeddings setup complete.")
    return sparse_embedding_func, dense_embedding_func


def get_prompt_template() -> PromptTemplate:
    """Create and return the prompt template."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE.strip()
    )
    logging.info("Prompt template created.")
    return prompt


def setup_retriever(
    hybrid: bool,
    collection: Collection,
    dense_field: str,
    sparse_field: str,
    top_k: int,
    dense_embedding_func: OpenAIEmbeddings,
    sparse_embedding_func: BM25SparseEmbedding
) -> object:
    """Set up the retriever based on the retrieval mode."""
    if hybrid:
        logging.info("Running in hybrid retrieval mode.")
        retriever = CustomHybridRetriever(
            collection=collection,
            dense_field=dense_field,
            sparse_field=sparse_field,
            top_k=top_k,
            embeddings_model=dense_embedding_func,
            sparse_embeddings_model=sparse_embedding_func,
        )
    else:
        logging.info("Running in dense-only retrieval mode.")
        retriever = StandardRetriever(
            collection=collection,
            dense_field=dense_field,
            top_k=top_k,
            embeddings_model=dense_embedding_func
        )
    logging.info("Retriever setup complete.")
    return retriever


def setup_qa_chain(llm: ChatOpenAI, retriever: object, prompt: PromptTemplate) -> RetrievalQA:
    """Set up the QA chain with the given LLM, retriever, and prompt."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    logging.info("QA chain setup complete.")
    return qa_chain


@click.command()
@click.option('--hybrid', is_flag=True, default=False, help="Use hybrid retrieval.")
def main(hybrid): 
    contexts = []
    answers = []
    if hybrid: 
        EXT = 'hybrid'
    else: 
        EXT = 'naive'
    """Main function to orchestrate the retrieval and QA process."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the QA system...")

    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable not found.")
        return
    
    # Connect to Milvus
    connect_to_milvus(CONNECTION_ARGS)

    # Load test data
    questions, ground_truths = load_test_data('testset.csv')
    
    # Initialize LLM
    llm = initialize_llm(OPENAI_API_KEY, "gpt-4o-mini")
    
    # Get collection
    collection = get_collection(COLLECTION_NAME)
    
    # Get corpus
    corpus = get_corpus(collection)
    
    # Set up embeddings
    sparse_embedding_func, dense_embedding_func = setup_embeddings(
        corpus, OPENAI_API_KEY)

    # Get prompt template
    prompt = get_prompt_template()
    
    # Set up retriever
    retriever = setup_retriever(
        hybrid=True,
        collection=collection,
        dense_field=DENSE_FIELD,
        sparse_field=SPARSE_FIELD,
        top_k=TOP_K,
        dense_embedding_func=dense_embedding_func,
        sparse_embedding_func=sparse_embedding_func
    )
    
    
    # Set up QA chain
    qa_chain = setup_qa_chain(llm, retriever, prompt)
    
    for question in questions: 
        response = qa_chain.invoke({"query": question})
        answer = response['result']
        context = [doc.page_content for doc in response['source_documents']]
        contexts.append(context)
        answers.append(answer)
        
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness
        ],
    )
    result_df = result.to_pandas()
    result_df.to_csv(f'eval_results_dataheavy_{EXT}.csv', index=False)

if __name__ == "__main__":
    main()

    
    
# for question in questions: 
#     # run rag and append to contexts list and answers list 
#     try:
#         # Invoke the RetrievalQA to get the answer
#         qa_chain = setup_chain(hybrid=True)
#         response = qa_chain.invoke({"query": question})
#         logging.info(f"\n\nBot: \n{response['result']}\n")
#         answers.append(response['result'])
#     except Exception as e:
#         logging.error(f"Error generating response: {e}")
#         continue
#     pass

