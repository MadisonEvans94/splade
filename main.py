import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from constants import CONNECTION_ARGS, COLLECTION_NAME

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to Milvus


def connect_to_milvus():
    connections.connect(**CONNECTION_ARGS)

# Set up a custom retrieval-augmented generation (RAG) chain


def setup_custom_rag_chain():
    connect_to_milvus()

    # Initialize the Milvus collection for retrieval
    collection = Collection(COLLECTION_NAME)

    # Create an OpenAI embeddings object
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    # Placeholder for documents - this should be your actual list of documents
    documents = []  # TODO: Load or generate documents for your use case

    # Set up the Milvus vector store from documents
    vector_store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args=CONNECTION_ARGS
    )

    # Set up the ChatOpenAI LLM
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)

    # Convert the vector store to a retriever with a more specific strategy
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={
                                          "k": 5, "lambda_mult": 0.25})

    # Create a custom chain for retrieval and QA
    custom_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True  # This enables returning source documents with metadata
    )

    return custom_chain


def main():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    rag_chain = setup_custom_rag_chain()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate response using the custom chain
        response = rag_chain({"query": user_input})

        # Print the main answer
        print(f"Bot: {response['result']}")

        # Print metadata about the retrieved documents
        if 'source_documents' in response:
            print("\nRetrieved Documents:")
            for doc in response['source_documents']:
                metadata = doc.metadata  # Access metadata
                # First 100 characters of the content
                content_snippet = doc.page_content[:100]
                print(
                    f"- Document ID: {metadata.get('id', 'N/A')}, Snippet: {content_snippet}")


if __name__ == "__main__":
    main()
