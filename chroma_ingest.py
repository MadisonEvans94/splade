import os
import sys
import argparse
from uuid import uuid4

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain_chroma import Chroma


def ingest_wikipedia_documents(topic, max_docs):
    """
    Ingest Wikipedia documents into a Chroma vector store.
    
    Args:
        topic (str): Wikipedia topic to load
        max_docs (int): Maximum number of documents to load
    
    Raises:
        ValueError: If no documents can be loaded for the given topic
    """
    try:
        # Directory configuration
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PERSIST_DIR = os.path.join(BASE_DIR, "chroma_langchain_db")

        # Initialize vector store
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=PERSIST_DIR,
        )

        # Load documents from Wikipedia
        loader = WikipediaLoader(topic, load_max_docs=max_docs)
        raw_documents = loader.load()

        # Check if any documents were loaded
        if not raw_documents:
            raise ValueError(f"No documents found for topic: {topic}")

        # Split the documents into chunks using semantic chunking
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        docs = text_splitter.create_documents(
            [doc.page_content for doc in raw_documents]
        )

        # Add the chunked documents to the vector store
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)

        print("Ingestion complete")
        print(f"Created {len(docs)} chunks from {topic}")

    except Exception as e:
        print(f"Error during document ingestion: {e}")
        sys.exit(1)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia documents into a Chroma vector store")
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        required=True,  # Make topic a required argument
        help="Wikipedia topic to load (required)"
    )
    parser.add_argument(
        "-m",
        "--max-docs",
        type=int,
        default=3,
        help="Maximum number of documents to load (default: 3)"
    )

    try:
        # Parse arguments
        args = parser.parse_args()

        # Validate topic is not an empty string
        if not args.topic.strip():
            raise ValueError("Topic cannot be an empty string")

        # Call ingestion function with parsed arguments
        ingest_wikipedia_documents(args.topic, args.max_docs)

    except ValueError as ve:
        print(f"Input Error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
