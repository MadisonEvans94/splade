from typing import List
from langchain.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RetrieveDocuments(BaseTool):
    """
    A tool to retrieve documents based on a query if the query is about the city of Atlanta.
    Uses a Chroma vector store to return relevant documents.
    """

    name: str = "retrieve_documents"
    description: str = "Retrieves relevant documents for a given query if the query is about the city of Atlanta"
    embeddings: Embeddings
    vector_store: VectorStore

    def _run(self, query: str) -> List[Document]:
        """
        Perform the document retrieval.
        """
        try:
            logger.debug("RetrieveDocuments _run method called.")
            logger.debug(f"Query: {query}")

            # Check vector store initialization
            if not self.vector_store:
                logger.error("Vector store is not initialized.")
                return []

            # Perform similarity search directly
            documents = self.vector_store.similarity_search(query, k=5)
            logger.debug(f"Number of documents retrieved: {len(documents)}")

            # Log retrieved documents
            for i, doc in enumerate(documents):
                logger.debug(
                    f"Document {i + 1}: {doc.page_content[:200]}... (truncated)")

            return documents

        except Exception as e:
            logger.error("Error occurred during document retrieval", exc_info=True)
            return []



    async def _arun(self, query: str) -> List[Document]:
        """
        Asynchronous version of document retrieval.
        """
        return self._run(query)
