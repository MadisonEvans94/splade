from typing import List
from langchain.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RetrieveDocuments(BaseTool):
    """
    A tool to retrieve documents based on a query if the query is about Intel Corporation.
    Uses a Chroma vector store to return relevant documents.
    """

    name: str = "retrieve_documents"
    description: str = (
        "Retrieves relevant documents for queries specifically about Intel Corporation, "
        "including its products, history, financials, and news. Do not use for unrelated topics."
    )
    embeddings: Embeddings
    vector_store: VectorStore

    def _run(self, query: str) -> List[Document]:
        """
        Perform the document retrieval.
        """
        try:
            # Perform similarity search directly
            documents = self.vector_store.similarity_search(query, k=5)
            return documents

        except Exception as e:
            logger.error("Error occurred during document retrieval", exc_info=True)
            return []

    async def _arun(self, query: str) -> List[Document]:
        """
        Asynchronous version of document retrieval.
        """
        return self._run(query)
