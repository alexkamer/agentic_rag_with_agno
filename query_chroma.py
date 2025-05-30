import os
from os import getenv
from typing import Optional, Dict, List, Any
from agno.utils.log import logger
from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.document import Document
from agno.agent import Agent


def get_llm():
    return AzureOpenAI(
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT", "https://azure-oai-east2.openai.azure.com/"),
        azure_deployment="gpt-4-1"
    )

def get_embedder():
    return AzureOpenAIEmbedder(
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT", "https://azure-oai-east2.openai.azure.com/"),
        azure_deployment="text-embedding-3-small"
    )

embedder = get_embedder()
llm = get_llm()
class ChromaQuery:
    """
    A simple class to query documents from ChromaDB collections.
    """
    def __init__(self, collection_name: str):
        """
        Initialize the ChromaQuery with a specific collection.
        Args:
            collection_name: Name of the ChromaDB collection to query
        """
        self.collection_name = collection_name
        self.client = ChromaDb(
            collection=collection_name,
            path="tmp/chromadb",
            persistent_client=True,
            embedder=embedder
        )
        logger.info(f"Initialized ChromaQuery for collection: {collection_name}")

    def query(self, query_text: str, n_results: int = 3) -> Dict:
        """
        Query the collection for similar documents.
        Args:
            query_text: The text to search for
            n_results: Number of results to return
        Returns:
            Dictionary containing the query results
        """
        try:
            logger.info(f"Querying collection {self.collection_name} with: {query_text}")
            results = self.client.search(
                query=query_text,
                limit=n_results
            )
            
            if results:
                logger.info(f"Found {len(results)} results")
                # Convert Document objects to dictionary format
                formatted_results = {
                    'documents': [[doc.content for doc in results]],
                    'distances': [[doc.meta_data.get('distances', 0) for doc in results]],
                    'metadatas': [[doc.meta_data for doc in results]]
                }
                return formatted_results
            else:
                logger.info("No results found")
                return None
                
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return None

    def format_results(self, results: Dict) -> str:
        """
        Format the query results into a readable string.
        Args:
            results: The results dictionary from query()
        Returns:
            Formatted string of results
        """
        if not results or not results['documents']:
            return "No results found."
        
        formatted_results = []
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results.get('metadatas', [[]])[0]
        
        for i, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
            similarity = 1 - distance
            page_num = metadata.get('page', 'N/A')
            page_info = f" (Page {page_num})" if page_num != 'N/A' else ""
            formatted_results.append(f"Result {i+1} [Similarity: {similarity:.2f}]{page_info}:\n{doc}\n")
        
        return "\n".join(formatted_results)

book_agent = Agent(
    model=llm,
    debug_mode=True
    )

def search_books(query: str, collection_name: str = "PokerBook", n_results: int = 3):
    """
    Search a specific book collection and return formatted results.
    Args:
        query: The search query
        collection_name: Name of the collection to search
        n_results: Number of results to return
    Returns:
        Formatted search results
    """

    searcher = ChromaQuery(collection_name)
    results = searcher.query(query, n_results)
    raw_results = searcher.format_results(results)
    book_prompt = f"""Based on the following search results from our books, please provide a clear and concise answer to the question: "{query}"
    Search Results:
    {raw_results}

    Please provide a well-structured response that:
    1. Directly answers the question
    2. Includes relevant details from the search results
    3. Cites the source (book and page number) when possible
    4. Is written in a clear, conversational style

    Your response:"""
    
    response = book_agent.run(book_prompt)
    return response.content




# Example usage
if __name__ == "__main__":
    # Example query
    query = "What are some tips for beginner singers?"
    results = search_books(query)
    print("\nSearch Results:")
    print(results)
