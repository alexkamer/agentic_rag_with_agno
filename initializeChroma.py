import os
from os import getenv
from typing import Optional, Dict, List, Any
import PyPDF2
import re

from agno.utils.log import logger

from agno.agent import Agent
from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.tools import Toolkit
from agno.vectordb.chroma import ChromaDb
from agno.document import Document


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

llm = get_llm()
embedder = get_embedder()





def process_pdf(pdf_path: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Process a PDF file and split it into chunks while preserving sentence boundaries.
    Returns a list of dictionaries containing the chunk text and its page number.
    """
    chunks = []
    current_chunk = []
    current_size = 0
    current_page = 1
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_size = len(sentence)
                
                if current_size + sentence_size > chunk_size and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'page': current_page
                    })
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(sentence)
                current_size += sentence_size
                current_page = page_num
            
            # Add page number to the last chunk of the page
            if current_chunk:
                current_chunk[-1] = f"{current_chunk[-1]}"
    
    # Add any remaining text as the final chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'page': current_page
        })
    
    return chunks


class ChromaStore:
    """
    Handles all Chroma DB operations including initialization and querying.
    """
    def __init__(self):
        # Initialize with a default collection that we'll use for temporary operations
        self.client = ChromaDb(
            collection="default",
            path="tmp/chromadb",
            persistent_client=True,
            embedder=embedder
        )
        
    def create_collection(self, name: str):
        """Create a new collection if it doesn't exist."""
        try:
            # Create a new client for this collection
            collection_client = ChromaDb(
                collection=name,
                path="tmp/chromadb",
                persistent_client=True,
                embedder=embedder
            )
            # Explicitly create the collection
            collection_client.create()
            logger.info(f"Successfully created collection {name}")
            return collection_client
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            return None

    def get_collection(self, name: str):
        """Get an existing collection."""
        try:
            return ChromaDb(
                collection=name,
                path="tmp/chromadb",
                persistent_client=True,
                embedder=embedder
            )
        except Exception as e:
            logger.error(f"Error getting collection {name}: {str(e)}")
            return None

    def store_chunks(self, collection_name: str, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        """Store text chunks in the collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        if not collection:
            logger.error(f"Failed to get or create collection {collection_name}")
            return
        
        # Create Document objects for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata["chunk_id"] = f"{collection_name}_chunk_{i}"
            doc_metadata["page"] = chunk['page']  # Add page number to metadata
            documents.append(Document(content=chunk['text'], meta_data=doc_metadata))
        
        try:
            collection.insert(documents=documents)
            logger.info(f"Successfully stored {len(chunks)} chunks in collection {collection_name}")
        except Exception as e:
            logger.error(f"Error storing chunks in collection {collection_name}: {str(e)}")

    def query_chunks(self, collection_name: str, query: str, n_results: int = 3):
        """Query the collection for relevant chunks."""
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return None
        
        try:
            return collection.query(
                query_texts=[query],
                n_results=n_results
            )
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
            return None



def initialize_books():
    """
    Initialize the book collections in Chroma DB
    """
    logger.info("Starting book initialization...")
    chroma_store = ChromaStore()
    logger.info("ChromaStore initialized")
    
    pdf_files = [f for f in os.listdir("textFiles/raw") if f.endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    for pdf_file in pdf_files:
        try:
            book_name = os.path.splitext(pdf_file)[0]
            pdf_path = f"textFiles/raw/{pdf_file}"
            logger.info(f"Processing {pdf_file}...")

            chunks = process_pdf(pdf_path)
            if not chunks:
                logger.error(f"No chunks found for {pdf_file}")
                continue
            
            logger.info(f"Generated {len(chunks)} chunks from {pdf_file}")
            first_chunk = chunks[0]['text']  # Get the text from the first chunk
            logger.info("Generating metadata...")
            
            metadata_prompt = f"""Based on this text excerpt from a book, generate a title, description, and relevant topics.

                Text excerpt:
                {first_chunk[:1000]}

                Please provide the following in JSON format:
                {{
                    "title": "Book title",
                    "description": "2-3 sentence description",
                    "topics": ["topic1", "topic2", "topic3", ...]
                }}

                Your response:"""
            
            metadata_agent = Agent(
                model=llm,
                context=metadata_prompt,
                debug_mode=True
            )
            metadata_response = metadata_agent.run(metadata_prompt)

            try:
                book_info = eval(metadata_response.content)
                logger.info(f"Generated metadata: {book_info}")
            except Exception as e:
                logger.error(f"Error parsing metadata response for {pdf_file}: {str(e)}")
                continue

            logger.info(f"Creating collection for {book_name}...")
            collection = chroma_store.create_collection(book_name)
            if not collection:
                logger.error(f"Failed to create collection for {book_name}")
                continue
                
            metadata = {
                "title": book_info["title"],
                "description": book_info["description"],
                "topics": ", ".join(book_info["topics"])
            }

            logger.info(f"Storing metadata for {book_name}...")
            chroma_store.store_chunks(book_name, [{'text': book_info["description"], 'page': 0}], metadata=metadata)
            
            logger.info(f"Storing {len(chunks)} chunks for {book_name}...")
            chroma_store.store_chunks(book_name, chunks)
            logger.info(f"Successfully initialized {book_name} with AI-generated metadata")

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")







if __name__ == "__main__":
    logger.info("Starting book initialization process...")
    initialize_books()
    logger.info("Book initialization completed")