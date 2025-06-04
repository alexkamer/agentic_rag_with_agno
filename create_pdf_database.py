from agno.agent import Agent
from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.document.reader.pdf_reader import PDFReader as BasePDFReader
from os import getenv
import logging
from pathlib import Path
import json
import time
from PyPDF2 import PdfReader as PyPDF2Reader
from sqlalchemy import select

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class CustomPDFReader(BasePDFReader):
    def __init__(self, chunk=True):
        super().__init__(chunk=chunk)
        self.llm = get_llm()
        logger.info("Initialized CustomPDFReader")
        self.document_keywords = {}  # Store keywords for each document
        
        # Map of source filenames to their download links
        self.source_links = {
            "emon_users_guide.pdf": "https://www.intel.com/content/www/us/en/search.html?ws=text#q=emon%20api%20guide&sort=relevancy&f:@tabfilter=[Developers]",
            "sep_user_guide.pdf": "https://www.intel.com/content/www/us/en/content-details/686066/sampling-enabling-product-user-s-guide.html?wapkw=SEP%20user%20guide",
            "socwatch_user_guide.pdf": "https://www.intel.com/content/www/us/en/docs/socwatch/user-guide/2023-1/overview.html",
            "oneapi_user_guide.pdf": "https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-1/overview.html",
            "vtune_profiler_installation_guide_2025.pdf" : "https://www.intel.com/content/www/us/en/docs/vtune-profiler/installation-guide/2025-0/overview.html",
            "vtune_profiler_user_guide_2025.pdf" : "https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/overview.html"
        }

    def _get_source_link(self, filename: str) -> str:
        """Get the source link for a given filename"""
        return self.source_links.get(filename, f"/download/{filename}")  # Fallback to default if not found

    def _generate_document_keywords(self, text):
        """Generate keywords for the entire document"""
        keywords_prompt = f"""Based on this document, generate a comprehensive list of relevant keywords that best describe its content.
        Return only a JSON array of keywords, nothing else.
        
        Document:
        {text}
        
        Keywords:"""
        
        try:
            keywords_response = self.llm.get_client().chat.completions.create(
                model=self.llm.azure_deployment,
                messages=[{"role": "user", "content": keywords_prompt}]
            )
            logger.info(f"Keywords response: {keywords_response}")
            return json.loads(keywords_response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating keywords: {str(e)}")
            return []

    def _select_relevant_keywords(self, chunk_text, all_keywords):
        """Select relevant keywords for a specific chunk from the document's keywords"""
        keywords_prompt = f"""From this list of keywords, select the ones most relevant to this text chunk.
        Return only a JSON array of selected keywords, nothing else.
        
        Available keywords: {json.dumps(all_keywords)}
        
        Text chunk:
        {chunk_text}
        
        Selected keywords:"""
        
        try:
            keywords_response = self.llm.get_client().chat.completions.create(
                model=self.llm.azure_deployment,
                messages=[{"role": "user", "content": keywords_prompt}]
            )
            logger.info(f"Selected keywords response: {keywords_response}")
            return json.loads(keywords_response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error selecting keywords: {str(e)}")
            return []

    def _generate_tool_tag(self, text):
        """Generate a tool tag for the text"""
        tool_tag_prompt = f"""Based on this text, generate a short, descriptive tool tag (max 3 words).
        The tool tag should be a single string without quotes or any additional formatting.
        Example responses:
        - Technical Documentation
        - User Guide
        - API Reference
        
        Text to analyze:
        {text[:500]}
        
        Tool tag:"""
        
        try:
            tool_tag_response = self.llm.get_client().chat.completions.create(
                model=self.llm.azure_deployment,
                messages=[{"role": "user", "content": tool_tag_prompt}]
            )
            logger.info(f"Raw tool tag response: {tool_tag_response}")
            
            # Clean up the response
            tool_tag = tool_tag_response.choices[0].message.content.strip()
            # Remove any quotes or special characters
            tool_tag = tool_tag.replace('"', '').replace("'", "").strip()
            
            logger.info(f"Cleaned tool tag: {tool_tag}")
            return tool_tag if tool_tag else "General Document"
        except Exception as e:
            logger.error(f"Error generating tool tag: {str(e)}")
            return "General Document"

    def read(self, pdf: Path):
        """Override read method to add our custom processing"""
        logger.info(f"Reading PDF: {pdf}")
        try:
            # First read the full document text
            with open(pdf, 'rb') as file:
                pdf_reader = PyPDF2Reader(file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text()
            
            # Generate keywords for the full document
            self.document_keywords[str(pdf)] = self._generate_document_keywords(full_text)
            logger.info(f"Generated keywords for {pdf}: {self.document_keywords[str(pdf)]}")
            
            # Process the document using the base class method
            documents = super().read(pdf)
            
            # Add our custom metadata to each document
            for i, doc in enumerate(documents):
                # Generate tool tag for this chunk
                tool_tag = self._generate_tool_tag(doc.content)
                logger.info(f"Generated tool tag: {tool_tag}")
                
                # Select relevant keywords for this chunk
                chunk_keywords = self._select_relevant_keywords(
                    doc.content,
                    self.document_keywords[str(pdf)]
                )
                
                # Get the source link for this document
                source_link = self._get_source_link(pdf.name)
                
                # Create our custom metadata structure
                custom_metadata = {
                    "page": doc.meta_data.get("page", 0),
                    "chunk": doc.meta_data.get("chunk", 0),
                    "chunk_size": doc.meta_data.get("chunk_size", 0),
                    "document_metadata": {
                        "source": pdf.name,
                        "tool_tag": tool_tag,
                        "chunk_index": i,
                        "upload_time": int(time.time()),
                        "document_name": pdf.name,
                        "source_link": source_link,  # Use the mapped source link
                        "page_number": doc.meta_data.get("page", 0),
                        "keywords": chunk_keywords,
                        "collection_id": "0678d463-0e46-4441-8d25-df983c0c04e5"
                    }
                }
                
                # Update document metadata
                doc.meta_data = custom_metadata
                logger.info(f"Set custom metadata: {json.dumps(custom_metadata, indent=2)}")
            
            return documents
        except Exception as e:
            logger.error(f"Error in read: {str(e)}")
            raise

def main():
    # Initialize the custom reader and vector store
    logger.info("Initializing custom reader and vector store...")
    custom_reader = CustomPDFReader(chunk=True)
    custom_vector_db = PgVector(
        table_name="langchain_pg_embedding_v2",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=get_embedder()
    )

    # Clear the existing table
    # logger.info("Clearing existing vector database table...")
    # custom_vector_db.drop()
    # logger.info("Table cleared successfully")

    # Create the table if it doesn't exist
    if not custom_vector_db.table_exists():
        logger.info("Creating new vector database table...")
        custom_vector_db.create()
    else:
        logger.info("Using existing vector database table...")

    pdf_knowledge_base = PDFKnowledgeBase(
        path="userGuides",
        vector_db=custom_vector_db,
        reader=custom_reader,
        embedder=get_embedder()
    )

    logger.info("Loading PDF knowledge base...")
    # Load each PDF document
    for pdf_path in Path("userGuides").glob("**/*.pdf"):
        logger.info(f"Processing document: {pdf_path}")
        try:
            # Check if document with this source already exists in metadata
            with custom_vector_db.Session() as sess:
                stmt = select(1).where(
                    custom_vector_db.table.c.meta_data['document_metadata']['source'].astext == pdf_path.name
                ).limit(1)
                result = sess.execute(stmt).first()
                
                if result is not None:
                    logger.info(f"Document {pdf_path.name} already exists in database metadata, skipping...")
                    continue

            # If document doesn't exist, load it
            logger.info(f"Loading new document: {pdf_path}")
            pdf_knowledge_base.load_document(
                path=pdf_path,
                recreate=False,  # Don't recreate the table
                upsert=True      # Use upsert to update existing documents
            )
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            continue

    logger.info("Finished loading PDF knowledge base")

if __name__ == "__main__":
    main() 