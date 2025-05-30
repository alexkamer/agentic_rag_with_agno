from agno.vectordb.chroma import ChromaDb as AgnoChromaDb
from agno.models.azure import AzureOpenAI as AgnoAzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder as AgnoAzureOpenAIEmbedder
from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
import os
import typer
from rich.prompt import Prompt
from typing import Optional

from agno.tools.knowledge import KnowledgeTools
from agno.knowledge.url import UrlKnowledge
from agno.vectordb.lancedb import LanceDb, SearchType

def get_llm():
    # Initialize Azure OpenAI client

    llm = AgnoAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint="https://azure-oai-east2.openai.azure.com",
        azure_deployment="gpt-4-1"
    )
    return llm

def get_embedder():
    embedder = AgnoAzureOpenAIEmbedder(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint="https://azure-oai-east2.openai.azure.com",
        azure_deployment="text-embedding-3-small"
    )
    return embedder


llm = get_llm()
embedder = get_embedder()


vector_db = ChromaDb(
    collection="poker_book",
    path="tmp/chromadb",
    persistent_client=True,
    embedder=embedder
)

agent = Agent(
    llm=llm,
    vector_db=vector_db,
    tools=[KnowledgeTools(vector_db=vector_db)]
)



