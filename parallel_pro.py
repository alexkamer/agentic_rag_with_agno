from agno.agent import Agent
from agno.tools.postgres import PostgresTools
from agno.models.azure.openai_chat import AzureOpenAI
from os import getenv
from agno.embedder.azure_openai import AzureOpenAIEmbedder
import pandas as pd
import numpy as np
import json
import ast
import psycopg2
from agno.vectordb.pgvector import PgVector, SearchType, HNSW, Distance
from pprint import pprint
from models_events import SRFCoreEvent
from models_cpu_families import CPUFamily
from agno.document import Document as AgnoDocument
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import asyncio

postgres_tools = PostgresTools(
    host="localhost",
    port=5432,
    db_name="smartagentdb",
    user="alexkamer",
    inspect_queries=True  # This will show all SQL commands being executed

)

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




sql_system_message =[
        "You are an assistant who has access to a database tool and can use it to answer questions.",
        "If the question is related to cpu events or metrics, you will use the database tool to answer the question.",
       # "Every question asked by the user should be to answered by the database, you will never use your own knowledge to answer the question",
        #"You are a domain-aware assistant for CPU performance event and metric information, specialized in Intel perfmon events and metrics. Your goal is to help performance engineers answer questions about events and metrics using database access.",
        "Guidelines:",
        "Prior to using the database tool, you must first understand the intent of the user's question. Carefully read the user's question to determine whether they are asking about:",
        "a. A specific event",
        "b. A specific metric",
        "c. The set of available events or metrics",
        "d. Details, formulas, configuration, or usage guidance",
        "2. Information Sufficiency Check: Before generating or executing a database query, ALWAYS verify that you have all required information to uniquely identify the event or metric or to fulfill the user's request.",
        "a. Required information may include: event or metric name, CPU family, event/metric type, and comparison entities.",
        "b. If any crucial information is missing, DO NOT generate or run a query—instead, ask a followup question to elicit the required details from the user.",
        "3. Database Query Generation: ",
        "a. Only after gathering sufficient information, generate an unambiguous query based on the user's intent and database schema. Pass this query to the SQL tool for execution.",
        "4. Follow-up Handling:",
        "If information is missing, ask a concise, specific follow-up question to obtain it—for example:",
        "'Which event or metric would you like details about?'",
        "'Which CPU family are you referring to?'",
        "'Could you specify the metric or event types you are interested in?'",
        "5. Schema Understanding:",
        "You are aware of the database schema, which includes tables for events and metrics, each keyed by CPU family, event/metric type, and name.",
        "6. Clarification Preference:",
        "When in doubt, clarify before taking action.",
        "Examples:",
        "User asks: 'What does loads_per_instr mean?' You respond: 'Which CPU family do you want to know about for the metric loads_per_instr?'",
        "User asks: 'List all events for GNR.' You have enough information to run the query.",
        "User asks: 'How do I configure MEM_INST_RETIRED.ALL_LOADS?' You respond: 'Which CPU family are you interested in?'",
        "User asks: 'What is the formula for ________ ?' You respond: 'Which CPU family are you interested in? or which metric are you interested in?'",
        "Policy:",
        "NEVER generate or execute a tool invocation unless you are certain you have all the critical information required for a correct and specific result. ALWAYS ask clarifying questions if information is incomplete or ambiguous. Ex. if someone asks for a formula, ensure that they also gave the specific CPU family and metric type aswell.",
        # "If the user asks for a formula, do no display the aliases, make sure to display the base formula as it is in the database.",
        "If the user asks for a formula, do no display the aliases, make sure to display the AI_formula with the events Name substituted in for the aliases. When referencing the events Names, include the event Name and a description of the event.",

        "Guidance for building the correct query:",
        ## inform to list the tables, then query the cpu_families table to get the names of the correct event or metric table, then describe the table to get the schema, then build the query based on the schema
        "1. First list the tables.",
        "2. Then query the cpu_families table to get the names of the correct event or metric table.",
        "3. Then describe the table to get the schema.",
        "4. Then build the query based on the schema.",
        "5. If the query is not successful, first use ILIKE and try to see if the user meant a specific column, with columns similar",
        "6. If the query is not successful, ask the user for the missing information",
        "7. If the query is successful, display the result",

    ]

sql_system_message = "\n".join(sql_system_message)

sql_query_agent = Agent(
    tools=[postgres_tools],
    model=llm,
    name="SQL Query Agent",
    system_message=sql_system_message,
    show_tool_calls=True,
    add_history_to_messages=True,
    num_history_responses=10,
)


# Initialize PgVector for storing embeddings
vector_db = PgVector(
    table_name="event_embeddings",
    schema="ai",
    db_url="postgresql://alexkamer@localhost:5432/smartagentdb",
    embedder=embedder,
    search_type=SearchType.vector,
    vector_index=HNSW(
        m=16,  # Number of connections per layer
        ef_construction=64,  # Size of the dynamic candidate list
        ef_search=40  # Size of the dynamic candidate list during search
    ),
    distance=Distance.cosine
)


# Verify the documents were stored
count = vector_db.get_count()
print(f"\nStored {count} documents in vector database")

# Example search
print("\nTesting search functionality...")
# search_results = vector_db.search("What are real world use cases for Machine clears fp assist?", limit=3)
query = "What counts the number of floating point operations?"



search_results = vector_db.search(query, limit=3)

for i, result in enumerate(search_results, 1):
    print(f"\nResult {i}:")
    print(f"Event Name: {result.name}")
    print(f"Content: {result.content[:200]}...")  # Print first 200 chars of content

sql_query_agent.print_response(query, show_tool_calls=True)

def get_vector_search_results(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Get results from vector search."""
    search_results = vector_db.search(query, limit=limit)
    return [{
        'source': 'vector_search',
        'event_name': result.name,
        'content': result.content,
        'metadata': result.meta_data
    } for result in search_results]

def get_sql_search_results(query: str) -> Dict[str, Any]:
    """Get results from SQL query agent."""
    response = sql_query_agent.run(query)

    return {
        'source': 'sql_query',
        'response': response.content,
        'tool_calls': response.tool_calls if hasattr(response, 'tool_calls') else []
    }

def parallel_search(query: str, limit: int = 3) -> Dict[str, Any]:
    """Perform parallel search using both vector and SQL methods."""
    # Run both searches
    vector_results = get_vector_search_results(query, limit)
    sql_results = get_sql_search_results(query)
    
    return {
        'vector_results': vector_results,
        'sql_results': sql_results,
        'query': query
    }

def format_combined_results(results: Dict[str, Any]) -> str:
    """Format the combined results in a readable way."""
    output = []
    output.append(f"\nQuery: {results['query']}")
    
    # Format vector search results
    output.append("\n=== Vector Search Results ===")
    for i, result in enumerate(results['vector_results'], 1):
        output.append(f"\nResult {i}:")
        output.append(f"Event Name: {result['event_name']}")
        output.append(f"Content: {result['content'][:200]}...")
    
    # Format SQL results
    output.append("\n=== SQL Query Results ===")
    output.append(f"Response: {results['sql_results']['response']}")
    if results['sql_results']['tool_calls']:
        output.append("\nSQL Queries Executed:")
        for tool_call in results['sql_results']['tool_calls']:
            output.append(f"- {tool_call}")
    
    return "\n".join(output)

def main():
    query = ""
    while query != "exit":
        query = input("Enter a query: ")
        results = parallel_search(query)
        print(format_combined_results(results))

# Run the example
if __name__ == "__main__":
    main()





