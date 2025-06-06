from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.agent import Agent
from agno.tools.postgres import PostgresTools
from os import getenv
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

postgres_tools = PostgresTools(
    host="localhost",
    port=5432,
    db_name="smartagentdb",
    user="alexkamer",
    inspect_queries=True  # This will show all SQL commands being executed
)

connection = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="smartagentdb",
    user="alexkamer"
)

cursor = connection.cursor()

cursor.execute("""
select sce.*, cf.* 
from srf_core_events sce  
join cpu_families cf on sce.cpu_family_id = cf.id """)
results = cursor.fetchall()

# Get column names from the models
srf_columns = [c.name for c in SRFCoreEvent.__table__.columns]
cpu_family_columns = [c.name for c in CPUFamily.__table__.columns]

# Combine the column names
all_columns = srf_columns + cpu_family_columns

# Create DataFrame with proper column names
df = pd.DataFrame(results, columns=all_columns)

# Create a function to combine relevant fields into a meaningful text chunk
def create_text_chunk(row):
    # Convert row values to native Python types
    row_dict = row.to_dict()
    
    # Combine event information
    event_info = f"""
Event Name: {row_dict['EventName']}
Event Code: {row_dict['EventCode']}
UMask: {row_dict['UMask']}
Brief Description: {row_dict['BriefDescription']}
Public Description: {row_dict['PublicDescription']}

CPU Family: {row_dict['name']}
Architecture: {row_dict['architecture_name']}
Generation: {row_dict['generation']}
Microarchitecture: {row_dict['microarchitecture']}

Technical Details:
- Counter: {row_dict['Counter']}
- Sample After Value: {row_dict['SampleAfterValue']}
- MSR Index: {row_dict['MSRIndex']}
- MSR Value: {row_dict['MSRValue']}
- Offcore: {row_dict['Offcore']}
- Deprecated: {row_dict['Deprecated']}

AI-Generated Content:
Overview: {json.dumps(row_dict['overview'], indent=2) if row_dict['overview'] else 'N/A'}
Contextual Explanation: {json.dumps(row_dict['contextual_explanation'], indent=2) if row_dict['contextual_explanation'] else 'N/A'}
Real World Use Cases: {json.dumps(row_dict['real_world_use_cases'], indent=2) if row_dict['real_world_use_cases'] else 'N/A'}
Code Examples: {json.dumps(row_dict['code_snippet_examples'], indent=2) if row_dict['code_snippet_examples'] else 'N/A'}
"""
    return event_info.strip()




# Create a list of documents for RAG
documents = []
for _, row in df.iterrows():
    row_dict = row.to_dict()
    doc = {
        'text': create_text_chunk(row),
        'metadata': {
            'event_id': int(row_dict['id']),  # Convert to native Python int
            'event_name': str(row_dict['EventName']),  # Convert to native Python str
            'event_code': str(row_dict['EventCode']),
            'cpu_family': str(row_dict['name']),
            'architecture': str(row_dict['architecture_name']),
            'generation': str(row_dict['generation'])
        }
    }
    documents.append(doc)

# Print example of first document
print("\nExample document for RAG:")
print(json.dumps(documents[0], indent=2))

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



# # Drop the table if it exists to ensure clean creation
# vector_db.drop()

# # Create a custom table without primary key
# with vector_db.Session() as sess, sess.begin():
#     # Create schema if it doesn't exist
#     sess.execute(text("CREATE SCHEMA IF NOT EXISTS ai;"))
    
#     # Create the table without primary key
#     create_table_sql = text("""
#     CREATE TABLE ai.event_embeddings (
#         id TEXT,
#         name TEXT,
#         meta_data JSONB DEFAULT '{}'::jsonb,
#         filters JSONB,
#         content TEXT,
#         embedding vector(1536),
#         usage JSONB,
#         created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
#         updated_at TIMESTAMP WITH TIME ZONE,
#         content_hash TEXT
#     );
#     """)
#     sess.execute(create_table_sql)

# # Convert our documents to agno Document format
# agno_documents = []
# for doc in documents:
#     agno_doc = AgnoDocument(
#         id=str(doc['metadata']['event_id']),  # Convert to string as PgVector expects string IDs
#         name=doc['metadata']['event_name'],
#         content=doc['text'],
#         meta_data=doc['metadata']
#     )
#     agno_documents.append(agno_doc)

# # Store the documents in the vector database
# print("\nStoring documents in vector database...")
# vector_db.insert(agno_documents)



# Verify the documents were stored
count = vector_db.get_count()
print(f"\nStored {count} documents in vector database")

# Example search
print("\nTesting search functionality...")
# search_results = vector_db.search("What are real world use cases for Machine clears fp assist?", limit=3)
search_results = vector_db.search("What counts the number of floating point operations?", limit=3)

for i, result in enumerate(search_results, 1):
    print(f"\nResult {i}:")
    print(f"Event Name: {result.name}")
    print(f"Content: {result.content[:200]}...")  # Print first 200 chars of content


