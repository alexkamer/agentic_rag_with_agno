from agno.agent import Agent
from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.tools.yfinance import YFinanceTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.document.chunking.document import DocumentChunking
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.shell import ShellTools
from agno.team import Team
from os import getenv
import streamlit as st
import json
import sqlite3
from datetime import datetime

class ConversationDB:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages TEXT NOT NULL
                )
            ''')
            conn.commit()

    def save_conversation(self, title, messages):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (title, messages) VALUES (?, ?)',
                (title, json.dumps(messages))
            )
            conn.commit()
            return cursor.lastrowid

    def update_conversation(self, conversation_id, messages):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE conversations SET messages = ? WHERE id = ?',
                (json.dumps(messages), conversation_id)
            )
            conn.commit()

    def rename_conversation(self, conversation_id, new_title):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE conversations SET title = ? WHERE id = ?',
                (new_title, conversation_id)
            )
            conn.commit()

    def delete_conversation(self, conversation_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            conn.commit()

    def get_conversation(self, conversation_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
            return cursor.fetchone()

    def get_all_conversations(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, title, created_at FROM conversations ORDER BY created_at DESC')
            return cursor.fetchall() 


db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


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




pdf_knowledge_base = PDFKnowledgeBase(
    path="textFiles/raw",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=embedder
    ),
    reader=PDFReader(chunk=True),
    embedder=embedder
)

# Load the pdf_knowledge_base once and comment out after first run
# pdf_knowledge_base.load(recreate=True, upsert=True)


book_agent = Agent(
    name="Book Agent",
    model=llm,
    knowledge=pdf_knowledge_base,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    search_knowledge=True,
    instructions="You are a helpful assistant that can answer questions about the pdfs. Always respond with the book and page number of the answer."

)

shell_agent = Agent(
    model=llm,
    tools=[ShellTools()],
    name="Shell Agent",
    role="You are a shell agent that can execute system commands and return the results.",
    instructions=[
        "You can only answer questions about the shell."
    ]
)

search_agent = Agent(
    name="Search Agent",
    model=llm,
    tools=[GoogleSearchTools()],
    role="You are a search agent that can search the web and return the results.",
    instructions=["Respond with the first 3 results from the web search.",
                  "Summarize the results in a well written response.",
                  "Include links to your sources at the bottom of your response."]
)


stock_agent = Agent(
    name="Stock Agent",
    model=llm,
    tools=[YFinanceTools()],
    role="Analyzes stock market data and provides insights"
)

basic_agent = Agent(
    model=llm,
    role="Basic AI assistant",
)


team = Team(
    model=llm,
    # mode="route",
    mode="coordinate",
    name="Main Team",
    members=[book_agent, shell_agent, search_agent, stock_agent],
    description="A team that can search through book collections and execute system commands.",
    enable_team_history=True,
    instructions=[
        "You are a helpful assistant that can answer questions about the pdfs and execute system commands.",
        "If the question is answered in the pdfs, also return the book and page number of the answer.",
        "If the answer is not in the pdfs, you can use the search agent to find the answer.",
        "If the question is about the stock market, you can use the stock agent to find the answer."
    ],
    show_tool_calls=True
)








# Initialize database
db = ConversationDB()

# Set up the Streamlit page
st.title("🤖 AI Chatbot")
st.write("Welcome! I'm your AI assistant. How can I help you today?")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "show_rename_input" not in st.session_state:
    st.session_state.show_rename_input = False
if "show_delete_confirmation" not in st.session_state:
    st.session_state.show_delete_confirmation = False

# Sidebar for conversation management
with st.sidebar:
    st.header("Conversations")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.rerun()
    
    st.divider()
    
    # Show existing conversations in a dropdown
    conversations = db.get_all_conversations()
    if conversations:
        # Create a dictionary of conversation titles for the dropdown
        conv_options = {f"{title}": conv_id for conv_id, title, _ in conversations}
        
        # Add "Select a conversation" as the default option
        conv_options = {"Select a conversation": None} | conv_options
        
        # Get the current conversation title
        current_title = None
        if st.session_state.current_conversation_id:
            for conv_id, title, _ in conversations:
                if conv_id == st.session_state.current_conversation_id:
                    current_title = title
                    break
        
        # Create the dropdown
        selected_title = st.selectbox(
            "Previous Conversations",
            options=list(conv_options.keys()),
            index=list(conv_options.keys()).index(current_title) if current_title else 0,
            label_visibility="collapsed"
        )
        
        # Handle conversation selection
        selected_id = conv_options[selected_title]
        if selected_id and selected_id != st.session_state.current_conversation_id:
            conversation = db.get_conversation(selected_id)
            if conversation:
                st.session_state.messages = json.loads(conversation[3])
                st.session_state.current_conversation_id = selected_id
                st.rerun()

        # Conversation management buttons
        if st.session_state.current_conversation_id:
            col1, col2 = st.columns(2)
            
            # Rename button
            if col1.button("✏️ Rename", use_container_width=True):
                st.session_state.show_rename_input = True
            
            # Delete button
            if col2.button("🗑️ Delete", use_container_width=True):
                st.session_state.show_delete_confirmation = True
            
            # Rename input
            if st.session_state.show_rename_input:
                new_title = st.text_input(
                    "New title",
                    value=current_title if current_title else "",
                    key="rename_input"
                )
                if new_title and new_title != current_title:
                    db.rename_conversation(st.session_state.current_conversation_id, new_title)
                    st.session_state.show_rename_input = False
                    st.rerun()
                elif st.button("Cancel", use_container_width=True):
                    st.session_state.show_rename_input = False
                    st.rerun()
            
            # Delete confirmation
            if st.session_state.show_delete_confirmation:
                st.warning(f"Are you sure you want to delete '{current_title}'?")
                col1, col2 = st.columns(2)
                if col1.button("✅ Yes, delete", use_container_width=True):
                    db.delete_conversation(st.session_state.current_conversation_id)
                    st.session_state.messages = []
                    st.session_state.current_conversation_id = None
                    st.session_state.show_delete_confirmation = False
                    st.rerun()
                if col2.button("❌ No, cancel", use_container_width=True):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])







if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = team.run(prompt, messages=st.session_state.messages)
            st.write(response.content)

    # team_response = team.run(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Handle conversation saving
    if not st.session_state.current_conversation_id:
        # Generate a title for new conversation
        title_prompt = f"Based on this conversation, generate a short, descriptive title (max 5 words):\n{prompt}\n{response.content}"
        title_response = basic_agent.run(title_prompt)
        title = title_response.content.strip()
        
        # Save new conversation
        conv_id = db.save_conversation(title, st.session_state.messages)
        st.session_state.current_conversation_id = conv_id
    else:
        # Update existing conversation
        db.update_conversation(st.session_state.current_conversation_id, st.session_state.messages)
    
    st.rerun()
    

    # print(team_response)
    # team.print_response(query, stream=True)
# book_agent.print_response("Who was Zeus?", stream=True)




# chef_agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
# chef_agent.print_response("What was my last question?", stream=True)