import os
from os import getenv
from typing import Optional, Dict, List, Any
from agno.utils.log import logger
from agno.models.azure.openai_chat import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.document import Document
from agno.agent import Agent
from agno.tools import Toolkit
from agno.tools.shell import ShellTools
from agno.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from datetime import datetime

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
                # Filter results based on similarity score
                filtered_results = []
                for doc in results:
                    # Convert distance to similarity score (1 - distance)
                    similarity = 1 - doc.meta_data.get('distances', 1.0)
                    
                    # Log the similarity score for debugging
                    logger.info(f"Result similarity score: {similarity:.3f}")
                    
                    # Only include results with high similarity
                    if similarity > 0.4:  # 70% similarity threshold
                        filtered_results.append(doc)
                    else:
                        logger.info(f"Filtered out result with low similarity: {similarity:.3f}")
                
                if filtered_results:
                    logger.info(f"Found {len(filtered_results)} high-quality results")
                    # Convert Document objects to dictionary format
                    formatted_results = {
                        'documents': [[doc.content for doc in filtered_results]],
                        'distances': [[doc.meta_data.get('distances', 0) for doc in filtered_results]],
                        'metadatas': [[doc.meta_data for doc in filtered_results]]
                    }
                    return formatted_results
                else:
                    logger.info("No high-quality results found")
                    return None
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


class BookTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(name="book_tools", tools=[self.search_books], **kwargs)

    def search_books(self, query: str, collection_name: Optional[str] = None, n_results: int = 3) -> str:
        """
        Search across book collections and return formatted results.

        Args:
            query (str): The search query to find relevant information in the books
            collection_name (Optional[str]): Optional name of a specific collection to search. If not provided, searches all collections.
            n_results (int): Number of results to return per collection
        Returns:
            str: Formatted search results with book sources and page numbers
        """
        logger.info(f"Searching books with query: {query}")
        try:
            # Get the actual collection names from the PDF files
            pdf_dir = "textFiles/raw"
            if not os.path.exists(pdf_dir):
                return "Error: Book directory not found."
                
            collections = [os.path.splitext(f)[0] for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            
            if not collections:
                return "No collections found in the database."
            
            if collection_name:
                if collection_name not in collections:
                    return f"Collection '{collection_name}' not found. Available collections: {', '.join(collections)}"
                collections = [collection_name]
            
            all_results = []
            for coll in collections:
                try:
                    searcher = ChromaQuery(coll)
                    results = searcher.query(query, n_results)
                    if results and results.get('documents') and results['documents'][0]:
                        formatted = searcher.format_results(results)
                        if formatted != "No results found.":
                            all_results.append(f"\nResults from {coll}:\n{formatted}")
                except Exception as e:
                    logger.error(f"Error searching collection {coll}: {str(e)}")
                    continue
            
            if not all_results:
                return f"No relevant results found for query: '{query}' in any collection. Available collections: {', '.join(collections)}"
            
            combined_results = "\n".join(all_results)
            print(combined_results)
            return combined_results

        except Exception as e:
            logger.error(f"Error searching books: {str(e)}")
            return f"Error searching books: {str(e)}"

# Create the book tools toolkit
book_tools = BookTools()

def add_qa_to_history(agent: Agent, question: str, answer: str) -> str:
    """Add a question and answer pair to the history.
    
    Args:
        question (str): The question asked
        answer (str): The answer provided
    """
    logger.info(f"Adding Q&A to history. Current state: {agent.team_session_state}")
    
    if "qa_history" not in agent.team_session_state:
        logger.info("Initializing qa_history in session state")
        agent.team_session_state["qa_history"] = []
    
    # Add new QA pair
    qa_pair = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    agent.team_session_state["qa_history"].append(qa_pair)
    
    # Keep only last 10 entries
    if len(agent.team_session_state["qa_history"]) > 10:
        agent.team_session_state["qa_history"].pop(0)
    
    logger.info(f"Updated history. New size: {len(agent.team_session_state['qa_history'])}")
    return f"Added Q&A to history. Current history size: {len(agent.team_session_state['qa_history'])}"

def get_qa_history(agent: Agent) -> str:
    """Get the current Q&A history.
    
    Returns:
        str: Formatted string of recent Q&A pairs
    """
    logger.info(f"Getting Q&A history. Current state: {agent.team_session_state}")
    
    if "qa_history" not in agent.team_session_state:
        logger.info("No qa_history in session state, initializing")
        agent.team_session_state["qa_history"] = []
        return "No previous questions and answers in history."
    
    if not agent.team_session_state["qa_history"]:
        logger.info("qa_history is empty")
        return "No previous questions and answers in history."
    
    history = agent.team_session_state["qa_history"]
    formatted_history = []
    
    for i, qa in enumerate(history, 1):
        formatted_history.append(
            f"Q&A Pair {i}:\n"
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}\n"
            f"Time: {qa['timestamp']}\n"
        )
    
    logger.info(f"Found {len(history)} Q&A pairs in history")
    return "\n".join(formatted_history)

def store_qa_pair(agent: Agent, question: str, answer: str) -> str:
    """Store a question and answer pair in the history.
    
    Args:
        question (str): The question asked
        answer (str): The answer provided
    """
    logger.info(f"Storing Q&A pair. Question: {question}")
    return add_qa_to_history(agent, question, answer)

# Update the book agent to use the toolkit and include history tools
book_agent = Agent(
    model=llm,
    tools=[book_tools, GoogleSearchTools(), add_qa_to_history, get_qa_history, store_qa_pair],
    name="Book Agent",
    role="You are a book agent that can search through the book collections and return the results.",
    instructions=[
        "Follow this exact process for answering questions:",
        "1. First, check if a similar question was asked before using get_qa_history",
        "2. If a similar question exists in history, use that answer as a reference",
        "3. If no similar question exists or the answer needs updating:",
        "   a. Search the book collections for the answer",
        "   b. If found in books, provide answer with book name and page number",
        "   c. If not in books, try general knowledge",
        "   d. If still no answer, use Google search",
        "4. After providing an answer, ALWAYS use store_qa_pair to store the question and your answer",
        "5. Always be clear about which source you're using (history, books, general knowledge, or web search)",
        "6. If you cannot find an answer through any method, clearly state that",
        "",
        "IMPORTANT: You MUST use store_qa_pair after EVERY answer you provide, no exceptions."
    ],
    show_tool_calls=True,
    markdown=True,
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
    model=llm,
    tools=[GoogleSearchTools()],
    name="Search Agent",
    role="You are a search agent that can search the web and return the results.",
    instructions=["Respond with the first 3 results from the web search.",
                  "Summarize the results in a well written response.",
                  "Include links to your sources at the bottom of your response."]
)

# Update the team configuration
team = Team(
    model=llm,
    mode="route",
    name="Book and Shell Team",
    members=[book_agent, shell_agent, search_agent],
    description="A team that can search through book collections and execute system commands.",
    team_session_state={"qa_history": []},
    instructions=[
        "You are a routing team that directs questions to the appropriate agent.",
        "For each question:",
        "1. First, analyze the question to determine which agent should handle it",
        "2. Then, forward the EXACT question to that agent",
        "3. Do not modify or change the question when forwarding it",
        "4. If the question is about past questions, check the qa_history in team_session_state",
        "",
        "Route to Book Agent for:",
        "- Questions about book content",
        "- Information from the book collections",
        "- Questions about past interactions",
        "",
        "Route to Shell Agent for ONLY:",
        "- System commands",
        "- File operations",
        "- System information",
        "",
        "Route to Search Agent for:",
        "- If the question cannot be found in the book collections or through general knowledge",
        "- Questions your common knowledge isn't trained on",
        "- Anything that is not general knowledge or related to Agents explained before",
        "",
        "Important: Always forward the original question exactly as received.",
        "Do not modify or change the question when forwarding it to an agent.",
        "The past questions and answers are stored in team_session_state['qa_history'] and can be used to answer questions.",
        "IMPORTANT: Make sure the Book Agent stores each Q&A pair using store_qa_pair after answering."
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

if __name__ == "__main__":
    print("\nWelcome to the Book and Shell Team!")
    print("You can ask questions about books or system commands.")
    print("Type 'exit' to quit.\n")
    
    # Initialize the main team with session state
    main_team = Team(
        model=llm,
        mode="route",
        name="Book, Shell, and Search Team",
        members=[book_agent, shell_agent, search_agent],
        description="A team that can search through book collections and execute system commands.",
        team_session_state={"qa_history": []},
        instructions=team.instructions,
        show_tool_calls=True,
        markdown=True,
        debug_mode=True
    )
    
    while True:
        try:
            query = input("Enter a question: ").strip()
            if query.lower() == "exit":
                print("\nGoodbye!")
                break
                
            if not query:
                print("Please enter a question.")
                continue
                
            print("\nProcessing your question...")
            logger.info(f"Current session state before query: {main_team.team_session_state}")
            
            # Create a new team instance for each query with the shared session state
            current_team = Team(
                model=llm,
                mode="route",
                name="Book, Shell, and Search Team",
                members=[book_agent, shell_agent, search_agent],
                description="A team that can search through book collections and execute system commands.",
                team_session_state=main_team.team_session_state.copy(),  # Create a copy of the session state
                instructions=team.instructions,
                show_tool_calls=True,
                markdown=True,
                debug_mode=True
            )
            
            team_data = current_team.run(query)
            
            # Update the main team's session state
            main_team.team_session_state = current_team.team_session_state
            logger.info(f"Updated session state after query: {main_team.team_session_state}")
            
            print(team_data.content)
            print("\n" + "-"*50 + "\n")  # Add separator between responses
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.\n")


