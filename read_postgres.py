from agno.agent import Agent
from agno.tools.postgres import PostgresTools
from agno.models.azure.openai_chat import AzureOpenAI
from os import getenv

def get_llm():
    return AzureOpenAI(
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT", "https://azure-oai-east2.openai.azure.com/"),
        azure_deployment="gpt-4-1"
    )
llm = get_llm()
postgres_tools = PostgresTools(
    host="localhost",
    port=5432,
    db_name="smartagentdb",
    user="alexkamer",
    inspect_queries=True  # This will show all SQL commands being executed

)

system_message =[
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
        "Guidance for building the correct query:",
        ## inform to list the tables, then query the cpu_families table to get the names of the correct event or metric table, then describe the table to get the schema, then build the query based on the schema
        "1. First list the tables.",
        "2. Then query the cpu_families table to get the names of the correct event or metric table.",
        "3. Then describe the table to get the schema.",
        "4. Then build the query based on the schema.",
        "5. If the query is not successful, ask the user for the missing information",
        "6. If the query is successful, display the result",


        # "",
        # "Database Schema:",
        # "",
        # "Events Table (events):",
        # "CREATE TABLE events (",
        # "    id SERIAL PRIMARY KEY,",
        # "    cpu_family VARCHAR(50) NOT NULL,",
        # "    event_type VARCHAR(50) NOT NULL,",
        # "    event_name VARCHAR(100) NOT NULL,",
        # "    description TEXT,",
        # "    formula TEXT,",
        # "    configuration TEXT,",
        # "    UNIQUE(cpu_family, event_type, event_name)",
        # ");",
        # "",
        # "Metrics Table (metrics):",
        # "CREATE TABLE metrics (",
        # "    id SERIAL PRIMARY KEY,",
        # "    cpu_family VARCHAR(50) NOT NULL,",
        # "    metric_type VARCHAR(50) NOT NULL,",
        # "    metric_name VARCHAR(100) NOT NULL,",
        # "    description TEXT,",
        # "    formula TEXT,",
        # "    configuration TEXT,",
        # "    UNIQUE(cpu_family, metric_type, metric_name)",
        # ");",
        # "",
        # "Common queries:",
        # "1. Get event details: SELECT * FROM events WHERE cpu_family = 'GRR' AND event_type = 'base' AND event_name = 'MEM_INST_RETIRED.ALL_LOADS';",
        # "2. Get metric details: SELECT * FROM metrics WHERE cpu_family = 'GRR' AND metric_type = 'base' AND metric_name = 'cpu_operating_frequency';",
        # "3. List all events for a CPU family: SELECT event_name, description FROM events WHERE cpu_family = 'GRR' AND event_type = 'base';",
        # "4. List all metrics for a CPU family: SELECT metric_name, description FROM metrics WHERE cpu_family = 'GRR' AND metric_type = 'base';"
    ]

system_message = "\n".join(system_message)

query_agent = Agent(
    tools=[postgres_tools],
    model=llm,
    name="Query Agent",
    system_message=system_message,
    show_tool_calls=True,
    add_history_to_messages=True,
    num_history_responses=10,
)

query = ""

while query != "exit":
    query = input("Enter a query: ")
    query_agent.print_response(query, stream=True, stream_intermediate_steps=True, show_tool_calls=True)
    print("--------------------------------")

    # print(    [
    #     m.model_dump(include={"role", "content"})
    #     for m in query_agent.get_messages_for_session()
    # ])


# query_agent.print_response("What does the cpu_operating_frequency metric indicate?")
# query_agent.print_response("What is the formula for CPU operating frequency?", stream=True, stream_intermediate_steps=True)
# query_agent.print_response("What is the formula for CPU operating frequency in the GRR CPU Family with a base metric type?")