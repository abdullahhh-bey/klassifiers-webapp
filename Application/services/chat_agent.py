import os
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class ChatService:
    def __init__(self):
        self.gemini_api_key = os.getenv("geminiAPI")
        if not self.gemini_api_key:
            raise RuntimeError("geminiAPI not found in .env")
        
        self.memory = MemorySaver()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=self.gemini_api_key)

    def process_chat(self, user_id: int, db_type: str, host: str, port: int, db_name: str, username: str, password: str, connection_id: int, message: str) -> str:
        # Construct DB URI dynamically based on user's connection details
        db_type = db_type.lower()
        if db_type == "postgresql":
            db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}?sslmode=require"
        elif db_type == "mysql":
            db_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
        elif db_type == "clickhouse":
            # Using clickhouse-connect for clickhouse
            db_uri = f"clickhouse+connect://{username}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError("Unsupported database type")

        # 1. Initialize DB Toolkit
        sql_db = SQLDatabase.from_uri(db_uri)

        # 2. Setup Agent Tools
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.llm)
        tools = toolkit.get_tools()
        
        # 3. Agent System Prompt
        system_prompt = f"""You are a helpful database assistant. Your name is Koli, made by Abdullah Amir.
You are querying a {db_type} database.
Given an input question, create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
If you get an error while executing a query, rewrite the query and try again.
If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
NEVER make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Explain the result to the user clearly."""

        # 4. Create LangGraph React Agent
        agent_executor = create_react_agent(
            self.llm,
            tools,
            prompt=system_prompt,
            checkpointer=self.memory,
        )

        # 5. Session Management (Per Database Connection)
        thread_id = f"user_{user_id}_db_{connection_id}"
        config = {"configurable": {"thread_id": thread_id}}

        # 6. Run Agent
        response = agent_executor.invoke(
            {"messages": [("user", message)]}, 
            config=config
        )

        final_msg_raw = response["messages"][-1].content
        if isinstance(final_msg_raw, list):
            parts = []
            for part in final_msg_raw:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            final_msg = "".join(parts)
        else:
            final_msg = str(final_msg_raw)

        return final_msg

# Instantiate a single instance of the service
chat_service = ChatService()
