import os
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class AdvancedRagChatService:
    def __init__(self):
        self.gemini_api_key = os.getenv("geminiAPI")
        if not self.gemini_api_key:
            raise RuntimeError("geminiAPI not found in .env")
        
        self.memory = MemorySaver()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=self.gemini_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=self.gemini_api_key)

    def _get_or_create_schema_vector_db(self, sql_db: SQLDatabase, connection_id: int):
        # We store the vector DB locally per database connection
        persist_directory = f"./chroma_db_connections/connection_{connection_id}"
        
        # Ensure the base directory exists
        os.makedirs("./chroma_db_connections", exist_ok=True)
        
        # Check if the DB was already created and persisted
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            return Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            
        print(f"Extracting schema for database connection {connection_id} and building Vector DB...")
        # If it doesn't exist, we extract schema and build it
        tables = sql_db.get_usable_table_names()
        docs = []
        for table in tables:
            # We get the exact schema for the table
            try:
                schema_info = sql_db.get_table_info_no_throw([table])
                docs.append(f"Table name: {table}\nSchema Details: {schema_info}")
            except Exception as e:
                print(f"Warning: Could not extract schema for table {table}: {e}")
            
        # Create Vector DB from the docs
        if not docs:
            docs = ["No accessible tables found in this database."]
            
        vector_db = Chroma.from_texts(
            texts=docs, 
            embedding=self.embeddings, 
            persist_directory=persist_directory
        )
        return vector_db

    def process_chat(self, user_id: int, db_type: str, host: str, port: int, db_name: str, username: str, password: str, connection_id: int, message: str) -> str:
        # Construct DB URI dynamically based on user's connection details
        db_type = db_type.lower()
        if db_type == "postgresql":
            db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}?sslmode=require"
        elif db_type == "mysql":
            db_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
        elif db_type == "clickhouse":
            db_uri = f"clickhouse+connect://{username}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError("Unsupported database type")

        # 1. Initialize DB Connection
        sql_db = SQLDatabase.from_uri(db_uri)

        # 2. RAG Retrieve Relevant Tables!
        vector_db = self._get_or_create_schema_vector_db(sql_db, connection_id)
        
        # We retrieve the top 5 tables that seem most relevant to the user's question
        retriever = vector_db.as_retriever(search_kwargs={"k": 5}) 
        relevant_docs = retriever.invoke(message)
        relevant_schema = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 3. Setup Agent Tools
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.llm)
        tools = toolkit.get_tools()
        
        # 4. Dynamic System Prompt with RAG Context
        system_prompt = f"""You are a helpful database assistant named Koli.
You are querying a {db_type} database.

Based on the user's question, I have already searched the database schema and found the following potentially relevant tables:

{relevant_schema}

INSTRUCTIONS:
1. ONLY utilize the tables provided in the schema above to answer the user's question. If you absolutely need a table that isn't listed, you can use your tools to discover it, but heavily prefer the ones listed above.
2. Given the input question, create a syntactically correct SQL query to run.
3. Look at the results of the query and return the answer.
4. Limit your query to 10 results unless specified otherwise.
5. NEVER make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).
6. Explain the result to the user clearly.
"""

        # 5. Create Agent
        agent_executor = create_react_agent(
            self.llm,
            tools,
            prompt=system_prompt,
            checkpointer=self.memory,
        )

        # 6. Session Management
        thread_id = f"user_{user_id}_db_{connection_id}"
        config = {"configurable": {"thread_id": thread_id}}

        # 7. Execute Return
        response = agent_executor.invoke({"messages": [("user", message)]}, config=config)
        
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
chat_service = AdvancedRagChatService()
