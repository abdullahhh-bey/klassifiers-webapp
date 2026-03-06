import os
import logging
from typing import List, Dict, Any, Union

from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRagChatService:
    """
    A service class that implements a Retrieval-Augmented Generation (RAG) agent
    for querying SQL databases using LangChain and HuggingFace Embeddings.
    """
    def __init__(self):
        self._initialize_keys()
        self._setup_models()
        self.memory = MemorySaver()

    def _initialize_keys(self) -> None:
        """Load and validate all required API keys from environment."""
        self.gemini_api_key = os.getenv("geminiAPI")
        if not self.gemini_api_key:
            logger.error("geminiAPI key not found in environment variables.")
            raise RuntimeError("geminiAPI key is missing.")
            
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.hf_api_key:
            logger.error("HUGGINGFACE_API_KEY not found in environment variables.")
            raise RuntimeError("HUGGINGFACE_API_KEY is missing. Get one at huggingface.co/settings/tokens")

    def _setup_models(self) -> None:
        """Initialize the LLM and Embedding models."""
        # Main LLM for reasoning and SQL generation (Google Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=self.gemini_api_key
        )
        
        # HuggingFace Embeddings for vectorizing the database schema
        # We use a fast, lightweight sentence-transformer model via HuggingFace Inference API
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=self.hf_api_key,
        )

    def _get_db_uri(self, db_type: str, host: str, port: int, db_name: str, username: str, password: str) -> str:
        """Construct the SQLAlchemy database URI securely."""
        db_type = db_type.lower()
        if db_type == "postgresql":
            return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}?sslmode=require"
        elif db_type == "mysql":
            return f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
        elif db_type == "clickhouse":
            return f"clickhouse+connect://{username}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _get_or_create_schema_vector_db(self, sql_db: SQLDatabase, connection_id: int) -> Chroma:
        """
        Extracts the schema from the SQL database and embeds it into a local ChromaDB.
        If it already exists, it loads the existing ChromaDB to save time.
        """
        persist_directory = f"./chroma_db_connections/connection_{connection_id}"
        os.makedirs("./chroma_db_connections", exist_ok=True)
        
        # 1. Load existing DB if available
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info(f"Loading existing Vector DB for connection {connection_id}")
            return Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            
        # 2. Extract schema and build new DB
        logger.info(f"Extracting schema and building new Vector DB for connection {connection_id}...")
        tables = sql_db.get_usable_table_names()
        docs = []
        
        for table in tables:
            try:
                schema_info = sql_db.get_table_info_no_throw([table])
                docs.append(f"Table name: {table}\nSchema Details: {schema_info}")
            except Exception as e:
                logger.warning(f"Could not extract schema for table {table}: {e}")
            
        if not docs:
            docs = ["No accessible tables found in this database."]
            
        # 3. Create and persist the Vector DB
        vector_db = Chroma.from_texts(
            texts=docs, 
            embedding=self.embeddings, 
            persist_directory=persist_directory
        )
        return vector_db

    def _get_relevant_schema(self, vector_db: Chroma, query: str, k: int = 5) -> str:
        """Retrieve the top k most relevant tables for the user's query."""
        retriever = vector_db.as_retriever(search_kwargs={"k": k}) 
        relevant_docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in relevant_docs])

    def _create_system_prompt(self, db_type: str, relevant_schema: str) -> str:
        """Create the dynamic system prompt guiding the LLM."""
        return f"""You are a helpful database assistant named Koli.
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

    def process_chat(self, user_id: int, db_type: str, host: str, port: int, db_name: str, username: str, password: str, connection_id: int, message: str) -> str:
        """
        Main entry point for processing a chat message against a specific database.
        """
        # 1. Connect to Database
        db_uri = self._get_db_uri(db_type, host, port, db_name, username, password)
        sql_db = SQLDatabase.from_uri(db_uri)

        # 2. Retrieve Relevant Schema (RAG)
        vector_db = self._get_or_create_schema_vector_db(sql_db, connection_id)
        relevant_schema = self._get_relevant_schema(vector_db, message)

        # 3. Setup Tools & Prompt
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.llm)
        tools = toolkit.get_tools()
        system_prompt = self._create_system_prompt(db_type, relevant_schema)

        # 4. Initialize Agent
        agent_executor = create_react_agent(
            self.llm,
            tools,
            prompt=system_prompt,
            checkpointer=self.memory,
        )

        # 5. Execute Agent with Session Thread
        thread_id = f"user_{user_id}_db_{connection_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            response = agent_executor.invoke({"messages": [("user", message)]}, config=config)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"An error occurred while processing your request: {e}"

        # 6. Parse and Return Response
        final_msg_raw = response["messages"][-1].content
        if isinstance(final_msg_raw, list):
            parts = []
            for part in final_msg_raw:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            return "".join(parts)
            
        return str(final_msg_raw)

# Instantiate a single instance of the service
chat_service = AdvancedRagChatService()
