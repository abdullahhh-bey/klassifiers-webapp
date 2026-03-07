import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_postgres import PGVector                     # Supabase uses Postgres
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# Load from environment variables (recommended) or hard-code for local testing.
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── LLM ──────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("geminiAPI", "")

    # ── Embeddings ───────────────────────────────────────────────────────────
    HF_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HF_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Supabase Vector Store ─────────────────────────────────────────────────
    SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")
    VECTOR_TABLE_NAME: str = "schema_embeddings"

    # ── Agent Behaviour ───────────────────────────────────────────────────────
    RELEVANT_TABLES_K: int = 5          
    QUERY_RESULT_LIMIT: int = 10        
    AGENT_NAME: str = "Koli"            

    # ── Supported DB drivers ──────────────────────────────────────────────────
    SUPPORTED_DB_TYPES = {"postgresql", "mysql", "clickhouse"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(cfg: Config) -> None:
    """Raise early with a clear message if any required key is missing."""
    missing = []
    if not cfg.GEMINI_API_KEY:
        missing.append("geminiAPI (in .env)")
    if not cfg.HF_API_KEY:
        missing.append("HUGGINGFACE_API_KEY (in .env)")
    if not cfg.SUPABASE_DB_URL:
        missing.append("SUPABASE_DB_URL (in .env)")
    if missing:
        raise RuntimeError(
            "Missing required configuration:\n  " + "\n  ".join(missing)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATABASE URI BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_db_uri(
    db_type: str, host: str, port: int,
    db_name: str, username: str, password: str
) -> str:
    """
    Construct an SQLAlchemy-compatible URI for the TARGET database being queried.
    """
    db_type = db_type.lower()
    if db_type not in Config.SUPPORTED_DB_TYPES:
        raise ValueError(
            f"Unsupported DB type '{db_type}'. "
            f"Supported: {Config.SUPPORTED_DB_TYPES}"
        )

    drivers = {
        "postgresql": f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}?sslmode=require",
        "mysql":      f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}",
        "clickhouse": f"clickhouse+connect://{username}:{password}@{host}:{port}/{db_name}",
    }
    return drivers[db_type]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VECTOR STORE (Supabase pgvector)
# ─────────────────────────────────────────────────────────────────────────────

def get_or_build_vector_store(
    sql_db: SQLDatabase,
    connection_id: int,
    embeddings: HuggingFaceEndpointEmbeddings,
    cfg: Config,
) -> PGVector:
    """
    Extract the schema from the SQL database and persist into Supabase pgvector.
    On subsequent calls, load existing collection directly.
    """
    collection_name = f"schema_conn_{connection_id}"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=cfg.SUPABASE_DB_URL,
        use_jsonb=True,
    )

    try:
        existing = vector_store.similarity_search("test", k=1)
        if existing:
            logger.info(f"[VectorDB] Loaded existing schema index for connection {connection_id}")
            return vector_store
    except Exception:
        pass

    logger.info(f"[VectorDB] Building schema index for connection {connection_id}...")
    tables = sql_db.get_usable_table_names()
    docs = []

    for table in tables:
        try:
            schema_info = sql_db.get_table_info_no_throw([table])
            docs.append(
                Document(
                    page_content=f"Table: {table}\n{schema_info}",
                    metadata={"table_name": table, "connection_id": str(connection_id)},
                )
            )
        except Exception as e:
            logger.warning(f"[VectorDB] Skipping table '{table}': {e}")

    if not docs:
        docs = [Document(page_content="No accessible tables found.", metadata={})]

    vector_store.add_documents(docs)
    logger.info(f"[VectorDB] Indexed {len(docs)} table(s) into Supabase.")
    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_relevant_schema(
    vector_store: PGVector, query: str, k: int
) -> str:
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in relevant_docs)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(db_type: str, relevant_schema: str, cfg: Config) -> str:
    return f"""You are a database assistant named {cfg.AGENT_NAME}.
You are connected to a {db_type.upper()} database.

Relevant table schemas:
{relevant_schema}

RULES:
1. ONLY use the tables listed above. Do not invent names.
2. Write {db_type.upper()} SQL.
3. LIMIT results to {cfg.QUERY_RESULT_LIMIT}.
4. NO DML (INSERT, UPDATE, DELETE).
5. Explain the result clearly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN AGENT SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class RagSqlAgentService:
    def __init__(self):
        self.cfg = Config()
        validate_config(self.cfg)
        self._setup_models()
        self.memory = MemorySaver()
        logger.info("[Service] RagSqlAgentService initialised successfully.")

    def _setup_models(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.cfg.GEMINI_API_KEY,
            temperature=0, 
        )
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.cfg.HF_EMBED_MODEL,
            task="feature-extraction",
            huggingfacehub_api_token=self.cfg.HF_API_KEY,
        )

    def chat(
        self,
        user_id: int,
        connection_id: int,
        db_type: str,
        host: str,
        port: int,
        db_name: str,
        username: str,
        password: str,
        message: str,
    ) -> str:
        db_uri = build_db_uri(db_type, host, port, db_name, username, password)
        sql_db = SQLDatabase.from_uri(db_uri)

        vector_store = get_or_build_vector_store(
            sql_db, connection_id, self.embeddings, self.cfg
        )
        relevant_schema = retrieve_relevant_schema(
            vector_store, message, k=self.cfg.RELEVANT_TABLES_K
        )

        system_prompt = build_system_prompt(db_type, relevant_schema, self.cfg)
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.llm)
        tools = toolkit.get_tools()

        agent = create_react_agent(
            self.llm,
            tools,
            prompt=system_prompt,
            checkpointer=self.memory,
        )

        thread_id = f"user_{user_id}_conn_{connection_id}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            response = agent.invoke(
                {"messages": [("user", message)]},
                config=config,
            )
        except Exception as e:
            logger.error(f"[Agent] Execution failed: {e}", exc_info=True)
            return f"Sorry, something went wrong while processing your request: {e}"

        raw = response["messages"][-1].content
        if isinstance(raw, list):
            return "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in raw
            )
        return str(raw)

# Provide single export reference for router:
chat_service = RagSqlAgentService()
