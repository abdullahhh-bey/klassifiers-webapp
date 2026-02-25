import os
from fastapi import FastAPI, Request, HTTPException, Depends, Body, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from Infrastructure.database import engine, Base, get_db
from Core.models import User, Connected_DataBases
from Application.auth import get_password_hash, verify_password


# ----------------- ENV -----------------
load_dotenv()

GEMINI_API_KEY = os.getenv("geminiAPI")
if not GEMINI_API_KEY:
    raise RuntimeError("geminiAPI not found in .env")

DB_SECRET_KEY = os.getenv("DB_SECRET_KEY")
if not DB_SECRET_KEY:
    raise RuntimeError("DB_SECRET_KEY not found in .env")


# ----------------- CLIENTS -----------------
fernet = Fernet(DB_SECRET_KEY.encode())
memory = MemorySaver()


# ----------------- MODELS -----------------
class AnalyzeRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    database_id: int
    message: str


# ----------------- APP -----------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------- PAGES -----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/database-connection", response_class=HTMLResponse)
async def database_connection(request: Request):
    return templates.TemplateResponse("databaseconnection.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# ----------------- AUTH -----------------
@app.post("/signup")
def signup(
    firstname: str = Body(...),
    lastname: str = Body(...),
    email: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    if db.query(User).filter(User.email == email).first():
        return {"ok": False, "message": "Email already registered"}

    user = User(
        firstname=firstname,
        lastname=lastname,
        email=email,
        hashed_password=get_password_hash(password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"ok": True, "user_id": user.id}


@app.post("/login")
def login(
    email: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        return {"ok": False, "message": "Invalid credentials"}

    return {"ok": True, "user_id": user.id}


# ----------------- SAVE DATABASE CONNECTION -----------------
@app.post("/api/database-connection", status_code=status.HTTP_201_CREATED)
def save_database_connection(
    user_id: int = Body(...),
    db_type: str = Body(...),
    host: str = Body(...),
    port: int = Body(...),
    db_name: str = Body(...),
    username: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    encrypted_password = fernet.encrypt(password.encode()).decode()

    connection = Connected_DataBases(
        user_id=user_id,
        db_type=db_type,
        host=host,
        port=port,
        db_name=db_name,
        username=username,
        encrypted_password=encrypted_password,
    )

    db.add(connection)
    db.commit()
    db.refresh(connection)

    return {"ok": True, "connection_id": connection.id}


# ----------------- LIST USER DATABASES -----------------
@app.get("/api/databases")
def list_databases(user_id: int, db: Session = Depends(get_db)):
    databases = db.query(Connected_DataBases).filter(
        Connected_DataBases.user_id == user_id
    ).all()

    return [
        {
            "id": d.id,
            "name": f"{d.db_name} ({d.db_type})",
        }
        for d in databases
    ]


# ----------------- CHAT API -----------------
@app.post("/api/chat")
def chat_with_database(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    try:
        connection = db.query(Connected_DataBases).filter(
            Connected_DataBases.id == payload.database_id
        ).first()

        if not connection:
            raise HTTPException(status_code=404, detail="Database not found")

        try:
            decrypted_password = fernet.decrypt(
                connection.encrypted_password.encode()
            ).decode()
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Failed to decrypt database credentials"
            )

        # Construct DB URI dynamically based on user's connection details
        db_type = connection.db_type.lower()
        if db_type == "postgresql":
            db_uri = f"postgresql+psycopg2://{connection.username}:{decrypted_password}@{connection.host}:{connection.port}/{connection.db_name}"
        elif db_type == "mysql":
            db_uri = f"mysql+pymysql://{connection.username}:{decrypted_password}@{connection.host}:{connection.port}/{connection.db_name}"
        elif db_type == "clickhouse":
            # Using clickhouse-connect for clickhouse
            db_uri = f"clickhouse+connect://{connection.username}:{decrypted_password}@{connection.host}:{connection.port}/{connection.db_name}"
        else:
            raise HTTPException(status_code=400, detail="Unsupported database type")

        # 1. Initialize DB Toolkit
        sql_db = SQLDatabase.from_uri(db_uri)

        # 2. Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)
        
        # 3. Setup Agent Tools
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
        tools = toolkit.get_tools()
        
        # 4. Agent System Prompt
        system_prompt = f"""You are a helpful database assistant.
You are querying a {connection.db_type} database.
Given an input question, create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
If you get an error while executing a query, rewrite the query and try again.
If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
NEVER make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Explain the result to the user clearly."""

        # 5. Create LangGraph React Agent
        agent_executor = create_react_agent(
            llm,
            tools,
            state_modifier=system_prompt,
            checkpointer=memory,
        )

        # 6. Session Management (Per Database Connection)
        thread_id = f"user_{connection.user_id}_db_{connection.id}"
        config = {"configurable": {"thread_id": thread_id}}

        # 7. Run Agent
        response = agent_executor.invoke(
            {"messages": [("user", payload.message)]}, 
            config=config
        )

        final_msg = response["messages"][-1].content

        return {
            "response": final_msg
        }

    except HTTPException:
        raise
    except Exception as e:
        print("CHAT ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
