import os
from fastapi import FastAPI, Request, HTTPException, Depends, Body, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from Infrastructure.database import engine, Base, get_db
from Core.models import User, Connected_DataBases
from Application.auth import get_password_hash, verify_password


# ----------------- ENV -----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

DB_SECRET_KEY = os.getenv("DB_SECRET_KEY")
if not DB_SECRET_KEY:
    raise RuntimeError("DB_SECRET_KEY not found in .env")


# ----------------- CLIENTS -----------------
client = OpenAI(api_key=OPENAI_API_KEY)
fernet = Fernet(DB_SECRET_KEY.encode())


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

        system_prompt = f"""
You are a secure AI database assistant.

Database type: {connection.db_type}

Rules:
- Never expose SQL
- Never expose credentials
- Never mention passwords
- Explain results in simple English
- Ask for clarification if query is unsafe or unclear
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.message},
            ],
        )

        return {
            "response": response.choices[0].message.content
        }

    except HTTPException:
        raise

    except Exception as e:
        print("CHAT ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to process your request"
        )


# ----------------- ANALYZE (OPTIONAL) -----------------
@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": payload.query},
        ],
    )
    return {"result": response.choices[0].message.content}


# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
