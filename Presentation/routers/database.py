from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
import os

from Infrastructure.database import get_db
from Core.models import User, Connected_DataBases

router = APIRouter()

DB_SECRET_KEY = os.getenv("DB_SECRET_KEY")
if not DB_SECRET_KEY:
    raise RuntimeError("DB_SECRET_KEY not found in .env")

fernet = Fernet(DB_SECRET_KEY.encode())

@router.post("/api/database-connection", status_code=status.HTTP_201_CREATED)
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

@router.get("/api/databases")
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
