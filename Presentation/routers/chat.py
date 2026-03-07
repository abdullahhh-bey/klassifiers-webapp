from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
import os

from Infrastructure.database import get_db
from Core.models import Connected_DataBases
from Core.schemas import ChatRequest
from Application.services.chat_agent import chat_service

router = APIRouter()

DB_SECRET_KEY = os.getenv("DB_SECRET_KEY")
if not DB_SECRET_KEY:
    raise RuntimeError("DB_SECRET_KEY not found in .env")

fernet = Fernet(DB_SECRET_KEY.encode())

@router.post("/api/chat")
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

        try:
            response_msg = chat_service.chat(
                user_id=connection.user_id,
                connection_id=connection.id,
                db_type=connection.db_type,
                host=connection.host,
                port=connection.port,
                db_name=connection.db_name,
                username=connection.username,
                password=decrypted_password,
                message=payload.message
            )
            return {"response": response_msg}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            print("CHAT ERROR:", e)
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        print("CHAT ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
