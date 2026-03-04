from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from Infrastructure.database import get_db
from Core.models import User
from Application.auth import get_password_hash, verify_password

router = APIRouter()

@router.post("/signup")
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

@router.post("/login")
def login(
    email: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        return {"ok": False, "message": "Invalid credentials"}

    return {"ok": True, "user_id": user.id}
