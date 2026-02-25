from datetime import datetime, timedelta
from typing import Optional

from passlib.context import CryptContext
from jose import jwt

SECRET_KEY = "change_this_secret_key_make_it_long_random"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Support both argon2 (new) and bcrypt (old), auto-detect by hash prefix
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default="argon2",
    deprecated="auto"
)

def get_password_hash(password: str) -> str:
    # New hashes will be argon2
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # If hash is garbage / unknown format, just treat as invalid
        return False

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
