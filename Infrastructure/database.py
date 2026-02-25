import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "sqlite:///./klassifier.db"


# ----- SQLAlchemy Engine -----
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=True,           # show SQL logs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
