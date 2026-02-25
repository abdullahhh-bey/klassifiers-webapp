import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = (
    "mssql+pyodbc://@DESKTOP-D642JM0/klassifier"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)


# ----- SQLAlchemy Engine -----
engine = create_engine(
    DATABASE_URL,
    echo=True,           # show SQL logs
    fast_executemany=True
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
