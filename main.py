import os
from dotenv import load_dotenv

# ----------------- ENV -----------------
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from Infrastructure.database import engine, Base
from Presentation.routers import pages, auth, database, chat

# ----------------- APP -----------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- ROUTERS -----------------
app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(database.router)
app.include_router(chat.router)

# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
