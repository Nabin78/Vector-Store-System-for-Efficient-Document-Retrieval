from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.routers.text import router as text_router



app = FastAPI(title="Faiss Vector DB Backend")



app.include_router(text_router)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")





