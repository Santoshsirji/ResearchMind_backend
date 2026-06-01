import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ingest import ingest_pdf
from query import query_rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    doc_id: Optional[str] = None

@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    doc_id: str = Form(...)
):
    tmp_path = f"tmp_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_pdf(tmp_path, doc_id=doc_id, user_id=user_id)
    os.remove(tmp_path)
    return {"status": "success", **result}

@app.post("/query")
async def query_endpoint(body: QueryRequest):
    result = query_rag(body.question, user_id=body.user_id, doc_id=body.doc_id)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
