import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import process_conversation_data, extract_user_persona, build_vector_database, embedder, get_llm_completion

# Global State
SYSTEM_STATE = {"persona": None, "vector_db": None, "ready": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTING: INITIALIZING ML PIPELINE ---")
    try:
        # Dynamically find the absolute path to conversations.csv
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "conversations.csv")
        
        print(f"Looking for CSV at: {csv_path}")
        
        if not os.path.exists(csv_path):
            print("CRITICAL: conversations.csv is missing from the folder!")
            return
            
        topics, chunks = process_conversation_data(csv_path, similarity_threshold=0.50)
        SYSTEM_STATE["persona"] = extract_user_persona(topics)
        SYSTEM_STATE["vector_db"] = build_vector_database(topics, chunks)
        SYSTEM_STATE["ready"] = True
        print("--- PIPELINE READY ---")
    except Exception as e:
        print(f"Initialization Error: {e}")
    yield
    print("Shutting down...")

    
app = FastAPI(title="KaStack Labs Assistant", lifespan=lifespan)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryPayload(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"ready": SYSTEM_STATE["ready"], "persona": SYSTEM_STATE["persona"]}
    )

@app.post("/query")
async def process_chat_query(payload: QueryPayload):
    if not SYSTEM_STATE["ready"]:
        raise HTTPException(status_code=400, detail="System is still initializing.")
        
    try:
        user_text = payload.query
        query_vector = embedder.encode([user_text])[0].tolist()
        
        search_matches = SYSTEM_STATE["vector_db"].query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        matched_documents = search_matches.get('documents', [[]])[0]
        context_string = "\n".join([f"- {doc}" for doc in matched_documents])
        
        rag_prompt = f"""
        Answer the user using only these extracted facts and summaries.
        
        Persona:
        {json.dumps(SYSTEM_STATE["persona"])}
        
        Summaries:
        {context_string}
        
        User Query: {user_text}
        """
        
        generated_answer = get_llm_completion(rag_prompt)
        return JSONResponse(content={"answer": generated_answer})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))