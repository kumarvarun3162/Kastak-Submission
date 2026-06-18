import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import process_conversation_data, extract_user_persona, build_vector_database, embedder, get_llm_completion

app = FastAPI(title="KaStack Labs Conversation Engine")

# Prepare paths for assets
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Server session state storage
SYSTEM_STATE = {
    "persona": None,
    "vector_db": None,
    "pipeline_initialized": False
}

class QueryPayload(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the single-page application dashboard workspace."""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "ready": SYSTEM_STATE["pipeline_initialized"]
    })

@app.post("/upload")
async def handle_csv_upload(file: UploadFile = File(...)):
    """Receives data file streams, runs analytical checkpoints, and provisions vector entries."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Unsupported media layout. System requires standard CSV extensions.")
    
    target_path = f"active_{file.filename}"
    with open(target_path, "wb") as storage_buffer:
        shutil.copyfileobj(file.file, storage_buffer)
        
    try:
        # Run processing functions
        topics, chunks = process_conversation_data(target_path, similarity_threshold=0.50)
        extracted_profile = extract_user_persona(topics)
        active_db = build_vector_database(topics, chunks)
        
        # Save results to state
        SYSTEM_STATE["persona"] = extracted_profile
        SYSTEM_STATE["vector_db"] = active_db
        SYSTEM_STATE["pipeline_initialized"] = True
        
        if os.path.exists(target_path):
            os.remove(target_path)
            
        return JSONResponse(content={
            "status": "synchronized",
            "persona": extracted_profile
        })
        
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=f"Pipeline construction processing crashed: {str(e)}")

@app.post("/query")
async def process_chat_query(payload: QueryPayload):
    """Executes search queries against structured persona elements and indexed vector records."""
    if not SYSTEM_STATE["pipeline_initialized"] or SYSTEM_STATE["vector_db"] is None:
        raise HTTPException(status_code=400, detail="Context vectors are offline. Upload the dataset first.")
        
    try:
        user_text = payload.query
        
        # Calculate search coordinates
        query_vector = embedder.encode([user_text])[0].tolist()
        search_matches = SYSTEM_STATE["vector_db"].query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        matched_documents = search_matches.get('documents', [[]])[0]
        context_string = "\n".join([f"- Found Checkpoint Summary: {doc}" for doc in matched_documents])
        
        rag_prompt = f"""
        You are an advanced AI engine analyzing user text profiles. 
        Synthesize a direct response to the user query using ONLY the verified facts listed below.
        
        Extracted Persona JSON Elements:
        {json.dumps(SYSTEM_STATE["persona"])}
        
        Relevant Segment Summaries from Vector Query:
        {context_string}
        
        User Query: {user_text}
        """
        
        generated_answer = get_llm_completion(rag_prompt)
        return JSONResponse(content={"answer": generated_answer})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processor encountered a critical calculation fault: {str(e)}")