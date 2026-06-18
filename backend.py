import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from openai import OpenAI

# 1. Load the .env file automatically
load_dotenv()

# 2. Local Embedding Model for Mathematical Topic Splitting (100% Free & Local)
print("Loading Local Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_llm_client():
    # 3. Pull the Groq API key from your .env file
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n[WARNING] GROQ_API_KEY not found in .env file! System will fail.\n")
    
    # 4. Point strictly to Groq's ultra-fast free cloud endpoints
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

def get_llm_completion(prompt: str, json_mode: bool = False) -> str:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Groq's free flagship model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"API Error: {str(e)}"
        print(f"\n--- CRITICAL API ERROR --- {error_msg}\n")
        
        # Graceful fallback so the system doesn't crash during a demo
        if json_mode:
            return json.dumps({
                "habits": ["Error: Could not connect to API"],
                "personal_facts": ["Check your .env file and internet connection"],
                "personality_traits": ["API Offline"],
                "communication_style": error_msg
            })
        return f"System fallback active: {error_msg}"

def process_conversation_data(csv_path: str, similarity_threshold: float = 0.50):
    print("Loading Conversation CSV...")
    df = pd.read_csv(csv_path, header=None, names=['raw_text'])
    
    all_messages = []
    for text in df['raw_text'].dropna():
        lines = [line.strip() for line in str(text).split('\n') if line.strip()]
        all_messages.extend(lines)
        
    # LIMIT to 300 messages to ensure you don't hit Groq's free-tier rate limits!
    all_messages = all_messages[:300] 
    print(f"Processing {len(all_messages)} messages chronologically...")

    topic_checkpoints = []
    message_chunks_100 = []
    
    current_topic_msgs = []
    current_topic_embs = []
    global_msg_buffer = []
    
    for idx, msg_text in enumerate(all_messages):
        global_msg_buffer.append(msg_text)
        
        # 100-Message Chunk Logic
        if len(global_msg_buffer) == 100:
            chunk_text = "\n".join(global_msg_buffer)
            summary_100 = get_llm_completion(f"Summarize these 100 chronological messages briefly:\n\n{chunk_text}")
            message_chunks_100.append(summary_100)
            global_msg_buffer = []
            
        # Topic Splitting Logic (Cosine Similarity)
        msg_emb = embedder.encode([msg_text])[0]
        
        if not current_topic_msgs:
            current_topic_msgs.append(msg_text)
            current_topic_embs.append(msg_emb)
        else:
            topic_mean_emb = np.mean(current_topic_embs, axis=0).reshape(1, -1)
            sim = cosine_similarity(msg_emb.reshape(1, -1), topic_mean_emb)[0][0]
            
            if sim >= similarity_threshold:
                current_topic_msgs.append(msg_text)
                current_topic_embs.append(msg_emb)
            else:
                topic_raw_text = "\n".join(current_topic_msgs)
                summary = get_llm_completion(f"Summarize the core topic discussed in these messages:\n\n{topic_raw_text}")
                
                topic_checkpoints.append({
                    "range": f"Messages {idx - len(current_topic_msgs)} to {idx - 1}",
                    "summary": summary
                })
                
                current_topic_msgs = [msg_text]
                current_topic_embs = [msg_emb]
                
    # Exhaust trailing buffers
    if current_topic_msgs:
        topic_checkpoints.append({
            "range": "Final trailing messages",
            "summary": get_llm_completion(f"Summarize topic:\n\n{chr(10).join(current_topic_msgs)}")
        })
    if global_msg_buffer:
        message_chunks_100.append(get_llm_completion(f"Summarize trailing chunk:\n\n{chr(10).join(global_msg_buffer)}"))
        
    return topic_checkpoints, message_chunks_100

def extract_user_persona(topic_checkpoints) -> dict:
    print("Extracting Persona using 70B Model...")
    combined_summaries = " ".join([t['summary'] for t in topic_checkpoints])
    
    prompt = f"""
    Analyze these conversation summaries. Extract user persona attributes as explicit facts.
    Format strictly as JSON: {{"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": ""}}
    Summaries: {combined_summaries}
    """
    
    json_str = get_llm_completion(prompt, json_mode=True)
    try:
        return json.loads(json_str)
    except:
        return {"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": "Failed to parse JSON"}

def build_vector_database(topic_checkpoints, message_chunks_100):
    print("Building Persistent ChromaDB Vector DB...")
    # FIX: Use PersistentClient to save the database to a folder
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    
    try:
        chroma_client.delete_collection(name="conversation_insights")
    except:
        pass
        
    collection = chroma_client.create_collection(name="conversation_insights")
    
    for i, t in enumerate(topic_checkpoints):
        text = t['summary']
        collection.add(
            ids=[f"topic_{i}"],
            embeddings=[embedder.encode([text])[0].tolist()],
            documents=[text],
            metadatas=[{"type": "topic_checkpoint"}]
        )
        
    for i, chunk in enumerate(message_chunks_100):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedder.encode([chunk])[0].tolist()],
            documents=[chunk],
            metadatas=[{"type": "hundred_chunk"}]
        )
        
    return collection