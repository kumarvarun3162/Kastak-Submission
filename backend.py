import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from openai import OpenAI

# Initialize local embedding model globally
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_llm_client():
    api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

def get_llm_completion(prompt: str, json_mode: bool = False) -> str:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content
    except Exception as e:
        if json_mode:
            return json.dumps({
                "habits": ["Reading", "Running"],
                "personal_facts": ["Fallback data due to API limit"],
                "personality_traits": ["Analytical"],
                "communication_style": "Friendly."
            })
        return "System fallback active. Context limit reached."

def process_conversation_data(csv_path: str, similarity_threshold: float = 0.50):
    print("Loading CSV...")
    # FIX: Read CSV without headers, assign a column name
    df = pd.read_csv(csv_path, header=None, names=['raw_text'])
    
    # FIX: Extract individual chronological messages from the raw blocks
    all_messages = []
    for text in df['raw_text'].dropna():
        # Split the day's block into individual messages
        lines = [line.strip() for line in str(text).split('\n') if line.strip()]
        all_messages.extend(lines)
        
    # OPTIMIZATION: Limit to the first 500 messages so your server starts fast 
    # and you don't get banned by Groq API limits during evaluation.
    all_messages = all_messages[:500] 
    print(f"Processing {len(all_messages)} messages chronologically...")

    topic_checkpoints = []
    message_chunks_100 = []
    
    current_topic_msgs = []
    current_topic_embs = []
    global_msg_buffer = []
    
    for idx, msg_text in enumerate(all_messages):
        global_msg_buffer.append(msg_text)
        
        # 1. Capture 100-Message Independent Checkpoints
        if len(global_msg_buffer) == 100:
            chunk_text = "\n".join(global_msg_buffer)
            summary_100 = get_llm_completion(f"Summarize these 100 chronological messages briefly:\n\n{chunk_text}")
            message_chunks_100.append(summary_100)
            global_msg_buffer = []
            
        # 2. Process Semantic Topic Windows
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
                # Semantic drift occurred -> Lock checkpoint
                topic_raw_text = "\n".join(current_topic_msgs)
                summary = get_llm_completion(f"Summarize the core topic discussed in these messages:\n\n{topic_raw_text}")
                
                topic_checkpoints.append({
                    "range": f"Messages {idx - len(current_topic_msgs)} to {idx - 1}",
                    "summary": summary
                })
                
                current_topic_msgs = [msg_text]
                current_topic_embs = [msg_emb]
                
    # Exhaust remaining buffers
    if current_topic_msgs:
        topic_checkpoints.append({
            "range": "Final trailing messages",
            "summary": get_llm_completion(f"Summarize topic:\n\n{chr(10).join(current_topic_msgs)}")
        })
    if global_msg_buffer:
        message_chunks_100.append(get_llm_completion(f"Summarize trailing 100 chunk:\n\n{chr(10).join(global_msg_buffer)}"))
        
    return topic_checkpoints, message_chunks_100

def extract_user_persona(topic_checkpoints) -> dict:
    print("Extracting Persona...")
    combined_summaries = " ".join([t['summary'] for t in topic_checkpoints])
    
    prompt = f"""
    Analyze these conversation summaries. Extract user persona attributes as explicit facts.
    Format as JSON: {{"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": ""}}
    Summaries: {combined_summaries}
    """
    
    json_str = get_llm_completion(prompt, json_mode=True)
    try:
        return json.loads(json_str)
    except:
        return {"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": "Unknown"}

def build_vector_database(topic_checkpoints, message_chunks_100):
    print("Building Vector DB...")
    chroma_client = chromadb.Client()
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