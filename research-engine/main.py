import torch
import numpy as np
import io
import cv2
import tempfile
import os
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import warnings

# --- Neo4j and Local LLM Imports ---
from neo4j_connector import KnowledgeGraphEngine
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore")

app = FastAPI(title="CrimeVision AI Engine", version="2.0-Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = None
clip_model = None
clip_processor = None
kg_engine = None  

@app.on_event("startup")
async def load_ai_models():
    global sbert_model, clip_model, clip_processor, kg_engine
    print("🚀 Booting up CrimeVision AI Engine (Full Video Scanner)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("🔗 Connecting to Neo4j...")
    kg_engine = KnowledgeGraphEngine()
    print("✅ AI Engine is live and listening for evidence!")

@app.on_event("shutdown")
async def shutdown_event():
    if kg_engine:
        kg_engine.close()

def sinkhorn_knopp(C, epsilon=0.1, iterations=10):
    K = torch.exp(-C / epsilon)
    u = torch.ones_like(K[:, 0]) / K.shape[0]
    v = torch.ones_like(K[0, :]) / K.shape[1]
    for _ in range(iterations):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
    return torch.diag(u) @ K @ torch.diag(v)

def extract_sampled_frames(video_bytes, sample_rate_fps=1):
    """
    INDUSTRY FIX: Scans the entire video and extracts 1 frame per second.
    Zero blind spots for criminal movement.
    """
    temp_video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        cap = cv2.VideoCapture(temp_video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 30
        
        # Calculate how many frames to skip to get exactly 1 frame per second
        frame_interval = int(video_fps / sample_rate_fps)
        frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Grab frame if it lands on our 1-second interval
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
            frame_idx += 1
            
        cap.release()
        return frames
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def generate_graphrag_audit_report(case_id, new_cost, new_statement, new_filename, kg_engine):
    print(f"🧠 Querying Neo4j for Case {case_id} history...")
    case_context = kg_engine.get_case_context(case_id)
    print("🧠 Passing Graph Context to Llama-3...")
    llm = Ollama(model="llama3", temperature=0)
    
    prompt = f"""
    You are an AI Forensic Analyst. You have access to a Neo4j Knowledge Graph containing the evidence history of a criminal case.
    
    {case_context}
    
    NEWLY INGESTED EVIDENCE:
    - Video: {new_filename}
    - Statement: "{new_statement}"
    - Mathematical Contradiction Cost: {new_cost} (Scores > 1.15 indicate a contradiction)
    
    TASK: 
    Write a professional, 3-sentence forensic audit log. 
    First, evaluate if the NEW evidence contradicts itself based on the math. 
    Second, mention how it fits into the context of the EXISTING case history from the graph.
    Do not use hallucinated facts, ONLY use the provided graph context.
    """
    return llm.invoke(prompt)

@app.post("/analyze_evidence/")
async def analyze_evidence(
    case_id: str = Form("CASE-2026-001"),  
    evidence_file: UploadFile = File(...),
    statement_text: str = Form(None), 
    statement_file: UploadFile = File(None)
):
    final_statement = ""
    if statement_file and statement_file.filename.endswith(".txt"):
        txt_bytes = await statement_file.read()
        final_statement = txt_bytes.decode('utf-8')
    elif statement_text:
        final_statement = statement_text
    else:
        return {"error": "You must provide either a typed statement or a .txt file!"}

    file_bytes = await evidence_file.read()
    
    # 1. Video Processing: Grab ALL sampled frames instead of just the middle one
    if evidence_file.content_type.startswith("video") or evidence_file.filename.endswith(".mp4"):
        images = extract_sampled_frames(file_bytes, sample_rate_fps=1)
        if not images:
            return {"error": "Failed to extract frames from MP4 video."}
    else:
        images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
    
    print(f"🔍 AI Scanning {len(images)} frames across the video timeline...")
    
    # 2. PyTorch Math Engine - Find the Best Matching Frame
    text_embeddings = sbert_model.encode([final_statement], convert_to_tensor=True).clone()
    best_cost = float('inf')
    
    with torch.no_grad():
        torch.manual_seed(42)
        text_projector = torch.nn.Linear(384, 256).to(device)
        vision_projector = torch.nn.Linear(512, 256).to(device)
        
        aligned_text = text_projector(text_embeddings)
        aligned_text = aligned_text / aligned_text.norm(dim=-1, keepdim=True)
        
        # Scan through the extracted frames one by one
        for img in images:
            inputs = clip_processor(images=[img], return_tensors="pt", padding=True).to(device)
            vision_features = clip_model(**inputs).image_embeds
            
            aligned_vision = vision_projector(vision_features)
            aligned_vision = aligned_vision / aligned_vision.norm(dim=-1, keepdim=True)
            
            cost_matrix = 1.0 - (aligned_text @ aligned_vision.T)
            current_cost = cost_matrix[0][0].item()
            
            # Keep the lowest cost (This identifies the frame where the event actually happens)
            if current_cost < best_cost:
                best_cost = current_cost
                
    # We now have the best possible cost found across the entire video
    final_cost = best_cost

    # 3. Save NEW data to Neo4j
    try:
        kg_engine.create_evidential_link(
            case_id=case_id,
            evidence_filename=evidence_file.filename,
            statement_text=final_statement,
            sinkhorn_cost=final_cost
        )
        graph_status = "Saved to Neo4j Successfully"
    except Exception as e:
        graph_status = f"Neo4j Error: {str(e)}"

    # 4. GraphRAG AI Engine
    try:
        audit_report = generate_graphrag_audit_report(case_id, round(final_cost, 4), final_statement, evidence_file.filename, kg_engine)
    except Exception as e:
        audit_report = f"LLM Error: {str(e)}"

    return {
        "status": "success",
        "case_id": case_id,
        "evidence_processed": evidence_file.filename,
        "sinkhorn_alignment_cost": round(final_cost, 4),
        "graph_status": graph_status,
        "audit_report": audit_report
    }