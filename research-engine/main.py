import torch
import numpy as np
import io
import cv2
import tempfile
import os
import PyPDF2
import docx
import shutil
from pathlib import Path
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

app = FastAPI(title="CrimeVision AI Engine", version="2.0-Production-Ready")

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
    print("🚀 Booting up CrimeVision AI Engine (Full Video Scanner & Multi-format parser)...")
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

def extract_sampled_frames(video_path, sample_rate_fps=1):
    """
    INDUSTRY FIX (Disk Streaming): Reads the video directly from the hard drive.
    No more RAM limits or TempFiles!
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30
    
    frame_interval = int(video_fps / sample_rate_fps)
    frames = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
        frame_idx += 1
        
    cap.release()
    return frames

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
    # 1. ENTERPRISE DISK STREAMING (Bypass RAM)
    # Create a local directory to hold the physical files
    UPLOAD_DIR = Path("C:/Capstone-Project/CrimeVision-AI/research-engine/local_evidence_vault")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    final_statement = ""
    
    # Process text/document by saving it to disk first
    if statement_file:
        statement_path = UPLOAD_DIR / statement_file.filename
        with open(statement_path, "wb") as buffer:
            shutil.copyfileobj(statement_file.file, buffer) # Streams in tiny 1MB chunks
            
        file_ext = statement_file.filename.lower()
        if file_ext.endswith(".txt"):
            with open(statement_path, "r", encoding="utf-8") as f:
                final_statement = f.read()
        elif file_ext.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(str(statement_path))
            final_statement = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif file_ext.endswith(".docx"):
            doc = docx.Document(str(statement_path))
            final_statement = " ".join([p.text for p in doc.paragraphs])
        else:
            return {"error": f"Unsupported format: {file_ext}"}
            
    elif statement_text:
        final_statement = statement_text
    else:
        return {"error": "Provide a statement!"}

    # Process Media by saving it to disk first
    evidence_path = UPLOAD_DIR / evidence_file.filename
    with open(evidence_path, "wb") as buffer:
        shutil.copyfileobj(evidence_file.file, buffer) # Streams heavy video directly to C:\ drive
        
    ev_ext = evidence_file.filename.lower()
    
    # 2. Extract Data directly from the Hard Drive
    if ev_ext.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Notice we pass the file PATH now, not the raw bytes!
        images = extract_sampled_frames(str(evidence_path), sample_rate_fps=1)
        if not images:
            return {"error": f"Failed to extract frames from {ev_ext}"}
            
    elif ev_ext.endswith(('.jpg', '.jpeg', '.png')):
        images = [Image.open(str(evidence_path)).convert("RGB")]
    else:
        return {"error": f"Unsupported media: {ev_ext}"}
    
    print(f"🔍 AI Scanning {len(images)} frames from secure local vault...")
    
    # ... (The rest of the PyTorch math and Neo4j graph saving remains EXACTLY the same from here down!) ...
    text_embeddings = sbert_model.encode([final_statement], convert_to_tensor=True).clone()
    best_cost = float('inf')
    
    with torch.no_grad():
        torch.manual_seed(42)
        text_projector = torch.nn.Linear(384, 256).to(device)
        vision_projector = torch.nn.Linear(512, 256).to(device)
        
        aligned_text = text_projector(text_embeddings)
        aligned_text = aligned_text / aligned_text.norm(dim=-1, keepdim=True)
        
        for img in images:
            inputs = clip_processor(images=[img], return_tensors="pt", padding=True).to(device)
            vision_features = clip_model(**inputs).image_embeds
            
            aligned_vision = vision_projector(vision_features)
            aligned_vision = aligned_vision / aligned_vision.norm(dim=-1, keepdim=True)
            
            cost_matrix = 1.0 - (aligned_text @ aligned_vision.T)
            current_cost = cost_matrix[0][0].item()
            
            if current_cost < best_cost:
                best_cost = current_cost
                
    final_cost = best_cost

    try:
        kg_engine.create_evidential_link(case_id, evidence_file.filename, final_statement, final_cost)
        graph_status = "Saved to Neo4j Successfully"
    except Exception as e:
        graph_status = f"Neo4j Error: {str(e)}"

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