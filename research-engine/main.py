import torch
import numpy as np
import io
import cv2
import tempfile
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="CrimeVision AI Engine", version="1.0")

# 1. CORS MIDDLEWARE (The Bridge to Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this will be your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold our AI models
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = None
clip_model = None
clip_processor = None

@app.on_event("startup")
async def load_ai_models():
    global sbert_model, clip_model, clip_processor
    print("🚀 Booting up CrimeVision AI Engine...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ AI Engine is live and listening for evidence!")

# Sinkhorn Math
def sinkhorn_knopp(C, epsilon=0.1, iterations=10):
    K = torch.exp(-C / epsilon)
    u = torch.ones_like(K[:, 0]) / K.shape[0]
    v = torch.ones_like(K[0, :]) / K.shape[1]
    for _ in range(iterations):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
    return torch.diag(u) @ K @ torch.diag(v)

# 2. MP4 VIDEO EXTRACTION LOGIC
def extract_keyframe_from_video(video_bytes):
    """Saves MP4 to a temp file, uses OpenCV to extract the middle frame, and returns a PIL Image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
    
    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fast-forward to the middle of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # OpenCV uses BGR, PIL uses RGB. We must convert it.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

@app.post("/analyze_evidence/")
async def analyze_evidence(
    evidence_file: UploadFile = File(...),
    statement_text: str = Form(None), 
    statement_file: UploadFile = File(None)
):
    """
    Advanced Endpoint: Handles JPG/MP4 visual evidence and String/TXT statement evidence.
    """
    # --- PROCESS TEXT EVIDENCE ---
    final_statement = ""
    if statement_file and statement_file.filename.endswith(".txt"):
        # 3. TXT DOCUMENT PARSING
        txt_bytes = await statement_file.read()
        final_statement = txt_bytes.decode('utf-8')
    elif statement_text:
        final_statement = statement_text
    else:
        return {"error": "You must provide either a typed statement or a .txt file!"}

    # --- PROCESS VISUAL EVIDENCE (JPG or MP4) ---
    file_bytes = await evidence_file.read()
    if evidence_file.content_type.startswith("video") or evidence_file.filename.endswith(".mp4"):
        image = extract_keyframe_from_video(file_bytes)
        if image is None:
            return {"error": "Failed to extract frame from MP4 video."}
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # --- RUN THE AI ENGINE ---
    inputs = clip_processor(images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vision_features = clip_model(**inputs).image_embeds
    
    text_embeddings = sbert_model.encode([final_statement], convert_to_tensor=True).clone()
    
    with torch.no_grad():
        torch.manual_seed(42)
        text_projector = torch.nn.Linear(384, 256).to(device)
        vision_projector = torch.nn.Linear(512, 256).to(device)
        
        aligned_text = text_projector(text_embeddings)
        aligned_vision = vision_projector(vision_features)
        
        aligned_text = aligned_text / aligned_text.norm(dim=-1, keepdim=True)
        aligned_vision = aligned_vision / aligned_vision.norm(dim=-1, keepdim=True)
        
        cost_matrix = 1.0 - (aligned_text @ aligned_vision.T)
        ot_plan = sinkhorn_knopp(cost_matrix)
        final_cost = cost_matrix[0][0].item()

    return {
        "status": "success",
        "evidence_processed": evidence_file.filename,
        "statement_source": "TXT File" if statement_file else "Manual Input",
        "extracted_statement": final_statement[:100] + "..." if len(final_statement) > 100 else final_statement,
        "sinkhorn_alignment_cost": round(final_cost, 4)
    }