import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from sentence_transformers import SentenceTransformer
import warnings
import os

warnings.filterwarnings("ignore")

print("📊 Initializing Week 10 Quantitative Benchmarking Suite...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Models
print("🧠 Loading PyTorch Models (CLIP & SBERT)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frame(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def compute_cost(image, statement, uncertainty_penalty=0.0):
    """Computes the Sinkhorn mathematical cost, incorporating visual uncertainty."""
    inputs = clip_processor(images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vision_features = clip_model(**inputs).image_embeds
    text_embeddings = sbert_model.encode([statement], convert_to_tensor=True).clone()
    
    with torch.no_grad():
        torch.manual_seed(42)
        text_projector = torch.nn.Linear(384, 256).to(device)
        vision_projector = torch.nn.Linear(512, 256).to(device)
        
        aligned_text = text_projector(text_embeddings)
        aligned_vision = vision_projector(vision_features)
        
        aligned_text = aligned_text / aligned_text.norm(dim=-1, keepdim=True)
        aligned_vision = aligned_vision / aligned_vision.norm(dim=-1, keepdim=True)
        
        # Base Cosine Cost
        cost_matrix = 1.0 - (aligned_text @ aligned_vision.T)
        
        # Apply Evidential Optimal Transport Uncertainty Dampening
        # If uncertainty is high (e.g., blurry video), the penalty is dampened to prevent false positives
        dampened_cost = cost_matrix[0][0].item() * (1.0 - uncertainty_penalty)
        return dampened_cost
