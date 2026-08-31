import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

# Suppress HuggingFace warnings for clean terminal output
warnings.filterwarnings("ignore")

print("Initializing Week 4 Multimodal Fusion Engine...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the Sentence-BERT Text Extractor
print("Loading Sentence-BERT (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# 2. Load the CLIP Visual Embeddings (from Week 3)
try:
    cctv_embeddings = np.load("cctv_clip_embeddings.npy")
    cctv_tensor = torch.tensor(cctv_embeddings, dtype=torch.float32).to(device)
except FileNotFoundError:
    print("Error: cctv_clip_embeddings.npy not found! Run clip_extractor.py first.")
    exit()

# 3. Ingest the Witness Transcript
witness_transcript = [
    "I saw a car driving down the street.",
    "There was a man walking his dog near the corner.",
    "Suddenly, I looked up and saw an aeroplane flying low.",
    "Then I walked past the empty classroom."
]

print("\nExtracting Witness Transcript Embeddings...")
# Convert to normal tensor to avoid the inference_mode lock
transcript_embeddings = sbert_model.encode(witness_transcript, convert_to_tensor=True).clone()

# 4. Multimodal Projection Bridge & Sinkhorn Math
print("Executing Evidential Optimal Transport (Sinkhorn Iterations)...")

# WRAP IN NO_GRAD to prevent PyTorch from trying to track training gradients
with torch.no_grad():
    torch.manual_seed(42) # For reproducible results
    text_projector = torch.nn.Linear(384, 256).to(device)
    vision_projector = torch.nn.Linear(512, 256).to(device)

    aligned_text = text_projector(transcript_embeddings)
    aligned_vision = vision_projector(cctv_tensor)

    # L2 Normalize for Cosine Cost Matrix
    aligned_text = aligned_text / aligned_text.norm(dim=-1, keepdim=True)
    aligned_vision = aligned_vision / aligned_vision.norm(dim=-1, keepdim=True)

    # 5. Calculate Cost Matrix (1 - Cosine Similarity)
    cost_matrix = 1.0 - (aligned_text @ aligned_vision.T)

    # Evidential Optimal Transport (Sinkhorn Distance)
    def sinkhorn_knopp(C, epsilon=0.1, iterations=10):
        K = torch.exp(-C / epsilon)
        u = torch.ones_like(K[:, 0]) / K.shape[0]
        v = torch.ones_like(K[0, :]) / K.shape[1]
        
        for _ in range(iterations):
            u = 1.0 / (K @ v)
            v = 1.0 / (K.T @ u)
        
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan

    # Run Sinkhorn
    optimal_transport_plan = sinkhorn_knopp(cost_matrix)

# 6. Output the Alignment Results
print("\n" + "="*50)
print("🎯 MULTIMODAL ALIGNMENT COMPLETE")
print("="*50)
print(f"Witness Statements Processed: {len(witness_transcript)}")
print(f"CCTV Frames Processed: {cctv_tensor.shape[0]}")
print(f"Shared Latent Space: 256-D")
print("-" * 50)
print("Optimal Transport Cost Matrix (Alignment Cost per Frame):")
print(torch.round(cost_matrix * 1000) / 1000) # Rounded for clean display
print("="*50 + "\n")