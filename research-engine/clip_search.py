import torch
import glob
import numpy as np
from transformers import CLIPProcessor, CLIPTextModelWithProjection

print("Loading CrimeVision AI Search Engine...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. Load the mathematical image embeddings
try:
    image_embeddings = np.load("cctv_clip_embeddings.npy")
    image_embeddings = torch.tensor(image_embeddings).to(device)
    frame_paths = sorted(glob.glob("cctv_frames/*.jpg"))[:50]
    
    # ⚠️ SAFETY CHECK: Prevent State Desync!
    if len(frame_paths) != image_embeddings.shape[0]:
        print(f"\n⚠️ SYSTEM DESYNC ERROR:")
        print(f"Folder has {len(frame_paths)} images, but the math file has {image_embeddings.shape[0]} embeddings.")
        print("Please run 'python clip_extractor.py' to synchronize the database!")
        exit()
        
except FileNotFoundError:
    print("Error: Run clip_extractor.py first!")
    exit()

# 2. Ask the investigator what to search for
print("\n" + "="*50)
query = input("🕵️ Enter a forensic search query: ")

# 3. Convert the text into a mathematical embedding
inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model(**inputs)
    text_features = outputs.text_embeds
    text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

# 4. Calculate Cosine Similarity
similarities = (text_embeddings @ image_embeddings.T).squeeze(0)

# 5. Get the Ranked Order (Highest to Lowest)
top_indices = similarities.argsort(descending=True)

print("\n" + "="*50)
print(f"🎯 FORENSIC SEARCH RESULTS FOR:\n'{query}'\n")

# Print out ALL images and their scores to see what the AI is thinking!
for rank, idx in enumerate(top_indices):
    score = similarities[idx].item()
    img_path = frame_paths[idx]
    print(f"Rank {rank + 1}: {img_path} (Confidence: {score:.4f})")

print("="*50 + "\n")