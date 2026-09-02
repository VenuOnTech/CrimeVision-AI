# CrimeVision AI
> A Multimodal Artificial Intelligence Framework for Intelligent Crime Scene Reconstruction, Uncertainty-Aware Evidence Fusion, and Investigation Support

[![Python](https://shields.io)](/static/img/python-logo.png)
[![FastAPI](https://shields.io)](https://tiangolo.com)
[![Framework](https://shields.io)]()
[![License](https://shields.io)](LICENSE)

CrimeVision AI is an intelligent decision-support system and digital forensics research framework designed to automate heterogeneous evidence organization, cross-modal relationship mapping, and chronological timeline generation. 

Unlike traditional deterministic forensics systems, CrimeVision AI introduces a novel **Uncertainty-Aware Evidential Fusion Layer** that mathematically quantifies visual and textual ambiguity on noisy field data, preventing false-contradiction flags.

---

## Key Features

*   **Multi-Format Ingestion:** Centralizes MP4, JPEG, WAV, PDF, and GPS data into a single, relational graph schema.
*   **Uncertainty-Aware Fusion:** Combines **Subjective Logic (Dirichlet Uncertainty)** with **Optimal Transport (Sinkhorn Divergence)** to evaluate noisy data before declaring timeline contradictions.
*   **Contradiction Heatmaps:** Explicitly maps and highlights conflicting information between witness transcripts and crime scene visual zones.
*   **Interactive Relationship Graphs:** Employs **Neo4j** to build searchable entity paths (`Suspect -> Vehicle -> Location`) leveraging specialized Graph-RAG architectures.
*   **Automated Chronology:** Automatically constructs and updates editable chronological case timelines.

---

## Research Core (Track 1 Math)

Standard AI engines confidently flag mismatches on blurry images, leading to catastrophic false-positive contradictions. CrimeVision AI models epistemic doubt across three steps:

1.  **Epistemic Parameterization:** Visual patches \(\mathbf{v}_i\) and text tokens \(\mathbf{t}_j\) map to the concentration parameters of a Dirichlet distribution \(\boldsymbol{\alpha}\):
    \[\alpha_v = \text{ReLU}(\mathbf{W}_v\mathbf{v}_i) + 1, \quad \alpha_t = \text{ReLU}(\mathbf{W}_t\mathbf{t}_j) + 1\]
2.  **Uncertainty-Dampened Cost Matrix:** Low-quality imagery dynamically inflates the uncertainty parameter \(u\), scaling down the penalization step inside our custom cost matrix \(\tilde{\mathbf{C}}_{ij}\).
3.  **Sinkhorn Divergence Objective:** Employs an entropy-regularized Optimal Transport plan to resolve word-to-region alignments.

---

## Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Research Engine** | PyTorch, POT (Python Optimal Transport), Hugging Face (CLIP, DeBERTa-v3), Whisper, YOLOv8, OpenCV |
| **Backend & APIs** | Python, FastAPI |
| **Databases** | MongoDB (Raw Evidence/Metadata), Neo4j (Entity Relationship Graphs) |
| **Frontend Web** | React.js, Next.js, Tailwind CSS, Material UI, React Flow, Chart.js |
| **Mobile App** | Flutter (Remote Geotagged Uploads) |
| **DevOps** | Docker, GitHub Actions |

---

## Repository Structure

```text
crimevision-ai/
├── .github/workflows/       # CI/CD Pipelines
├── backend/                 # FastAPI Asynchronous REST APIs
│   └── app/                 # Authentication, Case & Ingestion Endpoints
├── frontend-web/            # Next.js Web Dashboard & Interactive Canvas
├── frontend-mobile/         # Flutter Field Application
├── research-engine/         # Core PyTorch Mathematical Formulation Sandbox
├── docker-compose.yml       # Orchestration file for MongoDB & Neo4j
└── README.md
```

---

## Getting Started

### Prerequisites
*   [Docker & Docker Compose](https://docker.com)
*   Python 3.10+
*   Node.js v18+ & Flutter SDK (for frontends)

### 1. Database & Infrastructure Setup
Spin up local pre-configured instances of MongoDB and Neo4j using Docker Compose:
```bash
docker-compose up -d
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Frontend Web Setup
```bash
cd frontend-web
npm install
npm run dev
```

---

## 👥 Team Members

*   **Shyamal** (Computer Science & Engineering)
*   **Moukthika** (Computer Science & Engineering)
*   **Venu** (Computer Science & Engineering)
*   **Niteesha** (Computer Science & Engineering)

*Developed as a Capstone Project Report for the Bachelors of Technology (B.Tech) Degree Program.*
