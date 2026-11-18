# CrimeVision AI

> **A local research and decision-support prototype for comparing an uploaded image or a representative frame sampled from a video with a supplied witness statement.**  
> It combines frozen CLIP image–text representations, a quality-aware Optimal Transport (OT) alignment diagnostic, a split-conformal prediction layer, and a Neo4j case/evidence graph.

---

> [!WARNING]
> ### Research and Safety Boundary
> **CrimeVision AI is not a validated forensic instrument.**  
> Its outputs do not establish what physically occurred, identify a person, prove that a witness statement is true or false, or constitute a legal conclusion. The labels `MATCH` and `CONTRADICTION` are compatibility assessments evaluated using proxy labels described below. A qualified human reviewer must independently examine the original evidence. **Do not use this prototype as the sole basis for investigative, prosecutorial, or judicial action.**

---

## Table of Contents

- [What the Application Does](#what-the-application-does)
  - [Supported Uploads](#supported-uploads)
- [Current Implementation Status](#current-implementation-status)
- [System Architecture](#system-architecture)
  - [Docker Services and Ports](#docker-services-and-ports)
- [AI and Decision Pipeline](#ai-and-decision-pipeline)
  - [1. Frozen CLIP Representations](#1-frozen-clip-representations)
  - [2. Image Quality and Partial Optimal Transport](#2-image-quality-and-partial-optimal-transport)
  - [3. Active Split-Conformal Decision Mechanism](#3-active-split-conformal-decision-mechanism)
  - [4. Alternative Evidential Route](#4-alternative-evidential-route)
- [Dataset and Experimental Protocol](#dataset-and-experimental-protocol)
  - [Source and Label Construction](#source-and-label-construction)
  - [Supplied grouped-v3 Partitions](#supplied-grouped-v3-partitions)
  - [Dataset Availability and Reproduction Boundary](#dataset-availability-and-reproduction-boundary)
- [Recorded Research Results](#recorded-research-results)
  - [Active Three-Feature Conformal Artifact](#active-three-feature-conformal-artifact)
  - [Separate Full-OT vs. Partial-OT Comparison](#separate-full-ot-versus-partial-ot-comparison)
  - [Evaluation Caveats](#evaluation-caveats)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Run with Docker Compose](#run-with-docker-compose-recommended)
- [Run Services Locally for Development](#run-services-locally-for-development)
  - [1. Research Engine](#1-research-engine)
  - [2. Frontend](#2-frontend)
- [Research Reproduction Commands](#research-reproduction-commands)
- [API Overview](#api-overview)
- [Testing and Validation Status](#testing-and-validation-status)
- [Limitations and Security](#limitations-and-security)
- [Team](#team)

---

## What the Application Does

1. **Case Creation / Selection:** An investigator creates or selects a case in the web dashboard.
2. **Evidence & Statement Upload:** The investigator uploads an image or video and supplies a non-empty witness statement. The statement can be entered directly as text or uploaded as a UTF-8 `.txt` file.
3. **Integrity & Storage:** The research engine validates the media and statement, calculates a SHA-256 digest for the original evidence file, and stores the uploaded bytes separately from the graph metadata.
4. **Media Processing:**
   - **Images:** Analyzed directly.
   - **Videos:** The engine samples frames and selects one representative frame using a sharpness heuristic. *The current video path does not perform temporal or whole-video understanding.*
5. **Inference Pipeline:** Extracts CLIP features, calculates global similarity and patch/token alignment diagnostics, and applies the configured decision mechanism.
6. **Dashboard & Case Graph Review:** The dashboard displays the assessment, heatmap, and diagnostics, and exposes evidence and case relationships for human review.

The dashboard also provides case management, evidence records, case/evidence graph views, and cross-case relationship review.  
> [!NOTE]
> Cross-case suggestions are heuristic keyword-overlap candidates; they are **not** verified links between people, vehicles, weapons, or events.

### Supported Uploads

| Input | Supported Formats / Handling |
| :--- | :--- |
| **Images** | JPEG, PNG, BMP, WEBP, and TIFF |
| **Videos** | MP4, AVI, MOV, and MKV; sampled at 9 points by default between 10% and 90% of the clip |
| **Witness Statement** | Text entered in the form or a UTF-8 `.txt` file; the statement must be non-empty |

The video sampler chooses one decoded frame using an edge-energy-to-noise sharpness score. That single frame is passed to the image–text pipeline. The default maximum video duration is 300 seconds; resource limits can be changed through environment variables.

---

## Current Implementation Status

| Capability | Current Implementation |
| :--- | :--- |
| **Case management** | Create, retrieve, update, and delete cases through the FastAPI service and dashboard |
| **Image ingestion** | Validates supported image extensions against the decoded image format before analysis |
| **Video ingestion** | Samples multiple frames and selects one sharpness-based representative frame; it does not analyse temporal events |
| **Witness-statement input** | Accepts text or a `.txt` upload; rejects empty statements and does not insert a synthetic statement |
| **CLIP feature extraction** | Frozen `openai/clip-vit-base-patch32` backbone; extracts global image/text features, visual patch features, and text-token features |
| **Optimal Transport** | Log-domain Sinkhorn implementation with full-OT and partial-OT modes; the shipped runtime configuration selects partial OT |
| **Image-quality measurements** | Used to derive patch-level trust for partial OT; they are not themselves model epistemic uncertainty |
| **Conformal decision mechanism** | Active in the supplied configuration through the schema-3, recording-group-calibrated artifact `research-engine/models/uca_conformal_v3.json` |
| **Evidential classifier** | Alternative trainable head and calibration path exist in the code, but they are not configured as the active decision mechanism in the supplied `.env.example` |
| **Evidence integrity** | SHA-256 digest, evidence manifest, file retrieval, and hash-verification endpoints are implemented |
| **Case/evidence graph** | Neo4j integration with a process-local memory fallback and optional snapshot support |
| **Cross-case suggestions** | Heuristic suggestions based on shared text/keywords; investigator review is required |
| **Frontend** | Next.js/React dashboard with case tabs, evidence table, analysis panel, heatmap, and graph visualizations |
| **Real-world forensic validation** | Not established; the supplied experiments use UCF-Crime-derived proxy pairs |

> [!IMPORTANT]
> If neither a valid conformal artifact nor a trained evidential head plus its matching calibration file is loaded, the engine returns **`INDETERMINATE`** while still exposing available diagnostics. OT or CLIP similarity values must not be converted manually into a fallback verdict.

---

## System Architecture

The repository contains one FastAPI `research-engine` service. It is both the API layer and the host for the inference pipeline; it is not a separate business API plus a second independent ML server.

```
Investigator browser
       │
       ▼
Next.js 13 / React 18 dashboard
       │
       │  NEXT_PUBLIC_RESEARCH_API_URL
       ▼
FastAPI research engine (port 8001)
       │
       ├──► Request, statement and media validation
       │
       ├──► OpenCV / Pillow
       │      ├──► Image validation, or
       │      └──► Video frame sampling and representative-frame selection
       │
       ├──► SHA-256 hashing and evidence-file storage
       │
       ├──► Frozen Hugging Face CLIP
       │      ├──► Global image/text representations
       │      └──► Visual patch and text-token representations
       │
       ├──► Image-quality measurements and patch trust
       │
       ├──► Sinkhorn OT alignment diagnostics
       │
       ├──► Configured decision mechanism
       │      ├──► Split-conformal prediction (active in supplied configuration), or
       │      └──► Calibrated evidential head (alternative path)
       │
       └──► Neo4j case/evidence graph
              └──► Process-local memory fallback when Neo4j is unavailable
       │
       ▼
Assessment, diagnostics, evidence metadata, and graph data
       │
       ▼
Investigator dashboard
```

### Docker Services and Ports

| Service | Technology | Host Address |
| :--- | :--- | :--- |
| `frontend-web` | Next.js / React | `http://127.0.0.1:3000` |
| `research-engine` | FastAPI, CLIP, and OT pipeline | `http://127.0.0.1:8001` |
| `neo4j` | Neo4j Community 5.11 | Browser: `http://127.0.0.1:7474`<br>Bolt: `bolt://127.0.0.1:7687` |

> [!NOTE]
> The Docker Compose configuration binds published ports strictly to loopback (`127.0.0.1`). Do not expose these services on a public interface without adding appropriate authentication, authorization, network controls, and security review.

---

## AI and Decision Pipeline

### 1. Frozen CLIP Representations
The configured backbone is `openai/clip-vit-base-patch32`. Its parameters are put into evaluation mode and frozen; the project does not fine-tune the CLIP backbone as part of normal inference.

For each image–statement pair, the engine obtains:
- **Global image and text representations** for an overall similarity measurement.
- **Visual patch representations:** With the ViT-B/32 input configuration, the visual patch grid is $7 \times 7$ (49 patches).
- **Text-token representations** after removing special tokens.

The global cosine similarity is one of the features used by the conformal scorer. The patch and token representations are used to calculate local alignment diagnostics and the heatmap.

### 2. Image Quality and Partial Optimal Transport
The engine calculates image-quality measurements and patch sharpness. Under the supplied `TRUST_MODE=heterogeneity` configuration, patch trust is derived from relative patch sharpness. The trust values influence the partial-OT mass allocation.

- The OT solver builds a patch-to-token dissimilarity matrix and uses a log-domain Sinkhorn procedure.
- Its partial-transport implementation includes a dustbin mechanism so that some mass can remain unmatched.
- The engine reports diagnostics such as Sinkhorn divergence, discarded mass, the transport plan, and the patch/token cost matrix.

> [!CAUTION]
> The heatmap is a feature-alignment diagnostic. It is **not** an object detector, an object-location explanation, or proof that a particular word refers to a particular physical object in the image.

Full OT is also implemented and is used in comparative experiments. The supplied runtime artifact records `force_full_ot=false`, so the shipped inference configuration uses partial OT.

### 3. Active Split-Conformal Decision Mechanism
The supplied root `.env.example` selects the conformal path:
```env
EVIDENTIAL_HEAD_PATH=
EVIDENTIAL_CALIBRATION_PATH=
CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json
```

The conformal predictor uses three features, in this order:
1. `sinkhorn_divergence`
2. `discarded_mass`
3. `one_minus_global_clip_similarity`

The fit split is used to fit class-conditional Gaussian feature-density models for `MATCH` and `CONTRADICTION`. The calibration split is then used to calculate nonconformity scores. In schema 3, calibration is performed at the `source_recording_id` group level: the maximum true-label score within each calibration recording is used as that recording's calibration score. The finite-sample conformal quantile is then calculated across recording groups.

#### Artifact Configuration

| Artifact Setting | Value |
| :--- | :--- |
| **Artifact** | `research-engine/models/uca_conformal_v3.json` |
| **Schema** | `3` |
| **Alpha** ($\alpha$) | `0.10` |
| **Nominal Group-Level Coverage Target** | `90%` |
| **Calibration Groups** | 70 source recordings |
| **Calibration Pairs** | 140 |
| **$\hat{q}$ (q_hat)** | `0.6086335403` |
| **Feature Count** | 3 |

#### Decision Mapping
The predictor outputs a prediction set:
- $\{ \text{MATCH} \} \longrightarrow$ **`MATCH`**
- $\{ \text{CONTRADICTION} \} \longrightarrow$ **`CONTRADICTION`**
- $\{ \text{MATCH}, \text{CONTRADICTION} \} \longrightarrow$ **`INDETERMINATE`**
- $\emptyset$ (empty set) $\longrightarrow$ **`INDETERMINATE`**

This is a selective decision mechanism: it can abstain rather than force a binary assessment. The conformal guarantee is a simultaneous-within-recording-group coverage statement under exchangeability of recording groups and the same group-construction protocol. It is not class-conditional, does not guarantee that a singleton prediction is correct, does not bound the abstention rate, and does not protect against distribution shift.

### 4. Alternative Evidential Route
The repository includes an evidential pair head and associated calibration code under `research-engine/src/`. FastAPI supports an alternative configuration using both `EVIDENTIAL_HEAD_PATH` and `EVIDENTIAL_CALIBRATION_PATH` together. The evidential route and conformal route are mutually exclusive in the current inference initialization. The supplied configuration does not load the evidential checkpoint, so current application results should be described as conformal results, not as outputs from the evidential head.

---

## Dataset and Experimental Protocol

### Source and Label Construction
The grouped-v3 adapter uses UCF-Crime video frames and descriptions from `UCFCrime.json` to construct image–description pairs:
- **`MATCH`**: An image is paired with the annotation description from its own source recording.
- **`CONTRADICTION`**: An image is paired with a description from a different source recording and a different activity category within the same partition.

`CONTRADICTION` is therefore a synthetic cross-category description-mismatch proxy label. It is not a human-verified contradiction between real witness testimony and evidence. **The project must not describe these labels as forensic ground truth.**

### Supplied grouped-v3 Partitions
The current conformal artifact's recorded fit and calibration CSV hashes match the grouped-v3 dataset files. Their recorded split sizes are:

| Partition | Pairs | MATCH | CONTRADICTION | Source Recordings |
| :--- | :--- | :--- | :--- | :--- |
| **Fit** | 3,136 | 1,568 | 1,568 | 1,568 |
| **Calibration** | 140 | 70 | 70 | 70 |
| **Test** | 140 | 70 | 70 | 70 |

The grouped-v3 CSVs contain the following schema:
```csv
image_path,statement,label,camera_id,environment_id,source_recording_id,statement_source_recording_id,statement_source_environment_id
```

The supplied grouped-v3 CSVs have no overlap between fit, calibration, and test for either `source_recording_id` or `statement_source_recording_id`. This prevents the same source recording from being used across those partitions. It does not establish independent physical cameras: `camera_id` is derived from the source-recording identifier, not measured from camera hardware. `environment_id` represents a UCF-Crime activity category and may overlap across partitions.

### Dataset Availability and Reproduction Boundary
The repository package includes the grouped-v3 CSVs and experiment artifacts, but it does **not** include the complete original UCF-Crime/UCA image assets and source annotation dataset referenced by the CSV paths. A small number of selected review frames and contact sheets are present in the `reports` directory, but they are not a replacement for the full source dataset. 

Consequently, the metadata, split definitions, and saved results can be inspected, but the full CLIP feature extraction and model evaluation cannot be independently rerun from this archive alone. Restore the original dataset and preserve its directory structure before running research scripts.

---

## Recorded Research Results

> [!NOTE]
> The numbers below are transcribed from the supplied grouped-v3 reports. They are exploratory/retrospective results on this proxy dataset, not real-world forensic performance estimates.

### Active Three-Feature Conformal Artifact
Corresponds to the three-feature `all_three` grouped ablation and the supplied `uca_conformal_v3.json` artifact ($\alpha = 0.10$, $\hat{q} = 0.6086335403$):

| Metric | Recorded Result |
| :--- | :--- |
| **Test pairs** | 140 |
| **Test recording groups** | 70 |
| **Pair-level empirical coverage** | 93.57% |
| **Singleton prediction rate** | 30.00% |
| **Selective accuracy** | 78.57% |
| **Abstention rate** | 70.00% |
| **Average prediction-set size** | 1.70 |
| **Recording-group simultaneous coverage** | 61 / 70 groups = 87.14% |

The pair-level coverage and group-level simultaneous coverage are different summaries. The nominal 90% conformal target applies to simultaneous inclusion within an exchangeable recording group, not to per-pair correctness. The observed group-level coverage on this finite test split was 87.14%; it should be reported as observed, not rewritten as 90% measured accuracy or guaranteed per-case correctness.

### Separate Full-OT Versus Partial-OT Comparison
The grouped-v3 two-feature comparison below uses `sinkhorn_divergence` and `one_minus_global_clip_similarity`. It is a separate ablation and not the same three-feature configuration as the active artifact above.

| Metric | Full OT | Partial OT |
| :--- | :--- | :--- |
| **Pair-level empirical coverage** | 94.29% | 93.57% |
| **Singleton prediction rate** | 32.86% | 27.14% |
| **Selective accuracy** | 82.61% | 76.32% |
| **Abstention rate** | 67.14% | 72.86% |
| **Recording-group simultaneous coverage** | 88.57% | 87.14% |

*For this particular two-feature experiment, Full OT performed better on singleton rate and selective accuracy. This is an empirical comparison result, not evidence that Full OT will always outperform Partial OT or that either method is forensically valid.*

### Evaluation Caveats
- The supplied test split appears in earlier ablation and robustness artifacts. Results on this split are therefore retrospective/exploratory rather than an untouched confirmatory evaluation.
- The grouped-v3 label semantics are synthetic proxy labels, not human-adjudicated witness-statement contradictions.
- Recording-disjoint evaluation is not physical-camera-disjoint evaluation.
- Controlled image-degradation experiments are stress tests on the proxy data; they do not establish robustness to real CCTV conditions or semantic reliability.
- The recorded results cannot be independently recomputed from this archive because the full referenced image assets are absent.

---

## Repository Layout

```text
CrimeVision-AI/
├── frontend-web/                         # Next.js / React investigator dashboard
│   ├── src/components/                   # Case, evidence, graph, heatmap and analysis UI
│   ├── src/pages/                        # Next.js pages and routing
│   └── tests/                            # Vitest / Testing Library frontend tests
├── research-engine/                      # FastAPI service and AI research pipeline
│   ├── main.py                           # FastAPI routes, ingestion and orchestration
│   ├── src/
│   │   ├── inference.py                  # CLIP feature extraction and inference routing
│   │   ├── conformal.py                  # Schema-3 conformal model and prediction sets
│   │   ├── evidential_fusion.py          # Alternative evidential head and OT implementation
│   │   └── image_quality.py              # Image quality and patch-sharpness features
│   ├── datasets/
│   │   ├── uca/                          # Legacy UCA CSVs
│   │   └── uca_group_v3/                 # Current grouped-v3 CSVs and pairing manifest
│   ├── models/                           # Conformal artifacts, including uca_conformal_v3.json
│   ├── reports/                          # Metrics, audits, ablations and robustness reports
│   ├── tests/                            # Pytest suite
│   ├── calibrate_conformal.py            # Fit density models and calibrate q-hat
│   ├── evaluate_research_engine.py       # Evaluate a saved decision artifact
│   └── neo4j_connector.py                # Graph persistence interface
├── docker-compose.yml                    # Neo4j, research-engine and frontend services
├── .env.example                          # Safe configuration template
├── RESEARCH_PROTOCOL.md                  # Research protocol and claim boundaries
├── VALIDATION_STATUS.md                  # Supplied validation record and limitations
└── README.md
```

---

## Requirements

- **Docker & Docker Compose** (Recommended for full-stack deployment).
- **For local host development:**
  - Python 3.10 or newer (the `research-engine` Docker image uses Python 3.11).
  - Node.js 18 or newer, and `npm`.
- **Disk Space:** Sufficient storage for the CLIP model cache, Docker images/volumes, and any locally restored research datasets.
- **Internet Access:** Required on first model load to download the Hugging Face CLIP weights, unless already cached locally.
- **GPU (Optional):** `DEFAULT_DEVICE=auto` selects CUDA when available to the process and falls back to CPU. GPU passthrough must be configured separately if running in Docker.

---

## Run with Docker Compose (Recommended)

Run commands from the repository root:

### 1. Create and Configure Environment Variables
```bash
# PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```
Edit `.env` and replace `YOUR_ACTUAL_NEO4J_PASSWORD` with a strong local password. Keep `.env` private and never commit it.

The supplied `.env.example` pre-configures the schema-3 artifact:
```env
CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json
```
*(The Compose file mounts `research-engine/models/` read-only at `/app/models/` inside the container).*

### 2. Build and Start Services
```bash
docker compose up --build -d
```

Verify service status:
```bash
docker compose ps
```

Follow research engine startup logs:
```bash
docker compose logs -f research-engine
```
*(Initial startup may take longer while Hugging Face weights are downloaded).*

### 3. Open the Application
| Service / Resource | URL |
| :--- | :--- |
| **Investigator Dashboard** | [http://127.0.0.1:3000](http://127.0.0.1:3000) |
| **FastAPI Swagger Docs** | [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) |
| **FastAPI Health Check** | [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health) |
| **FastAPI Readiness Check** | [http://127.0.0.1:8001/ready](http://127.0.0.1:8001/ready) |
| **Neo4j Browser UI** | [http://127.0.0.1:7474](http://127.0.0.1:7474) |

### 4. Stopping the Services
```bash
docker compose down
```
To remove persisted database and evidence volumes as well:
```bash
docker compose down -v
```
*(Caution: Deletes all persisted cases, evidence files, and Neo4j data).*

---

## Run Services Locally for Development

For iterative development, run Neo4j first (e.g. `docker compose up -d neo4j`) and execute the services locally:

### 1. Research Engine
```powershell
cd research-engine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables (PowerShell)
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "YOUR_LOCAL_NEO4J_PASSWORD"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
$env:DEFAULT_DEVICE = "auto"
$env:CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
$env:CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
$env:TRUST_MODE = "heterogeneity"
$env:CONFORMAL_ARTIFACT_PATH = "models/uca_conformal_v3.json"

# Start FastAPI server
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```
*(For bash/zsh, use equivalent `export VARIABLE=value` syntax).*

### 2. Frontend
In a second terminal:
```bash
cd frontend-web
npm ci

# Create frontend environment file
echo "NEXT_PUBLIC_RESEARCH_API_URL=http://127.0.0.1:8001" > .env.local

# Run development server
npm run dev
```
Open [http://127.0.0.1:3000](http://127.0.0.1:3000) in your browser.

---

## Research Reproduction Commands

> [!NOTE]
> The following commands require the original UCA/UCF-Crime image assets to be restored and the `image_path` entries in the CSV files to resolve correctly. Run these commands from `research-engine/`.

### Refit a Schema-3 Recording-Group Conformal Artifact
```bash
python calibrate_conformal.py \
  --fit-csv datasets/uca_group_v3/fit.csv \
  --calibration-split-csv datasets/uca_group_v3/calibration.csv \
  --output models/uca_conformal_rebuilt.json \
  --alpha 0.10 \
  --model-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --trust-mode heterogeneity
```

### Evaluate on the Separate Test Partition
```bash
python evaluate_research_engine.py \
  --conformal models/uca_conformal_rebuilt.json \
  --test-csv datasets/uca_group_v3/test.csv \
  --output reports/uca_group_v3_test_metrics.json \
  --sweep-alpha 0.05,0.10,0.20,0.30 \
  --model-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
```

> [!IMPORTANT]
> Do not tune the model, select features, or choose trust parameters using the test partition. Retain CSV hashes, experiment manifests, and raw metrics reports with any results presented to reviewers.

#### Additional Experiment Scripts:
- `ablate_conformal_scores.py` — Compares conformal feature subsets.
- `ablate_full_vs_partial_ot.py` — Compares Full OT and quality-weighted Partial OT.
- `evaluate_image_robustness.py` & `evaluate_image_robustness_bootstrap.py` — Evaluates image degradation settings and recording-group resampling uncertainty.
- `datasets/adapt_uca.py` & `datasets/prepare_ucf_crime_proxy.py` — Dataset adaptation and proxy-pair preparation.

See `RESEARCH_PROTOCOL.md` for full protocol details, manifests, resampling units, and claim boundaries.

---

## API Overview

The complete endpoint schemas are available through the interactive Swagger UI at `/docs`.

| Method | Route | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check |
| `GET` | `/ready` | Service readiness status |
| `GET`, `POST` | `/cases` | List and create cases |
| `PUT`, `DELETE`| `/cases/{case_id}` | Update or delete a case |
| `GET` | `/graph/{case_id}` | Retrieve a case graph |
| `GET` | `/graph/all` | Retrieve the global cross-case graph |
| `POST` | `/cross_case_relation` | Record a cross-case relationship for human review |
| `GET` | `/cross_case_relations` | Retrieve recorded cross-case relationships |
| `GET` | `/cross_case_suggestions` | Retrieve heuristic keyword-overlap candidates |
| `POST` | `/analyze_evidence/` | Ingest and analyse evidence with a witness statement |
| `GET` | `/evidence/{case_id}` | Retrieve evidence metadata for a case |
| `GET` | `/evidence/{case_id}/manifest` | Retrieve the evidence manifest |
| `GET` | `/evidence/{case_id}/{sha256}/file` | Retrieve a stored evidence file after integrity checking |
| `POST` | `/evidence/{case_id}/verify` | Compare an uploaded file's SHA-256 with stored evidence record |
| `GET` | `/admin/memory` | Inspect memory-fallback status |
| `POST` | `/admin/memory/flush` | Flush the configured memory snapshot |

---

## Testing and Validation Status

### Running Tests Locally

#### Research Engine Tests
```bash
cd research-engine
python -m pytest -q
```

#### Frontend Tests and Build
```bash
cd frontend-web
npm ci
npm test
npm run build
```

### Existing Validation Record
`VALIDATION_STATUS.md` records the following status from the supplied remediation run:
- **Research Engine Tests:** 103 passed, 1 skipped (skipped test requires a live Neo4j test server).
- **Python Source Syntax & Compileall:** Passed for all checked research-engine files.
- **Synthetic UCA Adapter Smoke Run & Audit:** Passed source-recording split audit.
- **Docker Compose Ports:** Verified as loopback-bound (`127.0.0.1`).

*Items noted as not verified in that run:*
- Full real-UCA CLIP inference and updated model metrics (due to absence of complete source image assets).
- Live Neo4j transaction/integration behavior.
- Frontend `npm ci`, Vitest execution, and Next.js production build.
- Public network security, production authentication/authorization, penetration testing, and evidence governance.

---

## Limitations and Security

- **Not a Forensic Instrument:** The system is an academic research prototype. It must not be treated as proof, forensic identification, or legal evidence of what occurred.
- **Proxy Labels:** `MATCH` and `CONTRADICTION` in the experiments are constructed UCF-Crime description-pair proxy labels, not human-adjudicated real-world witness testimony.
- **No Identity Recognition:** The project does not identify suspects or perform facial recognition.
- **No Object Detector or Grounded Explanation:** Patch/token heatmaps show feature alignment diagnostics, not verified physical object boundaries or semantic explanations.
- **Limited Video Analysis:** Selects one representative frame using sharpness; temporal dynamics and sequential events across the video clip are not modeled.
- **Heuristic Cross-Case Candidates:** Shared keywords are heuristic review candidates, not verified entities or calibrated probabilities.
- **No Active LLM Analysis:** `LLMAnalysisPanel` is a frontend UI presentation component for diagnostics; an external LLM is not part of the active inference pipeline.
- **Authentication Not Implemented:** The current login view is a client-side demonstration gate. There is no server-side user authentication, RBAC, or case-level authorization.
- **Local Prototype Only:** Do not expose this application on a public or shared network. Loopback binding reduces accidental exposure but does not substitute for network security and access control.
- **Evidence Integrity Boundary:** SHA-256 verifies byte fidelity against a stored digest; it does not constitute a signed, append-only legal chain of custody or evidence governance.
- **Persistence Fallback:** When Neo4j is unavailable, the system falls back to in-memory storage with optional snapshots, which is not durable production database persistence.
- **Conformal Coverage Bounds:** Group-level coverage relies on exchangeability of recording groups under the exact same grouping protocol. It is not class-conditional, does not bound abstention rates, and does not guarantee robustness under distribution shift.
- **Evaluation Boundary:** The test split was previously utilized in exploratory ablations; results should be interpreted retrospectively.

> [!CAUTION]
> Before sharing or pushing the repository, remove any local `.env` file and confidential credentials. Use `.env.example` as the distribution template.

---

## Team

- **Shyamal** — Computer Science & Engineering
- **Moukthika** — Computer Science & Engineering
- **Venu** — Computer Science & Engineering
- **Niteesha** — Computer Science & Engineering

*Developed as a B.Tech Capstone Research Prototype.*