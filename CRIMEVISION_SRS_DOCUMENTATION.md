# Software Requirements Specification (SRS)

## CrimeVision AI — Multimodal Evidence Comparison and Investigation Decision-Support Prototype

**Document Version:** `1.2.0` (Current-Repository Implementation Alignment)  
**Project Type:** B.Tech Capstone Research Prototype  
**Development Team:** Shyamal, Moukthika, Venu, Niteesha  
**Status:** Prototype implementation with saved UCA-derived grouped-conformal research results; independent real-world forensic validation pending  

---

> [!WARNING]
> ### Scope and Safety Statement
> **CrimeVision AI is a research and investigation decision-support prototype.**  
> It does not establish what physically occurred, verify a suspect's identity, establish guilt, or create court-admissible findings. Automated outputs are **not** forensic proof. Original evidence, acquisition provenance, independent examination, and qualified human judgment remain necessary.

---

### Document Purpose and Conventions

This document describes the current repository implementation and distinguishes implemented behaviour from research scaffolding and future work. A feature described as **implemented** means corresponding source code exists; it does **not** imply production hardening, independent validation, or a measured performance guarantee.

#### Status Labels
- `Implemented (prototype)`: Source implementation exists in the current repository.
- `Scaffolded`: Training, calibration, or integration code exists, but it still needs real data or environment-level validation.
- `Planned / not implemented`: The current repository does not provide the described capability.
- `Not validated`: The capability has not been demonstrated on a suitable held-out dataset or in a production environment.

#### Mathematical Notation
- Vectors are bold lowercase ($\mathbf{v}, \mathbf{t}, \mathbf{x}$).
- Matrices are bold uppercase ($\mathbf{C}, \mathbf{P}$).
- Probability distributions are represented by Greek or calligraphic symbols.

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Purpose](#11-purpose)
  - [1.2 Product Scope](#12-product-scope)
  - [1.3 Intended Audience](#13-intended-audience)
  - [1.4 Definitions](#14-definitions)
- [2. Overall Description and Architecture](#2-overall-description-and-architecture)
  - [2.1 System Context](#21-system-context)
  - [2.2 Logical Architecture](#22-logical-architecture)
  - [2.3 Deployment Boundaries](#23-deployment-boundaries)
  - [2.4 User Classes](#24-user-classes)
  - [2.5 Dependencies and Constraints](#25-dependencies-and-constraints)
- [3. Technology Stack and Operating Environment](#3-technology-stack-and-operating-environment)
- [4. Data Model and Persistence](#4-data-model-and-persistence)
  - [4.1 Case Records](#41-case-records)
  - [4.2 Evidence and Statement Records](#42-evidence-and-statement-records)
  - [4.3 Case-Scoped Identifiers and Legacy Records](#43-case-scoped-identifiers-and-legacy-records)
  - [4.4 Persistence and Fallback Semantics](#44-persistence-and-fallback-semantics)
  - [4.5 Cross-Case Relations and Heuristic Candidates](#45-cross-case-relations-and-heuristic-candidates)
- [5. Mathematical and Algorithmic Formulation](#5-mathematical-and-algorithmic-formulation)
  - [5.1 CLIP Feature Extraction](#51-clip-feature-extraction)
  - [5.2 Cross-Modal Cost Matrix](#52-cross-modal-cost-matrix)
  - [5.3 Entropic and Debiased Sinkhorn Transport](#53-entropic-and-debiased-sinkhorn-transport)
  - [5.4 Quality-Aware Partial OT and Full OT Comparison](#54-quality-aware-partial-ot-and-full-ot-comparison)
  - [5.5 Pair Features and Image-Quality Features](#55-pair-features-and-image-quality-features)
  - [5.6 Alternative Evidential Pair-Level Classifier](#56-alternative-evidential-pair-level-classifier-implemented-not-active-in-supplied-configuration)
  - [5.7 Active Split-Conformal Decision Mechanism](#57-active-split-conformal-decision-mechanism)
  - [5.8 Heatmap Interpretation](#58-heatmap-interpretation)
- [6. Functional Requirements and Implementation Status](#6-functional-requirements-and-implementation-status)
- [7. External Interfaces](#7-external-interfaces)
  - [7.1 Frontend](#71-frontend)
  - [7.2 Research-Engine HTTP API](#72-research-engine-http-api)
  - [7.3 Authentication and Identity Boundary](#73-authentication-and-identity-boundary)
- [8. Non-Functional Requirements and Constraints](#8-non-functional-requirements-and-constraints)
  - [8.1 Performance](#81-performance)
  - [8.2 Security and Privacy](#82-security-and-privacy)
  - [8.3 Reliability and Persistence](#83-reliability-and-persistence)
  - [8.4 Research Reproducibility](#84-research-reproducibility)
  - [8.5 Human Oversight](#85-human-oversight)
- [9. Verification and Validation Plan](#9-verification-and-validation-plan)
  - [9.1 Validation Evidence Recorded for Supplied Repository Snapshot](#91-validation-evidence-recorded-for-supplied-repository-snapshot)
  - [9.2 Saved UCA group-v3 Conformal Evaluation](#92-saved-uca-group-v3-conformal-evaluation-artifact-derived)
  - [9.3 Saved Full OT vs. Partial OT Comparison](#93-saved-full-ot-versus-partial-ot-comparison-two-feature-ablation)
- [10. Limitations and Future Research](#10-limitations-and-future-research)
  - [10.1 Retired Experimental Artifacts](#101-retired-experimental-artifacts)
  - [10.2 UCA Proxy Dataset and Interpretation Limits](#102-uca-proxy-dataset-and-interpretation-limits)
  - [10.3 Capabilities Not Implemented or Not Validated](#103-capabilities-not-implemented-or-not-validated)

---

## 1. Introduction

### 1.1 Purpose
CrimeVision AI is a research and investigation decision-support prototype for comparing a still image, or one representative frame sampled from a video, with a supplied natural-language witness statement. The current configured analysis path uses a frozen CLIP vision-language backbone, quality-aware Partial Optimal Transport (OT) diagnostics, and a split-conformal prediction wrapper. The application also provides case/evidence management, local evidence-file preservation, a Neo4j case/evidence graph, and an investigator dashboard.

A supervised evidential pair-level neural head is implemented as an alternative research path. It is not the active decision mechanism in the supplied project configuration. This distinction is critical: the current conformal prediction result must not be described as an output from the evidential head.

The current implementation is intended for research prototyping, software integration, and evaluation-method development. It is **not** an operational forensic system.

### 1.2 Product Scope
The current research-engine API supports the following prototype workflow:
1. An investigator creates or selects a case in the Next.js dashboard.
2. The investigator uploads one supported image or video and supplies a genuine statement either as text or as a UTF-8 `.txt` file. The API rejects a request that does not contain a statement; it does not insert a synthetic default statement.
3. The API validates the case ID, filename, upload size, declared media type, and decoded media:
   - Supported image extensions: `JPG`/`JPEG`, `PNG`, `BMP`, `WEBP`, `TIF`, and `TIFF`.
   - Supported video extensions: `MP4`, `AVI`, `MOV`, and `MKV`.
   - Default limits: evidence upload max 100 MiB; statements max 64 KiB and 20,000 characters.
4. For a video, OpenCV samples a configurable number of approximately uniform frame positions (default $N = 9$) spanning 10% to 90% of the clip. It selects the sharpest successfully decoded sample using an edge-energy/noise heuristic and returns a sampling report containing the attempted positions, frame indices, timestamps, sharpness values, and selected frame. Default maximum video duration is 300 seconds. *This is representative-frame selection, not temporal event understanding.*
5. Before inference, the original uploaded bytes are stored by the evidence store under a case-scoped, SHA-256-derived path. The store verifies the digest on write and on read. Analysis failure after this point does not imply that the original file was not preserved.
6. Hugging Face CLIP produces image-patch features, text-token features, and global image/text features. The current ViT-B/32 configuration yields a $7 \times 7$ grid (49) of image patches before alignment.
7. A log-domain Sinkhorn implementation calculates a debiased Sinkhorn divergence and partial-transport diagnostics. The configured inference path uses quality-aware Partial OT with a dustbin/unmatched-mass mechanism. These values are diagnostics and are not independently converted into a contradiction verdict.
8. In the supplied configuration, a split-conformal artifact (`research-engine/models/uca_conformal_v3.json`) is the active decision mechanism. It uses three features—Sinkhorn divergence, discarded mass, and one minus global CLIP similarity—and returns a prediction set. A singleton set produces `MATCH` or `CONTRADICTION`; a two-label or empty set produces `INDETERMINATE`. If neither a conformal artifact nor a compatible trained evidential head with calibration is configured, the engine returns `INDETERMINATE` while still providing diagnostics.
9. The evidential head plus its matching calibration artifact is an alternative decision mechanism. The inference engine rejects a configuration that enables both the evidential and conformal paths simultaneously.
10. Case/evidence metadata and analysis artefacts are written to Neo4j when available. When Neo4j is unavailable, graph records are held in a process-local memory store and are explicitly reported as non-durable. An optional memory snapshot can be restored/flushed, but it is not equivalent to transactional Neo4j persistence.
11. The Next.js dashboard displays the assessment status, diagnostics, evidence table, patch/token heatmap, case graph, and storage/persistence status.

> [!NOTE]
> The current implementation does not provide a validated object detector, object/attribute verifier, supervised region-word grounding model, automatic suspect identification, temporal video-event understanding, or an end-to-end forensic chain-of-custody service. A SHA-256 check can detect a changed file; it does not establish who acquired the evidence or make storage immutable.

### 1.3 Intended Audience
- **Project Evaluators:** Assess the relationship between stated research aims, source implementation, and saved experiments.
- **AI/ML Developers:** Maintain CLIP feature extraction, OT/Partial OT, the conformal pipeline, and the alternative evidential-head implementation.
- **Dataset & Research-Evaluation Developers:** Maintain UCA proxy-pair construction, source-recording splits, provenance audits, ablations, bootstrap analysis, and robustness experiments.
- **Backend/Database Developers:** Maintain the FastAPI research engine, evidence-file store, and Neo4j graph persistence.
- **Frontend Developers:** Maintain the Next.js case workspace, upload flow, visualizations, and presentation of uncertainty and persistence state.
- **Investigative-Domain Reviewers:** Assess data suitability, label definitions, and the operational boundaries of any proposed use.

### 1.4 Definitions

| Term | Meaning in this Project |
| :--- | :--- |
| **CLIP** | Contrastive Language–Image Pretraining model used as a frozen visual/text feature backbone |
| **OT** | Optimal Transport |
| **Partial OT** | OT formulation permitting some mass to remain unmatched through a dustbin/discarded-mass mechanism |
| **Sinkhorn Solver** | Entropy-regularized OT solver implemented in log space |
| **Sinkhorn Divergence** | Debiased combination of cross-modal and self-transport objectives; an alignment diagnostic |
| **Dustbin / Discarded Mass** | Mechanism and diagnostic for visual or textual mass not matched by the partial-transport plan |
| **Split-Conformal Prediction** | Calibration procedure forming a prediction set using a quantile computed from held-out calibration nonconformity scores |
| **Prediction Set** | Set of candidate labels retained by the conformal predictor for an input |
| **Conformal Calibration Unit** | Unit on which calibration scores are aggregated; supplied v3 artifact uses the maximum true-label score per `source_recording_id` |
| **Evidential Head** | Supervised two-class neural head outputting Dirichlet concentration parameters; not active in supplied configuration |
| **Model Uncertainty** | Dirichlet-based uncertainty estimate from the optional evidential head; not a guarantee of correctness |
| **Proxy Label** | Constructed research label used as a stand-in for a difficult concept; `CONTRADICTION` is a synthetic cross-recording, cross-category mismatch |
| **Image-Quality Metrics** | Descriptive measurements (high-frequency residual, Laplacian variance, luminance standard deviation); not epistemic uncertainty |
| **OOD** | Out-of-distribution relative to a training distribution or configured feature range |
| **Abstention** | Returning `INDETERMINATE` rather than forcing a binary classification |
| **Evidence Hash** | SHA-256 digest of original uploaded bytes; an integrity identifier, not full chain-of-custody proof |
| **Volatile Fallback** | Process-local graph memory used when Neo4j is unavailable; reported as non-durable |

---

## 2. Overall Description and Architecture

### 2.1 System Context
The supplied repository contains one working application API: the FastAPI research engine in `research-engine/main.py`, exposed on port `8001`. The Next.js dashboard in `frontend-web/` communicates with that API. Neo4j serves as the graph database. The current repository snapshot does not contain a separate `backend/` service, a MongoDB service, or a second active API on port `8000`.

The root `docker-compose.yml` defines three runtime services: `neo4j`, `research-engine`, and `frontend-web`. Host-published ports are strictly bound to loopback (`127.0.0.1`) in the supplied Compose file. This represents a local-development/research deployment boundary, not a production network-security design.

### 2.2 Logical Architecture

```
Investigator Browser
  Next.js 13 / React 18 Dashboard
       │
       │ NEXT_PUBLIC_RESEARCH_API_URL
       │ multipart evidence + statement upload; JSON case/graph requests
       ▼
Research Engine API (FastAPI / Uvicorn, port 8001)
       │
       ├──► Case ID, filename, size, content-type and decoded-media validation
       ├──► Original-byte SHA-256 calculation and case-scoped EvidenceStore
       ├──► OpenCV video sampling / Pillow image decoding
       ├──► Frozen Hugging Face CLIP ViT-B/32
       │      ├──► Image patch features (7 × 7 = 49 for the configured model)
       │      ├──► Text token features
       │      └──► Global image/text features
       ├──► Quality-aware Partial OT / log-domain Sinkhorn diagnostics
       │      ├──► Patch trust weights
       │      ├──► Dustbin / discarded mass
       │      └──► Divergence, transport plan and patch-token cost matrix
       ├──► Decision mechanism (mutually exclusive configuration)
       │      ├──► Supplied active path: split-conformal prediction
       │      │      ├──► Class-conditional Gaussian feature densities
       │      │      ├──► Recording-group calibration threshold
       │      │      └──► MATCH / CONTRADICTION / INDETERMINATE
       │      ├──► Alternative path: trained evidential head + calibration
       │      └──► Neither configured: INDETERMINATE-only diagnostics
       ├──► Neo4j case/evidence/statement graph
       │      └──► Process-local memory fallback; optional snapshot
       └──► Case-scoped filesystem evidence storage

Docker Compose Services:
  Neo4j + Research Engine API + Next.js Frontend
```

The research engine is both the HTTP API and the inference host. OT diagnostics do not independently generate an assessment verdict; the verdict comes strictly from the active decision mechanism (or evaluates to `INDETERMINATE`).

### 2.3 Deployment Boundaries
- **Frontend Gateway:** The dashboard points to `NEXT_PUBLIC_RESEARCH_API_URL` (`http://127.0.0.1:8001`).
- **Graph Connection:** `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` configure the Neo4j bolt driver.
- **CLIP Model:** Configured via `CLIP_MODEL_NAME`, optional `CLIP_MODEL_REVISION`, and `DEFAULT_DEVICE`. Default is pinned `openai/clip-vit-base-patch32`.
- **Conformal Decision:** Configured via `CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json`. `EVIDENTIAL_HEAD_PATH` and `EVIDENTIAL_CALIBRATION_PATH` are empty.
- **Mutual Exclusivity:** The engine accepts either both evidential paths OR the conformal path. If neither is configured, inference returns diagnostics with `INDETERMINATE` status.
- **Trust Mode:** `TRUST_MODE=heterogeneity`, `OT_TRUST_TAU`, and `OT_TRUST_K` govern patch trust weighting.
- **Storage Roots:** `EVIDENCE_STORAGE_ROOT` dictates case filesystem storage. Docker Compose mounts named volumes for evidence, Neo4j data, and memory snapshots.
- **Local Access Only:** Services are intended for local loopback; public exposure without an authenticating reverse proxy is unsafe.

### 2.4 User Classes
The prototype UI assumes an investigator/reviewer who creates and manages cases, uploads media and statements, inspects diagnostic heatmaps, and browses case graphs. The login screen is a client-side demonstration gate and does not authenticate API requests. The API does not implement multi-user role-based access control (RBAC).

### 2.5 Dependencies and Constraints
- Python 3.10+ (Docker image uses 3.11), PyTorch, Transformers, OpenCV, Pillow, FastAPI, Uvicorn, and `neo4j` Python driver.
- Node.js 18+ and `npm` for the Next.js frontend.
- Docker Engine/Desktop with Docker Compose.
- First-time CLIP initialization downloads weights from Hugging Face unless cached.
- CUDA GPU is optional; defaults to CPU if unavailable.
- Neo4j writes fall back to process memory if unreachable.
- UCA research CSVs reference complete source image sets not fully included in archive packages; full real-image reruns require external asset restoration.

---

## 3. Technology Stack and Operating Environment

| Layer | Repository Technologies | Implementation Notes |
| :--- | :--- | :--- |
| **Research Engine & API** | Python 3.11, FastAPI, Uvicorn, Pydantic | Single service provides HTTP API and hosts the inference pipeline |
| **Vision-Language Backbone** | `openai/clip-vit-base-patch32` via Hugging Face Transformers | Frozen CLIP backbone; pinned commit revision in supplied configuration |
| **Numerical Computation** | PyTorch, NumPy, SciPy | Feature operations, density modeling, numerical routines, and evaluation |
| **Optimal Transport** | Custom log-domain Sinkhorn in PyTorch | Full OT comparison path and quality-aware Partial OT with dustbin mechanism |
| **Conformal Prediction** | `research-engine/src/conformal.py` | Class-conditional Gaussian feature densities and split-conformal prediction sets |
| **Alternative Evidential Head** | `research-engine/src/evidential_fusion.py` | Trainable two-class Dirichlet head; implemented as alternative research path |
| **Image / Video Processing** | OpenCV, Pillow | Image validation/decoding, quality metrics, and $N$-point video sampling |
| **Evidence-File Storage** | Filesystem `EvidenceStore` | Case-scoped content-addressed storage, SHA-256 verification, manifest, and retrieval |
| **Graph Database** | Neo4j 5.x with Neo4j Python driver | Case, visual-evidence, witness-statement, analysis, and cross-case relationships |
| **Frontend Dashboard** | Next.js 13, React 18, Tailwind CSS, React Flow, Recharts, Lucide-React | Case dashboard, evidence upload, interactive graph, heatmap, and diagnostics |
| **Testing Framework** | `pytest` (backend), Vitest (frontend configured) | Unit and integration suites for backend engine and frontend components |
| **Containerization** | Docker Compose | Multi-container definitions for Neo4j, research-engine, and frontend |

---

## 4. Data Model and Persistence

### 4.1 Case Records
A `Case` node stores a case identifier, name, description, category, and creation timestamp. `case_id` is the root identifier for case-scoped data. The dashboard supports creating, updating, switching between, clearing evidence from, and deleting cases through the API.

### 4.2 Evidence and Statement Records
Original uploaded bytes are preserved separately from graph metadata. The filesystem store is case-scoped and content-addressed:
$$\text{Path} = \text{StorageRoot} / \text{case\_id} / (\text{sha256} + \text{ext})$$

The store verifies the SHA-256 digest before writing via a temporary file with an atomic replace, and re-verifies upon read.

#### Graph Pattern
```
(:Case)-[:CONTAINS]->(:VisualEvidence)
(:Case)-[:CONTAINS]->(:WitnessStatement)
(:WitnessStatement)-[:CORRELATES_TO]->(:VisualEvidence)
```

The `CORRELATES_TO` relationship stores:
- `sinkhorn_cost`: Debiased Sinkhorn divergence diagnostic.
- `assessment_status`: `MATCH`, `CONTRADICTION`, or `INDETERMINATE`.
- `flagged_contradiction`: Nullable boolean derived from explicit status.
- `model_epistemic_uncertainty` and class probabilities (when evidential route is used).
- JSON-serialized visual-quality metrics and OT diagnostics.
- Patch/token heatmap matrix, token labels, and patch labels.
- Model name/revision, calibration metadata, evidence hash, file size, timestamps, investigator advisory, and audit report.

> [!NOTE]
> The `X-Investigator-Id` header is assertion-only metadata, not verified identity. The graph connector does not derive a decision from a Sinkhorn threshold; it stores the explicit assessment returned by the configured decision engine.

### 4.3 Case-Scoped Identifiers and Legacy Records
Case-scoped IDs prevent nodes from merging when two cases share a filename or statement text. Unique constraints on `Case.id`, `VisualEvidence.id`, and `WitnessStatement.id` are applied on startup.

Clearing a case removes case ownership relationships and deletes owned nodes. It intentionally does not delete unowned legacy nodes to avoid impacting other cases.

### 4.4 Persistence and Fallback Semantics
- **Neo4j Available:** Writes execute transactionally and report `persisted=true` (`storage_backend="neo4j"`).
- **Neo4j Unavailable:** Writes fall back to in-memory store (`persisted=false`, `storage_backend="memory"`, non-durable).
- **Memory Snapshot:** Optional snapshot path loads on startup and flushes on shutdown or via `/admin/memory/flush`. Snapshots do not provide ACID guarantees or transactional logging.
- **Decoupled Outcomes:** Evidence-file write and graph write are separate. An API `status="success"` indicates request completion, not guaranteed database durability.
- **Clearing vs. Deletion:** Clearing a case retains the case node; deletion permanently removes the case.

### 4.5 Cross-Case Relations and Heuristic Candidates
Manual cross-case links are stored as `CROSS_CASE_CORRELATION` with relation type, notes, and optional strength. Investigator strength ratings are not calibrated probabilities.

The suggestion endpoint performs keyword overlap matching. Suggestions are heuristic review candidates and do not imply true shared suspects, vehicles, or modus operandi.

---

## 5. Mathematical and Algorithmic Formulation

### 5.1 CLIP Feature Extraction
Uses `openai/clip-vit-base-patch32` as a frozen backbone:
- Visual patch grid: For $224 \times 224$ inputs, ViT-B/32 produces $7 \times 7 = 49$ visual patches (excluding `[CLS]`).
- Text tokens: Sequence of token embeddings (excluding special tokens).
- Global representations: Projected global image and text vectors.

Features are projected into the shared embedding space and $\ell_2$-normalized:
$$\mathbf{v}_i = \frac{f_v(I_i)}{\|f_v(I_i)\|_2}, \qquad \mathbf{t}_j = \frac{f_t(T_j)}{\|f_t(T_j)\|_2}$$

### 5.2 Cross-Modal Cost Matrix
Let normalized patch features be $V = \{\mathbf{v}_i\}_{i=1}^N$ ($N=49$) and token features be $T = \{\mathbf{t}_j\}_{j=1}^M$. The pairwise dissimilarity cost is:

$$C_{ij} = \max\left(0, 1 - \mathbf{v}_i^\top \mathbf{t}_j\right)$$

The matrix $\mathbf{C} \in \mathbb{R}^{N \times M}$ represents exploratory feature dissimilarity. It is **not** supervised phrase grounding or object localization.

### 5.3 Entropic and Debiased Sinkhorn Transport
For probability marginals $a \in \Delta^N$ and $b \in \Delta^M$, the entropy-regularized transport problem is:

$$\operatorname{OT}_{\epsilon}(a, b; \mathbf{C}) = \min_{\mathbf{P} \ge 0, \; \mathbf{P}\mathbf{1} = a, \; \mathbf{P}^\top\mathbf{1} = b} \left[ \langle \mathbf{P}, \mathbf{C} \rangle + \epsilon \, \mathrm{KL}(\mathbf{P} \,\|\, a b^\top) \right]$$

Solved in log-space for numerical stability. The debiased Sinkhorn divergence is computed as:

$$S_\epsilon(V, T) = \operatorname{OT}_\epsilon(V, T) - \frac{1}{2}\operatorname{OT}_\epsilon(V, V) - \frac{1}{2}\operatorname{OT}_\epsilon(T, T)$$

Small numerical negatives are clamped to zero. The divergence is an alignment diagnostic; there is no universal fixed threshold (e.g., 1.15) for contradiction.

### 5.4 Quality-Aware Partial OT and Full OT Comparison
Partial OT incorporates a virtual dustbin mass to permit unaligned patches/tokens without penalty:
- Mass is scaled by patch-level trust weights derived from sharpness under `TRUST_MODE=heterogeneity`.
- The reported `discarded_mass` measures unaligned visual/textual content.
- Shipped runtime settings: $\epsilon = 0.1$, dustbin cost $= 10.0$.

### 5.5 Pair Features and Image-Quality Features
The alternative evidential classifier constructs a 1,025-dimensional pair representation:

$$\mathbf{z}_{\text{pair}} = \left[ |\mathbf{g}_v - \mathbf{g}_t| \;\Vert\; (\mathbf{g}_v \odot \mathbf{g}_t) \;\Vert\; S_\epsilon(V, T) \right]$$

Image-quality features include:
1. Mean absolute high-frequency residual
2. Laplacian variance
3. Luminance standard deviation
4. Image dimensions (metadata)

In the evidential path, the 3 quality metrics are standardized and concatenated with $\mathbf{z}_{\text{pair}}$ ($1,028$ total dimensions). In the conformal path, only the 3 specific features in Section 5.7 are used.

### 5.6 Alternative Evidential Pair-Level Classifier (Implemented, Not Active in Supplied Configuration)
Computes Dirichlet parameters via a 2-class head:

$$\mathbf{e} = \operatorname{softplus}(f_\theta(\mathbf{x})), \qquad \boldsymbol{\alpha} = \mathbf{e} + 1$$

$$p_k = \frac{\alpha_k}{\sum_{j=1}^K \alpha_j}, \qquad u = \frac{K}{\sum_{j=1}^K \alpha_j}, \quad (K=2)$$

Classes are `MATCH` and `CONTRADICTION`. Requires trained checkpoints and calibration files. Kept disabled in supplied configuration.

### 5.7 Active Split-Conformal Decision Mechanism
Uses `uca_conformal_v3.json` with class-conditional Gaussian density models on fit data.

The active 3-feature vector:
$$\mathbf{x} = \begin{bmatrix} S_\epsilon(V, T) \\ \text{discarded\_mass} \\ 1 - \text{global\_CLIP\_similarity} \end{bmatrix}$$

Nonconformity scores are aggregated at the recording-group level:
$$s_{\text{rec}} = \max_{i \in \text{Recording}} s(x_i, y_i)$$

At $\alpha = 0.10$, nominal simultaneous coverage is 90% across calibration groups ($\hat{q} = 0.6086335403$, 70 groups, 140 pairs).

#### Set-Valued Decision Rule:
- $\{\text{MATCH}\} \longrightarrow$ **`MATCH`**
- $\{\text{CONTRADICTION}\} \longrightarrow$ **`CONTRADICTION`**
- $\{\text{MATCH}, \text{CONTRADICTION}\} \longrightarrow$ **`INDETERMINATE`**
- $\emptyset \longrightarrow$ **`INDETERMINATE`**

### 5.8 Heatmap Interpretation
Displays $\mathbf{C} \in \mathbb{R}^{49 \times M}$ without synthetic interpolation. Missing matrices in historical records show an unavailable state. It represents feature dissimilarity, not bounding boxes.

---

## 6. Functional Requirements and Implementation Status

| ID | Requirement | Current Status & Acceptance Boundary |
| :--- | :--- | :--- |
| **FR-01** | Prototype investigator login | **Demo-only client-side UI gate.** No server-side authentication or session validation. |
| **FR-02** | Create, update, switch, clear, and delete cases | **Implemented (prototype).** API & dashboard support full lifecycle; clear retains case node; delete is permanent. |
| **FR-03** | Ingest image/video with genuine statement | **Implemented (prototype).** Rejects empty statements; validates mime & decoded bytes. Images: JPEG, PNG, BMP, WEBP, TIFF. Videos: MP4, AVI, MOV, MKV. |
| **FR-04** | Sample video & select representative frame | **Implemented (prototype).** Uniform 9-point sampling between 10%–90%; edge-energy/noise sharpness selection. Returns sampling report. |
| **FR-05** | Compute image-quality & patch-trust features | **Implemented (prototype).** Descriptive metrics & patch trust used in Partial OT; not epistemic uncertainty. |
| **FR-06** | Extract CLIP global & sequence features | **Implemented (prototype).** Frozen ViT-B/32 generates 49 patches, token sequence, and global embeddings. |
| **FR-07** | Assess MATCH / CONTRADICTION / INDETERMINATE | **Implemented conditionally.** Active via 3-feature split-conformal artifact. Evidential alternative exists. Fallback returns `INDETERMINATE`. |
| **FR-08** | Display patch/token dissimilarity visualization | **Implemented (exploratory).** Direct CLIP patch/token cost matrix; not supervised phrase grounding. |
| **FR-09** | Preserve & verify original evidence files | **Implemented (prototype).** Case-scoped SHA-256 content-addressed filesystem storage with read/write hash checks. |
| **FR-10** | Persist evidence/statement graph relationships | **Implemented (prototype).** Neo4j persistence with volatile in-memory fallback. Storage state explicitly reported. |
| **FR-11** | Record investigator cross-case links | **Implemented (prototype).** Stores investigator annotations; not automated entity matching. |
| **FR-12** | Suggest cross-case overlaps | **Heuristic only.** Shared-keyword candidate matching; requires human investigation. |
| **FR-13** | Generate LLM-grounded forensic briefing | **Not implemented.** `LLMAnalysisPanel` is a frontend UI component; no active LLM service is connected. |
| **FR-14** | Interactive case graph | **Implemented (prototype).** React Flow visualization of case nodes, evidence nodes, and assessment edges. |
| **FR-15** | Evidence audit table & detail view | **Implemented (prototype).** Shows digests, assessment state, quality scores, and transport diagnostics. |
| **FR-16** | Clear case evidence or delete case | **Implemented (prototype).** Independent endpoints; retains case metadata on clear. |
| **FR-17** | Health/readiness & storage status | **Implemented (prototype).** `/health`, `/ready`, and `/admin/memory` expose system state. |

---

## 7. External Interfaces

### 7.1 Frontend
The Next.js dashboard connects via `NEXT_PUBLIC_RESEARCH_API_URL`. Principal views:
- Case selector and metadata editor
- Multipart ingestion form (media file + statement text/file)
- React Flow interactive case graph
- CLIP patch/token dissimilarity heatmap
- Evidence ledger with SHA-256 integrity status
- Analysis and diagnostics inspection panel

#### Visual Status Encoding:
- **Green:** `MATCH`
- **Red:** `CONTRADICTION`
- **Grey / Neutral:** `INDETERMINATE`

### 7.2 Research-Engine HTTP API

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Basic service health response |
| `GET` | `/ready` | Readiness status, model load state, graph backend status |
| `GET` | `/cases` | List cases and evidence counts |
| `POST` | `/cases` | Create a new case |
| `PUT` | `/cases/{case_id}` | Update case metadata |
| `DELETE` | `/cases/{case_id}` | Permanently delete a case and its graph nodes |
| `GET` | `/graph/{case_id}` | Retrieve case graph and evidence list |
| `GET` | `/graph/all` | Retrieve all cases and cross-case relationships |
| `POST` | `/cross_case_relation` | Record a cross-case association |
| `GET` | `/cross_case_relations` | List cross-case relationships with persistence state |
| `GET` | `/cross_case_suggestions` | Query heuristic keyword-overlap candidate links |
| `POST` | `/clear/{case_id}` | Clear case evidence while keeping case metadata |
| `GET` | `/evidence/{case_id}` | Return evidence metadata list for a case |
| `GET` | `/evidence/{case_id}/manifest` | Return evidence file manifest with hashes |
| `GET` | `/evidence/{case_id}/{sha256}/file` | Download stored evidence file after hash verification |
| `POST` | `/evidence/{case_id}/verify` | Verify uploaded byte stream against stored digest |
| `POST` | `/analyze_evidence/` | Multipart ingestion and inference analysis |
| `GET` | `/admin/memory` | Query in-memory fallback state and snapshot configuration |
| `POST` | `/admin/memory/flush` | Flush current in-memory graph to disk snapshot |

### 7.3 Authentication and Identity Boundary
The dashboard login view is a client-side mock. The research-engine API does not validate bearer tokens or user sessions. The `X-Investigator-Id` header is recorded as assertion-only string metadata and must not be used for access control. The API should remain bound to `127.0.0.1`.

---

## 8. Non-Functional Requirements and Constraints

### 8.1 Performance
Performance depends on hardware (CPU vs. CUDA GPU), video duration, frame resolution, and token length. No formal latency SLAs are guaranteed. Typical inference on GPU: ~200–500 ms per image–statement pair.

### 8.2 Security and Privacy
- **Secret Management:** `.env` must never be checked into version control. Use `.env.example` as a template.
- **Network Isolation:** Service ports in `docker-compose.yml` are bound to `127.0.0.1`. Do not bind to `0.0.0.0` without an authenticating reverse proxy.
- **Input Validation:** Restricts uploads to 100 MiB evidence, 64 KiB / 20,000 char statements, and 300 s video clips. Decodes media with Pillow/OpenCV before passing to inference.
- **Evidence Integrity:** SHA-256 verifies byte integrity; it does not replace signed hardware write-once (WORM) storage or cryptographic timestamps.
- **Data Hygiene:** Exclude uploaded files, checkpoints, and datasets from git tracking.

### 8.3 Reliability and Persistence
- Neo4j writes execute in transactional sessions.
- Process-local memory fallback activates automatically if Neo4j is disconnected, marking responses as `persisted=false`.
- Filesystem storage operates independently of graph database connectivity.

### 8.4 Research Reproducibility
- Pinned CLIP model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Record data partition hashes and calibration artifacts with evaluation outputs.
- Maintain disjoint partitions by `source_recording_id`.
- Avoid interpreting `camera_id` as hardware-verified camera independence.

### 8.5 Human Oversight
All model outputs are decision-support signals. Independent human examination of physical evidence and contextual circumstances is mandatory.

---

## 9. Verification and Validation Plan

| Test Area | Verification Method | Acceptance Condition |
| :--- | :--- | :--- |
| **Sinkhorn Mathematics** | Unit tests with finite/non-square cost matrices | Solver converges within tolerance; no silent NaN/Inf |
| **Partial OT & Dustbin** | Unit tests for transport plan & discarded mass | All values finite; unmatched mass correctly accounted |
| **Evidential Head** | Unit tests for output shapes and Dirichlet bounds | $\alpha_k \ge 1$; probabilities sum to 1; uncertainty $\in [0, 1]$ |
| **Conformal Density & Scores** | Unit tests for Gaussian fitting & quantiles | Threshold applied accurately; schema mismatches rejected |
| **Recording-Group Calibration** | Unit tests with grouped calibration identifiers | Calibration score uses recording-group maximum; zero group leakage |
| **Decision Mechanism Config** | Runtime configuration assertion tests | Evidential and conformal cannot both be active; missing yields `INDETERMINATE` |
| **Feature Pipeline** | Mocked and smoke tests on CLIP embeddings | Sequence shapes match schema; no synthetic feature substitution |
| **Artifact Compatibility** | Checkpoint mismatch tests | Invalid model revision or schema version triggers rejection |
| **Abstention Logic** | Conformal set evaluation tests | Multi-label or empty prediction sets output `INDETERMINATE` |
| **Dataset Splits** | Group-disjoint audit scripts | Zero overlap of `source_recording_id` across fit/cal/test |
| **Evidence Storage** | File read/write/digest corruption tests | Correct byte retrieval; corrupt files raise integrity exception |
| **Graph Memory Fallback** | Neo4j offline simulation tests | Writes route to memory; report `persisted=false` |
| **Graph Transactions** | Live Neo4j integration suite | ACID transaction semantics verified across case lifecycle |
| **Frontend Status Rendering** | Vitest / component tests | Distinct visual treatments for MATCH, CONTRADICTION, INDETERMINATE |
| **Frontend Async Isolation** | Delayed response mocking | Previous case responses do not overwrite active case UI |
| **Production Build** | `npm run build` | Next.js production build completes without syntax or type errors |
| **Empirical Evaluation** | Held-out evaluation scripts | Metrics, coverage, and bootstrap intervals match report manifests |

### 9.1 Validation Evidence Recorded for Supplied Repository Snapshot
As documented in `VALIDATION_STATUS.md`:
- **Research Engine Test Suite:** 103 passed, 1 skipped (live Neo4j integration test skipped due to unset `NEO4J_TEST_URI`).
- **Python Syntax & Bytecode Compilation:** 40 files validated with zero syntax errors (`python -m compileall -q research-engine` passed).
- **Synthetic Adapter Smoke Run:** Confirmed zero partition leakage across fit, calibration, and test splits.
- **Docker Compose Ports:** Verified bound to `127.0.0.1`.

*Items marked unverified in the archive environment:*
- Full-scale CLIP inference on original UCF-Crime raw assets (due to dataset asset size).
- Production network security, penetration testing, and access control infrastructure.

### 9.2 Saved UCA group-v3 Conformal Evaluation (Artifact-Derived)

#### Dataset Partitions
| Partition | Pairs | MATCH | CONTRADICTION | Source Recordings |
| :--- | :--- | :--- | :--- | :--- |
| **Fit** | 3,136 | 1,568 | 1,568 | 1,568 |
| **Calibration** | 140 | 70 | 70 | 70 |
| **Test** | 140 | 70 | 70 | 70 |

#### Conformal Evaluation ($\alpha = 0.10$, Schema-3)
| Metric | Saved Result |
| :--- | :--- |
| **Conformal Threshold ($\hat{q}$)** | `0.6086335` |
| **Pair-Level Empirical Coverage** | 93.57% |
| **Singleton Prediction Rate** | 30.00% |
| **Selective Accuracy** | 78.57% |
| **Abstention Rate** | 70.00% |
| **Average Prediction-Set Size** | 1.70 |
| **Simultaneous Recording-Group Coverage** | 61 / 70 groups (87.14%) |

### 9.3 Saved Full OT Versus Partial OT Comparison (Two-Feature Ablation)

| Metric | Full OT | Quality-Weighted Partial OT |
| :--- | :--- | :--- |
| **Pair-Level Empirical Coverage** | 94.29% | 93.57% |
| **Singleton Prediction Rate** | 32.86% | 27.14% |
| **Selective Accuracy** | 82.61% | 76.32% |
| **Abstention Rate** | 67.14% | 72.86% |
| **Recording-Group Simultaneous Coverage** | 88.57% (62/70) | 87.14% (61/70) |

*Bootstrap analysis (2,000 replicates across 70 recording groups) showed a Full OT selective-accuracy advantage of 6.29 percentage points (95% CI: [2.02, 11.67]) on this proxy split.*

---

## 10. Limitations and Future Research

### 10.1 Retired Experimental Artifacts
- The random-projection code in `multimodal_fusion.py` and the Phase 1 sandbox are retired.
- Historical sandbox outputs (e.g. synthetic `84.5% / PROVED`) are deprecated and must not be cited as measured results.
- `test_clip_calib.py` serves solely as a deprecation notice.

### 10.2 UCA Proxy Dataset and Interpretation Limits
- Constructed from UCF-Crime video frames and descriptions.
- `CONTRADICTION` represents cross-recording, cross-category mismatch, not human-adjudicated perjury or factual falsification.
- `camera_id` is derived from recording IDs and does not represent independent physical hardware.
- The held-out test split has been utilized in development ablations and must be treated as retrospective.

### 10.3 Capabilities Not Implemented or Not Validated
1. **Real Forensic Dataset:** Lack of CCTV datasets containing authentic witness statements and forensic adjudications.
2. **Camera-Disjoint Evaluation:** Lack of multi-camera disjoint benchmarks with physical environment variation.
3. **Fine-Grained Grounding:** No bounding-box detection, color verification, or supervised phrase grounding.
4. **Temporal Modeling:** No whole-video sequential event or action recognition.
5. **Entity Resolution:** No cross-case biometric, vehicle, or weapon identification.
6. **LLM Integration:** No active LLM-based factual cross-examination or narrative generation.
7. **Queued Persistence:** No transactional write-ahead replay queue during Neo4j downtime.
8. **Security Hardening:** No server-side authentication, RBAC, WORM storage, or cryptographic chain of custody.
9. **Forensic Admissibility:** No formal validation for judicial or prosecutorial proceedings.

---

*End of Implementation-Aligned Software Requirements Specification — CrimeVision AI.*