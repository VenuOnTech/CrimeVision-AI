SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
CrimeVision AI — Multimodal Evidence Comparison and Investigation Decision-Support Prototype
Document version: 1.2.0 (current-repository implementation alignment)
Project type: B.Tech capstone research prototype
Development team: Shyamal, Moukthika, Venu, Niteesha
Status: Prototype implementation with saved UCA-derived grouped-conformal research results; independent real-world forensic validation pending  
Scope and safety statement: CrimeVision AI is a research and investigation decision-support prototype. It does not establish what physically occurred, verify a suspect's identity, establish guilt, or create court-admissible findings. Automated outputs are not forensic proof. Original evidence, acquisition provenance, independent examination, and qualified human judgement remain necessary.

Document purpose and conventions
This document describes the current repository implementation and distinguishes implemented behaviour from research scaffolding and future work. A feature described as implemented means corresponding source code exists; it does not imply production hardening, independent validation, or a measured performance guarantee.

Status labels used below:
- Implemented (prototype): Source implementation exists in the current repository.
- Scaffolded: Training, calibration, or integration code exists, but it still needs real data or environment-level validation.
- Planned / not implemented: The current repository does not provide the described capability.
- Not validated: The capability has not been demonstrated on a suitable held-out dataset or in a production environment.
Vectors are bold lowercase, matrices are bold uppercase, and probability distributions are represented by Greek or calligraphic symbols.
Table of contents
1. Introduction
2. Overall description and architecture
3. Technology stack and operating environment
4. Data model and persistence
5. Mathematical and algorithmic formulation
6. Functional requirements and implementation status
7. External interfaces
8. Non-functional requirements and constraints
9. Verification and validation plan
10. Limitations and future research
1. Introduction
1.1 Purpose
CrimeVision AI is a research and investigation decision-support prototype for comparing a still image, or one representative frame sampled from a video, with a supplied natural-language witness statement. The current configured analysis path uses a frozen CLIP vision-language backbone, quality-aware Partial Optimal Transport (OT) diagnostics, and a split-conformal prediction wrapper. The application also provides case/evidence management, local evidence-file preservation, a Neo4j case/evidence graph, and an investigator dashboard.

A supervised evidential pair-level neural head is implemented as an alternative research path. It is not the active decision mechanism in the supplied project configuration. This distinction is important: the current conformal prediction result must not be described as an output from the evidential head.

The current implementation is intended for research prototyping, software integration, and evaluation-method development. It is not an operational forensic system.
1.2 Product scope
The current research-engine API supports the following prototype workflow:
1. An investigator creates or selects a case in the Next.js dashboard.
2. The investigator uploads one supported image or video and supplies a genuine statement either as text or as a UTF-8 .txt file. The API rejects a request that does not contain a statement; it does not insert a synthetic default statement.
3. The API validates the case ID, filename, upload size, declared media type, and decoded media. Supported image extensions are JPG/JPEG, PNG, BMP, WEBP, TIF, and TIFF. Supported video extensions are MP4, AVI, MOV, and MKV. The default evidence upload limit is 100 MiB; statement limits are configurable (default 64 KiB and 20,000 characters).
4. For a video, OpenCV samples a configurable number of approximately uniform frame positions (default N = 9) spanning 10% to 90% of the clip. It selects the sharpest successfully decoded sample using an edge-energy/noise heuristic and returns a sampling report containing the attempted positions, frame indices, timestamps, sharpness values, and selected frame. The default maximum video duration is 300 seconds. This is representative-frame selection, not temporal event understanding.
5. Before inference, the original uploaded bytes are stored by the evidence store under a case-scoped, SHA-256-derived path. The store verifies the digest on write and on read. Analysis failure after this point does not imply that the original file was not preserved.
6. Hugging Face CLIP produces image-patch features, text-token features, and global image/text features. The current ViT-B/32 configuration yields a 7 × 7 grid (49) of image patches before alignment.
7. A log-domain Sinkhorn implementation calculates a debiased Sinkhorn divergence and partial-transport diagnostics. The configured inference path uses quality-aware Partial OT with a dustbin/unmatched-mass mechanism. These values are diagnostics and are not independently converted into a contradiction verdict.
8. In the supplied configuration, a split-conformal artifact (research-engine/models/uca_conformal_v3.json) is the active decision mechanism. It uses three features—Sinkhorn divergence, discarded mass, and one minus global CLIP similarity—and returns a prediction set. A singleton set produces MATCH or CONTRADICTION; a two-label or empty set produces INDETERMINATE. If neither a conformal artifact nor a compatible trained evidential head with calibration is configured, the engine returns INDETERMINATE while still providing diagnostics.
9. The evidential head plus its matching calibration artifact is an alternative decision mechanism. The inference engine rejects a configuration that enables both the evidential and conformal paths simultaneously.
10. Case/evidence metadata and analysis artefacts are written to Neo4j when available. When Neo4j is unavailable, graph records are held in a process-local memory store and are explicitly reported as non-durable. An optional memory snapshot can be restored/flushed, but it is not equivalent to transactional Neo4j persistence.
11. The Next.js dashboard displays the assessment status, diagnostics, evidence table, patch/token heatmap, case graph, and storage/persistence status.
The current implementation does not provide a validated object detector, object/attribute verifier, supervised region-word grounding model, automatic suspect identification, temporal video-event understanding, or an end-to-end forensic chain-of-custody service. A SHA-256 check can detect a changed file; it does not establish who acquired the evidence or make storage immutable.
1.3 Intended audience
- Project evaluators: assess the relationship between stated research aims, source implementation, and saved experiments.
- AI/ML developers: maintain CLIP feature extraction, OT/Partial OT, the conformal pipeline, and the alternative evidential-head implementation.
- Dataset and research-evaluation developers: maintain UCA proxy-pair construction, source-recording splits, provenance audits, ablations, bootstrap analysis, and robustness experiments.
- Backend/database developers: maintain the FastAPI research engine, evidence-file store, and Neo4j graph persistence.
- Frontend developers: maintain the Next.js case workspace, upload flow, visualizations, and presentation of uncertainty and persistence state.
- Investigative-domain reviewers: assess data suitability, label definitions, and the limits of any proposed use.
1.4 Definitions
Term	Meaning in this project
CLIP	Contrastive Language–Image Pretraining model used as a frozen visual/text feature backbone
OT	Optimal Transport
Partial OT	OT formulation that permits some mass to remain unmatched through a dustbin/discarded-mass mechanism
Sinkhorn solver	Entropy-regularized OT solver implemented in log space
Sinkhorn divergence	Debiased combination of cross-modal and self-transport objectives; an alignment diagnostic in this project
Dustbin / discarded mass	Mechanism and diagnostic for visual or textual mass that is not matched by the partial-transport plan
Split-conformal prediction	A calibration procedure that forms a prediction set using a threshold computed from held-out calibration nonconformity scores
Prediction set	Set of possible labels retained by the conformal predictor for an input
Conformal calibration unit	The unit on which calibration scores are aggregated; the supplied v3 artifact uses the maximum true-label score per source_recording_id
Evidential head	An alternative supervised two-class neural head that outputs Dirichlet concentration parameters; not active in the supplied configuration
Model uncertainty	Dirichlet-based uncertainty estimate from the optional evidential head; not a guarantee of correctness
Proxy label	A constructed research label used as a stand-in for a more difficult human-verified concept; the UCA CONTRADICTION label is a cross-recording, cross-category description mismatch
Image-quality metrics	Descriptive measurements such as high-frequency residual, Laplacian variance, and luminance standard deviation; not epistemic uncertainty
OOD	Out-of-distribution relative to a training distribution or configured feature range
Abstention	Returning INDETERMINATE rather than issuing a binary assessment
Evidence hash	SHA-256 digest of the uploaded file bytes; an integrity identifier, not full chain-of-custody proof
Volatile fallback	Process-local graph memory used when Neo4j is unavailable; it is reported as non-durable even if an optional snapshot is configured


2. Overall description and architecture
2.1 System context
The supplied repository contains one working application API: the FastAPI research engine in research-engine/main.py, normally exposed on port 8001. The Next.js dashboard in frontend-web/ calls that API. Neo4j is the graph database. The current repository snapshot does not contain a separate backend/ service, a MongoDB service, or a second active API on port 8000.

The root docker-compose.yml defines three runtime services: neo4j, research-engine, and frontend-web. The host-published application and database ports are bound to loopback (127.0.0.1) in the supplied Compose file. This is a local-development/research deployment boundary, not a production network-security design.
2.2 Logical architecture
Investigator browser
  Next.js 13 / React 18 dashboard
       |
       | NEXT_PUBLIC_RESEARCH_API_URL
       | multipart evidence + statement upload; JSON case/graph requests
       v
Research Engine API (FastAPI / Uvicorn, port 8001)
       |
       +-- Case ID, filename, size, content-type and decoded-media validation
       +-- Original-byte SHA-256 calculation and case-scoped EvidenceStore
       +-- OpenCV video sampling / Pillow image decoding
       +-- Frozen Hugging Face CLIP ViT-B/32
       |      +-- image patch features (7 x 7 = 49 for the configured model)
       |      +-- text token features
       |      +-- global image/text features
       +-- Quality-aware Partial OT / log-domain Sinkhorn diagnostics
       |      +-- patch trust weights
       |      +-- dustbin / discarded mass
       |      +-- divergence, transport plan and patch-token cost matrix
       +-- Decision mechanism (mutually exclusive configuration)
       |      +-- Supplied active path: split-conformal prediction
       |      |      +-- class-conditional Gaussian feature densities
       |      |      +-- recording-group calibration threshold
       |      |      `-- MATCH / CONTRADICTION / INDETERMINATE
       |      +-- Alternative path: trained evidential head + calibration
       |      `-- Neither configured: INDETERMINATE-only diagnostics
       +-- Neo4j case/evidence/statement graph
       |      `-- process-local memory fallback; optional snapshot
       `-- Case-scoped filesystem evidence storage

Docker Compose services:
  Neo4j + Research Engine API + Next.js Frontend
The research engine is both the HTTP API and the inference service in this architecture. OT diagnostics do not independently create an assessment status. The actual status comes from the selected decision mechanism or is INDETERMINATE when no decision mechanism is available.
2.3 Deployment boundaries
- The dashboard uses NEXT_PUBLIC_RESEARCH_API_URL to reach the research-engine API. The supplied example points to http://127.0.0.1:8001.
- The research engine uses NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD for graph storage.
- CLIP configuration is controlled by CLIP_MODEL_NAME, optional CLIP_MODEL_REVISION, and DEFAULT_DEVICE. The supplied configuration identifies openai/clip-vit-base-patch32 and a pinned model revision.
- The supplied .env.example selects the conformal route using CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json; EVIDENTIAL_HEAD_PATH and EVIDENTIAL_CALIBRATION_PATH are empty in that configuration. The supplied Docker Compose file mounts research-engine/models read-only at /app/models.
- The inference engine accepts either (a) both EVIDENTIAL_HEAD_PATH and EVIDENTIAL_CALIBRATION_PATH, or (b) CONFORMAL_ARTIFACT_PATH. These decision mechanisms are mutually exclusive. If neither is configured, analysis returns diagnostics with INDETERMINATE status.
- TRUST_MODE, OT_TRUST_TAU, and OT_TRUST_K control patch trust weighting. The supplied configuration uses TRUST_MODE=heterogeneity and quality-aware Partial OT.
- EVIDENCE_STORAGE_ROOT selects the evidence-file directory. Docker Compose mounts a named volume for evidence files. The Compose file also defines named volumes for Neo4j data and the memory snapshot.
- Docker Compose starts all three current services. For local development without Docker, Neo4j may be started separately and the research API and frontend may be launched in separate terminals.
- The research API and frontend are intended to be accessed locally. Do not publish the unauthenticated research API to a public interface.
2.4 User classes
The prototype UI assumes an investigator/reviewer who can create and manage cases, upload evidence and statements, examine diagnostic visualizations, and inspect graph records. The login page is a client-side demo gate; it does not authenticate API requests. The research-engine API does not implement case-level authorization or a multi-user role system.
2.5 Dependencies and constraints
- A supported Python environment, PyTorch, Transformers, OpenCV, Pillow, FastAPI, Uvicorn, and the Neo4j Python driver are required for the research engine.
- Node.js, npm, and the frontend dependencies are required for the Next.js dashboard.
- Docker Engine/Desktop with Docker Compose is used for the supplied three-service deployment.
- The first CLIP load may require downloading model weights from Hugging Face unless the pinned model revision is already cached.
- CPU inference is supported, but speed and memory requirements depend on the host and input size. A CUDA-capable GPU is optional; DEFAULT_DEVICE=auto selects CUDA when available and CPU otherwise.
- Neo4j availability is not guaranteed. When unavailable, graph writes fall back to memory and are not reported as durable Neo4j writes.
- The UCA research CSVs refer to source image assets that are not all included in the supplied ZIP. Reproducing the complete real-image experiment requires the separately obtained source dataset assets.
3. Technology stack and operating environment
Layer	Current repository technologies	Implementation notes
Research engine and API	Python, FastAPI, Uvicorn, Pydantic	One service provides the HTTP API and invokes the inference pipeline
Vision-language backbone	openai/clip-vit-base-patch32 via Hugging Face Transformers	Frozen CLIP backbone; the supplied configuration pins a model revision
Numerical computation	PyTorch, NumPy, SciPy	Feature operations, density modelling, numerical routines and evaluation
Optimal Transport	Custom log-domain Sinkhorn implementation in PyTorch	Full OT comparison path and quality-aware Partial OT with dustbin/discarded mass
Conformal prediction	Project implementation in research-engine/src/conformal.py	Class-conditional Gaussian feature densities and split-conformal prediction sets
Alternative evidential classifier	PyTorch implementation in research-engine/src/evidential_fusion.py	Trainable two-class Dirichlet head; implemented as an alternative, not selected by the supplied runtime configuration
Image/video processing	OpenCV, Pillow	Image validation/decoding, image-quality features, and N-point video sampling
Evidence-file storage	Filesystem EvidenceStore	Case-scoped content-addressed files, SHA-256 verification, evidence manifest and retrieval
Graph database	Neo4j 5.x with Neo4j Python driver	Case, visual-evidence, witness-statement, analysis and cross-case relationships
Frontend	Next.js 13, React 18, Tailwind CSS, React Flow, Axios, Recharts, lucide-react	Case dashboard, evidence upload/table, graph and diagnostics
Testing	pytest; Vitest configured for frontend	Research-engine test suite and frontend test configuration
Containerization	Docker Compose	Neo4j, research-engine API, and frontend service definitions


The current repository does not include an active MongoDB/Motor backend service. The case/evidence graph flow used by the dashboard is handled by the research-engine API and Neo4j connector. Versions in requirements files are not, by themselves, a fully reproducible environment lock; pin and test Python, PyTorch, CUDA, Transformers, Neo4j, Node, npm, and container image versions before deployment.
4. Data model and persistence
4.1 Case records
A Case node stores a case identifier, name, description, category, and creation timestamp. The case_id is the root identifier for case-scoped data. The dashboard supports creating, updating, switching between, clearing evidence from, and deleting cases through the research-engine API.
4.2 Evidence and statement records
The research engine stores the original uploaded bytes separately from Neo4j graph metadata. The filesystem evidence store is case-scoped and content-addressed: the stored file path is derived from the case identifier, SHA-256 digest, and validated extension. The store computes the SHA-256 digest before writing, verifies that it matches the supplied digest, writes through a temporary file followed by an atomic replace, and verifies the digest again when reading a stored file.

The graph connector derives stable case-scoped identifiers for visual evidence and witness statements. The logical graph pattern is:
(:Case)-[:CONTAINS]->(:VisualEvidence)
(:Case)-[:CONTAINS]->(:WitnessStatement)
(:WitnessStatement)-[:CORRELATES_TO]->(:VisualEvidence)
The CORRELATES_TO relationship can store:
- sinkhorn_cost — the debiased Sinkhorn divergence diagnostic;
- assessment_status — MATCH, CONTRADICTION, or INDETERMINATE;
- flagged_contradiction — nullable Boolean derived from the explicit assessment status;
- optional model_epistemic_uncertainty and class probabilities when the alternative evidential path is used;
- JSON-serialized visual-quality metrics and OT diagnostics;
- the patch/token heatmap matrix, token labels, and patch labels when available;
- model name/revision, feature-schema/calibration metadata, evidence hash, file size/path, timestamps, investigator advisory, audit report, and uploader-header value.
The X-Investigator-Id header is assertion-only metadata; it is not verified identity. The graph connector does not create a classifier decision from a Sinkhorn threshold. The frontend and API must use the explicit assessment_status returned by the configured decision mechanism.
4.3 Case-scoped identifiers and legacy records
Case-scoped IDs prevent new evidence or statement nodes from being merged solely because two cases use the same filename or statement text. Unique constraints for Case.id, VisualEvidence.id, and WitnessStatement.id are attempted during connector startup.

Legacy graph nodes created by earlier filename-only/text-only logic may not have a case_id property. The clearing logic removes case ownership relationships and deletes nodes explicitly owned by the selected case. It deliberately does not blindly delete legacy shared nodes without ownership metadata, because doing so could affect another case. Such legacy orphans require a separate reviewed migration/cleanup.
4.4 Persistence and fallback semantics
- When Neo4j is connected and a write transaction confirms the operation, the API reports the graph record as persisted to neo4j.
- When Neo4j is unavailable, the connector stores graph records in a process-local memory_store and reports persisted=false, storage_backend="memory", and non-durable storage status.
- The project can be configured with a memory snapshot path. When configured, the connector can restore a snapshot at startup and flush it on shutdown or through the administrative flush endpoint. A snapshot is a filesystem copy of fallback state; it does not provide Neo4j transaction semantics, immutable audit history, or a production backup/recovery guarantee.
- Evidence files are stored separately from graph records. A successful evidence-file write and a successful graph write are separate outcomes.
- An API analysis response with status="success" does not by itself prove that the graph write was durable. Clients must inspect graph_status and storage_status.
- A case clear retains the case metadata and case node. Permanent case deletion is a separate operation.
4.5 Cross-case relations and heuristic candidates
Manual cross-case relations are stored as CROSS_CASE_CORRELATION relationships with relation type, investigator notes, and optional investigator-assigned strength. An investigator-assigned value is not a calibrated probability.

The suggestion endpoint compares text for shared keywords. A returned keyword overlap is only a candidate for human review. It does not establish that cases share a suspect, vehicle, weapon, location, or modus operandi. No trained cross-case identity-resolution model is part of the current implementation.
5. Mathematical and algorithmic formulation
5.1 CLIP feature extraction
The research engine uses the CLIP ViT-B/32 model as a fixed feature backbone. It extracts:
- a sequence of image patch features (excluding the CLS position from the patch list); for the configured 224-pixel ViT-B/32 input, the visual patch grid is 7 × 7, or 49 patches;
- text token features after excluding special tokens;
- a global image feature and a global text feature.
The visual patch and text token features are projected into the shared CLIP feature space and L2-normalized before pairwise cosine-distance calculation. The supplied configuration freezes CLIP parameters for inference. This is cross-modal feature comparison, not an object detector or a supervised object-localization model.
5.2 Cross-modal cost matrix
Let normalized visual patch features be \(V=\{v_i\}_{i=1}^N\) and normalized text token features be \(T=\{t_j\}_{j=1}^M\). The implemented pairwise cost is:
\[
C_{ij}=\max\left(0,1-v_i^\top t_j\right).
\]
The matrix \(C\in\mathbb{R}^{N\times M}\) is exposed as an exploratory patch/token dissimilarity visualization. It is not supervised region-word grounding, object localization, or a direct description of which physical object is present in a region.
5.3 Entropic and debiased Sinkhorn transport
For source and target marginals \(a\in\Delta^N\) and \(b\in\Delta^M\), the standard entropic OT objective used by the solver is:
\[
\operatorname{OT}_{\epsilon}(a,b;C)=
\min_{P\geq 0,\;P\mathbf{1}=a,\;P^\top\mathbf{1}=b}
\left[\langle P,C\rangle+\epsilon\,\mathrm{KL}(P\,\|\,ab^\top)\right].
\]
The solver operates in the log domain to reduce numerical instability, checks marginal errors, and raises a convergence error when configured tolerances are not reached.

The reported debiased Sinkhorn divergence is:
\[
S_\epsilon(V,T)=\operatorname{OT}_\epsilon(V,T)
-\frac{1}{2}\operatorname{OT}_\epsilon(V,V)
-\frac{1}{2}\operatorname{OT}_\epsilon(T,T).
\]
Small negative numerical values within a configured tolerance are clamped to zero; materially negative values are treated as an error. The divergence is an alignment diagnostic. There is no universal 1.15 contradiction threshold in the current research inference design.
5.4 Quality-aware Partial OT and Full OT comparison
The inference engine supports both a Full OT comparison path and a Partial OT path. Partial OT augments the transport problem with a dustbin/unmatched-mass mechanism so that some feature mass need not be forced into a patch-to-token match. The reported discarded_mass describes the unmatched portion under the configured partial-transport calculation.

In the supplied configuration, TRUST_MODE=heterogeneity and force_full_ot=false. Image-quality and patch-sharpness information is used to derive patch-level trust weights for the Partial OT path. The supplied artifact records a Sinkhorn regularization value of 0.1 and a dustbin cost of 10.0. These are implementation settings, not universal constants.

The repository includes Full OT versus Partial OT experiments. The existence of a Partial OT implementation does not establish that it outperforms Full OT; the saved grouped-v3 comparison reports higher selective accuracy for the tested Full OT two-feature variant. See Section 9 for the stored experiment summary and its scope.
5.5 Pair features and image-quality features
Let \(g_v\) and \(g_t\) be normalized global CLIP image and text features. The optional evidential pair head uses:
\[
z_{pair}=\left[|g_v-g_t|\;\Vert\;(g_v\odot g_t)\;\Vert\;S_\epsilon(V,T)\right].
\]
For the current 512-dimensional CLIP projection, this produces 1,025 pair features: 512 absolute-difference features, 512 elementwise-product features, and one Sinkhorn-divergence scalar. This pair vector is not the active conformal feature vector.

The image-quality module reports:
- mean absolute high-frequency residual;
- Laplacian variance;
- luminance standard deviation;
- image width and height as metadata.
These values describe image characteristics. They are not calibrated probabilities and are not, by themselves, epistemic uncertainty. In the optional evidential training path, the three scalar quality features are standardized using statistics fitted on the training split. In the active conformal path, the decision features are the three explicitly listed in Section 5.7; the evidential quality-feature normalization is not part of that conformal feature vector.
5.6 Alternative evidential pair-level classifier (implemented, not active in supplied configuration)
The pair classifier receives the 1,025 pair features concatenated with the three standardized quality features, for a total of 1,028 inputs. The repository contains the evidential-head implementation, training code, and calibration-related code.

The two-class evidential head computes non-negative evidence, Dirichlet concentration, probabilities, and a Dirichlet-based uncertainty estimate:
\[
e=\operatorname{softplus}(f_\theta(x)),\qquad
\alpha=e+1,\qquad
p_k=\frac{\alpha_k}{\sum_j\alpha_j},\qquad
u=\frac{K}{\sum_j\alpha_j},\quad K=2.
\]
The two classes are MATCH and CONTRADICTION. The head must be trained on appropriately labelled image/statement pairs; a randomly initialized head is never used for inference decisions. At runtime it requires both a compatible checkpoint and its matching calibration artifact. The supplied project configuration leaves EVIDENTIAL_HEAD_PATH and EVIDENTIAL_CALIBRATION_PATH empty, so this alternative is not the configured decision path.

When this alternative is deliberately enabled, its calibrated contradiction-probability threshold and evidential uncertainty/OOD checks govern its output. These classifier-specific checks must not be described as part of the active conformal predictor.
5.7 Active split-conformal decision mechanism
The supplied configuration selects the split-conformal predictor in research-engine/models/uca_conformal_v3.json. It does not train a new neural pair classifier. It fits class-conditional Gaussian density parameters on the fit partition, calculates true-label nonconformity scores on a separate calibration partition, and obtains a conformal threshold from the calibration scores.

The active three-feature vector is:
\[
x=\left[
S_\epsilon(V,T),\;
\text{discarded\_mass},\;
1-\text{global\_CLIP\_similarity}
\right].
\]
The fitted Gaussian densities are used to calculate class-wise nonconformity scores. For the supplied v3 artifact, calibration scores are aggregated by taking the maximum true-label nonconformity score within each source_recording_id. The resulting calibration unit is source_recording_id_max, with 70 calibration recording groups (140 paired examples). The supplied artifact has \(\alpha=0.10\), nominal target coverage 0.90, and \(\hat q=0.6086335403\). Its fit partition contains 3,136 examples (1,568 per class).

For a new feature vector, the predictor returns a set of candidate classes:
- {MATCH} → MATCH;
- {CONTRADICTION} → CONTRADICTION;
- {MATCH, CONTRADICTION} → INDETERMINATE;
- an empty set → INDETERMINATE.
The conformal coverage statement is marginal at the declared calibration unit. For the supplied artifact, the intended guarantee is simultaneous inclusion of all labelled examples within a future source-recording group under exchangeability of recording groups constructed by the same protocol. It is not class-conditional, does not guarantee correctness for an individual accepted prediction, does not guarantee selective accuracy or abstention rate, and does not guarantee robustness under distribution shift. It is not a legal or forensic assurance.

If the conformal artifact is not configured and no compatible evidential checkpoint/calibration pair is configured, the engine returns INDETERMINATE while retaining the feature diagnostics. The conformal and evidential decision mechanisms are mutually exclusive in the current inference-engine configuration.
5.8 Heatmap interpretation
The dashboard heatmap uses the CLIP patch/token cost matrix. The displayed matrix is exploratory and does not support claims of supervised phrase grounding, object-level detection, or attribute localization. Historical graph records do not reconstruct missing matrices with synthetic values; when a stored record lacks the original matrix, the frontend displays an empty/unavailable state.
6. Functional requirements and implementation status
ID	Requirement	Current status and acceptance boundary
FR-01	Prototype investigator login	Demo-only client-side UI gate. It does not authenticate requests to the research-engine API. There is no active server-side identity provider or verified user session.
FR-02	Create, update, switch, clear, and delete cases	Implemented (prototype) in the research-engine API and dashboard. Clearing evidence retains the case; permanent deletion is separate. Inspect returned storage status to determine whether graph changes were written to Neo4j.
FR-03	Ingest image/video with a genuine witness statement	Implemented (prototype). Images: JPG/JPEG, PNG, BMP, WEBP, TIF/TIFF. Videos: MP4, AVI, MOV, MKV. A statement must be provided as text or a UTF-8 .txt file. Default byte/character/duration limits are configurable. PDF, audio transcription, GPS ingestion, and general document analysis are not part of this analysis route.
FR-04	Sample video and select a representative frame	Implemented (prototype). N (default 9) approximately uniform frame positions spanning [10%, 90%] are sampled; the sharpest decoded frame is selected using an edge-energy/noise heuristic, and a per-video sampling report is returned. This is not event-level, adaptive, or multi-frame temporal reasoning.
FR-05	Compute image-quality and patch-trust features	Implemented (prototype). Descriptive image-quality features and patch-level trust weighting are used by the Partial OT path; they are not labelled as model epistemic uncertainty.
FR-06	Extract CLIP global and sequence features	Implemented (prototype). The configured frozen CLIP backbone produces patch/token and global features. Model name/revision are recorded. Global similarity is not used as a universal fixed-cutoff final classifier.
FR-07	Assess MATCH / CONTRADICTION / INDETERMINATE	Implemented conditionally. The supplied configuration uses the three-feature split-conformal artifact. A singleton prediction set produces a binary status; a two-label or empty set produces INDETERMINATE. An alternative trained evidential head plus compatible calibration can be enabled instead, but not simultaneously. If neither route is configured, the engine is diagnostics-only and returns INDETERMINATE.
FR-08	Display patch/token dissimilarity visualization	Implemented as exploratory visualization. It is the CLIP patch/token cost matrix, not a three-zone supervised object/phrase-grounding heatmap.
FR-09	Preserve and verify original evidence files	Implemented (prototype). Original upload bytes are stored in case-scoped filesystem storage using SHA-256 content addressing. Hashes are checked on write and read; the case evidence manifest and file-retrieval/verification routes are available. This is not immutable or signed chain-of-custody infrastructure.
FR-10	Persist evidence/statement graph relationships	Implemented (prototype). Case-scoped identifiers and explicit assessment metadata are stored in Neo4j when available. Memory fallback is reported as non-durable; file storage and graph persistence are separate outcomes.
FR-11	Record investigator cross-case links	Implemented (prototype). Relationship records carry investigator-provided metadata; they are not automatic identity verification.
FR-12	Suggest cross-case overlaps	Heuristic only. Shared-keyword candidates require investigator review. There is no trained entity-resolution or calibrated confidence model.
FR-13	Generate an LLM-grounded forensic briefing	Not implemented in the current analysis path. LLMAnalysisPanel is a frontend presentation component for returned analysis/audit information; its name does not imply an active LLM call.
FR-14	Interactive case graph	Implemented (prototype) using React Flow. It reflects stored case/evidence relationships and explicit assessment statuses.
FR-15	Evidence audit table and evidence detail	Implemented (prototype). Evidence metadata, hashes, status and available diagnostics are presented. Historical records without the original heatmap matrix cannot display a reconstructed matrix.
FR-16	Clear case evidence or delete case	Implemented (prototype). These operations are separate; legacy orphan nodes without case-ownership metadata may require reviewed migration.
FR-17	Health/readiness and storage status	Implemented (prototype). Health/readiness and memory-storage administration endpoints expose service or persistence state; a successful health response does not prove that full ML analysis or durable graph storage is available.


7. External interfaces
7.1 Frontend
The Next.js dashboard uses NEXT_PUBLIC_RESEARCH_API_URL to connect to the research-engine API. Its principal panels include case tabs, an ingestion form, case/evidence graph, CLIP patch/token heatmap, evidence table, evidence detail, and analysis summary.

The frontend must render the three assessment states distinctly:
- Green: MATCH;
- Red: CONTRADICTION;
- Grey/neutral: INDETERMINATE.
No panel may present INDETERMINATE as a match or derive assessment status from Sinkhorn cost. The heatmap is a feature-dissimilarity visualization, not an object detector or supervised explanation.
7.2 Research-engine HTTP API
The following routes are present in research-engine/main.py:
Method	Endpoint	Purpose
GET	/health	Basic research-engine health response
GET	/ready	Readiness, inference-engine-loaded state, graph-storage status and evidence root
GET	/cases	Return case metadata and evidence counts
POST	/cases	Create a case
PUT	/cases/{case_id}	Update existing case metadata
DELETE	/cases/{case_id}	Permanently delete a case
GET	/graph/{case_id}	Return one case graph and evidence list
GET	/graph/all	Return all cases and cross-case relationships
POST	/cross_case_relation	Record a cross-case relationship
GET	/cross_case_relations	Return persisted or volatile relationship records with storage status
GET	/cross_case_suggestions	Return heuristic keyword-overlap candidates
POST	/clear/{case_id}	Clear case evidence while retaining the case
GET	/evidence/{case_id}	Return case evidence metadata
GET	/evidence/{case_id}/manifest	Return the case evidence-file manifest
GET	/evidence/{case_id}/{sha256}/file	Retrieve a stored evidence file after integrity verification
POST	/evidence/{case_id}/verify	Verify supplied file bytes against the stored evidence digest
POST	/analyze_evidence/	Analyze one uploaded image/video and a genuine statement
GET	/admin/memory	Report memory-fallback/snapshot status
POST	/admin/memory/flush	Explicitly flush the optional memory snapshot when configured and applicable


The analysis request accepts case_id, evidence_file, and either statement_text or a UTF-8 .txt statement_file (not both). An optional acquisition timestamp may be supplied. The API's status="success" indicates that the analysis request completed. It does not mean that the classifier returned a binary decision or that graph persistence was durable. Clients must inspect assessment_status, graph_status, and storage_status independently.
7.3 Authentication and identity boundary
The current login page is a demo-only client-side gate. The research-engine API has no authentication or per-case authorization. The optional X-Investigator-Id request header is accepted as an assertion-only metadata value and is not authenticated. It must not be treated as proof of operator identity or used as an authorization control. The research API must remain loopback-only until real authentication, authorization, and secure deployment controls are implemented.
8. Non-functional requirements and constraints
8.1 Performance
No latency target is currently established by a reproducible benchmark report. Performance depends on CPU/GPU, image size, video decoding, model load/caching, sequence length, OT iterations, and database availability. Benchmark claims must be accompanied by hardware, software versions, input distribution, and repeated measurements.
8.2 Security and privacy
- Secrets must be loaded from environment configuration and never committed to source control. Distributable archives should include .env.example, not a populated .env file.
- Archive hygiene: the supplied project ZIP contains a root .env in addition to .env.example. Remove the populated file from any distributable copy and rotate credentials if the archive has been shared outside the trusted development team.
- The research-engine API currently has no authentication or per-case authorization. The supplied Compose file publishes the API on 127.0.0.1:8001; do not change this to a public bind such as 0.0.0.0:8001 without an authenticated and security-reviewed gateway.
- The research API uses an explicit CORS_ALLOWED_ORIGINS allowlist with credentials disabled. CORS is a browser policy, not an authentication mechanism.
- The research API bounds uploaded evidence and statement sizes, validates case IDs and filenames, checks declared content type against supported extensions, decodes media before inference, and applies video duration/frame-count limits. These controls do not provide malware scanning, sandboxing, or a complete secure evidence-ingestion service.
- The supplied example defaults include a 100 MiB evidence upload limit, a 64 KiB statement byte limit, a 20,000-character statement limit, a 300-second video duration limit, and a maximum video frame-count guard. These limits are environment-configurable and must be reviewed for deployment.
- SHA-256 is computed over original uploaded bytes and checked by the evidence store on write/read. The repository does not implement signed acquisition logs, trusted timestamps, immutable/WORM storage, verified operator identity, or a full evidentiary chain-of-custody process.
- Filesystem permissions and administrators can still alter or delete stored files. Hash verification can detect a mismatch when a file is read; it does not prevent tampering.
- Evidence files, downloaded datasets, checkpoints, model outputs, and local caches must be excluded from Git unless intentionally selected for a controlled private artifact store.
8.3 Reliability and persistence
- Neo4j write operations should execute transactionally. The API reports graph persistence separately from the result of model analysis and from evidence-file storage.
- If Neo4j is unavailable, graph records are held in process memory and are reported as non-durable. A configured memory snapshot may be loaded/flushed, but it is not a transactional database or a guaranteed disaster-recovery mechanism.
- Evidence files use a separate filesystem store and Docker named volume in the supplied Compose deployment. A graph write failure does not necessarily mean that the original uploaded bytes were not preserved.
- Database reconnection/replay of volatile writes is not implemented as a durable write-ahead queue.
- Legacy graph nodes without case-ownership metadata require careful migration; automatic destructive cleanup is intentionally avoided.
- Docker Compose defines named volumes for Neo4j data, research evidence, downloaded/model cache data, and memory snapshot data. Operators remain responsible for tested backups, restoration, retention, and access control.
8.4 Research reproducibility
- Pin the CLIP model revision and record the resolved revision with experiment outputs.
- Save checkpoint/artifact metadata, data-split hashes, feature-schema version, inference settings, class mapping, and calibration metadata.
- Keep fit, calibration, and test partitions disjoint by the declared source-recording unit. For the supplied UCA group-v3 adapter, also retain statement-source recording provenance and audit it across partitions.
- Do not interpret camera_id as an independently verified physical camera identifier in the supplied UCA group-v3 data; it is derived from the source recording ID. The environment_id field represents an activity category and overlaps across partitions.
- Calibration guarantees depend on the calibration unit and exchangeability assumption. The supplied artifact uses the maximum true-label nonconformity score per source recording, not independent per-frame calibration.
- The supplied UCA test split has been used in prior ablations. Its saved results must be reported as exploratory/retrospective, not as a fully untouched confirmatory evaluation.
- Report class counts, pair-level metrics, recording-group coverage, abstention, selective accuracy, bootstrap uncertainty, and limitations. Do not report coverage as per-case accuracy.
8.5 Human oversight
Every automated assessment is a decision-support signal only. A reviewer must inspect original evidence and context. In particular, heuristic cross-case keyword overlaps, low-quality visual signals, and model abstentions require explicit review.
9. Verification and validation plan
Test area	Verification method	Acceptance condition
Sinkhorn mathematics	Unit tests with finite, non-square cost matrices and marginal checks	Solver converges within configured tolerance or fails explicitly; no silent invalid result
Partial OT and dustbin	Unit tests for finite transport plan, discarded mass, and trust-weight behaviour	Outputs remain finite and internally valid; unmatched mass is reported rather than hidden
Evidential head	Unit tests for shape, finite values, class probabilities and uncertainty	Alpha is finite and at least one; probabilities sum to one; uncertainty remains in valid range
Conformal density and scores	Unit tests for class-density fitting, feature-schema validation, quantiles and prediction sets	Correct class ordering and threshold are applied; invalid or incompatible artifacts are rejected
Recording-group calibration	Unit tests with grouped calibration IDs	Calibration score is the maximum true-label nonconformity within each source recording; group leakage is rejected
Decision mechanism configuration	Runtime configuration tests	Evidential and conformal decision mechanisms cannot be enabled simultaneously; neither configured yields INDETERMINATE
Feature pipeline	Mocked CLIP tests plus a separate real-model smoke test	Patch/token and feature shapes match the schema; no synthetic fallback features are used
Checkpoint/artifact compatibility	Mismatch tests	Wrong model revision, inference configuration, feature schema, checkpoint hash or class map is rejected
Abstention	Conformal set tests and evidential OOD/uncertainty tests	Non-singleton/empty conformal sets return INDETERMINATE; evidential abstention conditions apply only when that alternative is enabled
Dataset splits	Group-disjoint validation and provenance audit	No prohibited evidence-recording or statement-source-recording overlap across fit/calibration/test
Evidence storage	Write/read/hash-mismatch/path-validation tests	Original bytes are stored and read with matching SHA-256; modified bytes are detected; unsafe paths are rejected
Graph memory fallback	Unit tests without Neo4j	Memory-only writes report persisted=false and non-durable storage status
Graph transaction behaviour	Integration tests against a real Neo4j instance	Case creation/update, clear, delete and cross-case isolation behave transactionally
Frontend statuses	Component/browser test	MATCH, CONTRADICTION, and INDETERMINATE remain visually distinct
Frontend async isolation	Browser test with delayed responses	Upload result for a previous case does not overwrite the currently selected case
Production build	npm run build	Next.js production build succeeds in the supported Node environment
Empirical model validation	Appropriate held-out dataset with declared grouping	Pair metrics, recording-group coverage, abstention, bootstrap intervals and data limitations are reported without forensic overclaiming


9.1 Validation evidence recorded for the supplied repository snapshot
The repository's VALIDATION_STATUS.md records the following checks for the supplied snapshot:
- Research-engine test suite: 103 passed, 1 skipped. The skipped test is the live Neo4j integration test because NEO4J_TEST_URI was not configured.
- Python source syntax was checked against Python 3.11 grammar for 40 files; no syntax errors were reported. python -m compileall -q research-engine passed.
- A synthetic UCA adapter smoke run generated fit/calibration/test CSVs and confirmed source-recording disjointness and annotation-source provenance for that smoke run.
- The supplied legacy UCA CSV metadata audit recorded zero source-recording overlap across fit/calibration/test.
- An automated check verified that Docker Compose host-published ports are loopback-bound.
The same validation record explicitly states that the following were not verified in that environment:
- Real UCA CLIP inference and updated metrics, because the complete source image assets referenced by the CSVs were not included in the ZIP;
- live Neo4j transactions/integration;
- frontend npm ci, Vitest execution, and the Next.js production build;
- public-network security, authentication, authorization, penetration testing, backups, and production evidence governance.
A saved test/evaluation artifact is not the same as independently rerunning the underlying model experiment. The saved UCA group-v3 metrics below must therefore be described as results recorded in the supplied experiment artifacts, not as an independently reproduced run from this ZIP.
9.2 Saved UCA group-v3 conformal evaluation (artifact-derived)
The supplied uca_group_v3 split contains:
Partition	Pairs	MATCH	CONTRADICTION	Source recordings
Fit	3,136	1,568	1,568	1,568
Calibration	140	70	70	70
Test	140	70	70	70


For the active three-feature Partial-OT conformal artifact at \(\alpha=0.10\), the saved grouped-v3 report records:
Metric	Saved result
Conformal threshold \(\hat q\)	0.6086335
Pair-level empirical coverage	93.57%
Singleton prediction rate	30.00%
Selective accuracy	78.57%
Abstention rate	70.00%
Average prediction-set size	1.70
Recording groups simultaneously covered	61 / 70 (87.14%)


The test report's pair-level coverage and recording-group simultaneous coverage are different quantities and must not be conflated. The nominal conformal target is 90% at the declared recording-group calibration unit under exchangeability; the observed held-out group coverage is the descriptive result shown above. This does not establish individual prediction correctness, forensic accuracy, or robustness to distribution shift.
9.3 Saved Full OT versus Partial OT comparison (two-feature ablation)
The saved final reproduction report compares a matched two-feature variant using Sinkhorn divergence plus one minus global CLIP similarity. This is not the active three-feature all_three conformal artifact described in Section 9.2.
Metric	Full OT	Quality-weighted Partial OT
Pair-level empirical coverage	94.29%	93.57%
Singleton prediction rate	32.86%	27.14%
Selective accuracy	82.61%	76.32%
Abstention rate	67.14%	72.86%
Recording-group simultaneous coverage	88.57% (62/70)	87.14% (61/70)


The associated paired recording-group bootstrap used 2,000 replicates over 70 source-recording groups. The observed Full-OT-minus-Partial-OT selective-accuracy difference was 6.29 percentage points (95% percentile interval: 2.02 to 11.67 percentage points). The observed pair-level coverage difference was 0.71 percentage points (95% interval: 0.00 to 2.14 percentage points). These intervals describe resampling uncertainty on this proxy dataset only; they do not establish generalization or forensic validity.

The current unit suite is not a substitute for a live Neo4j integration run, a complete Next.js build, real CLIP inference over the complete source dataset, or independent empirical validation.
10. Limitations and future research
10.1 Retired experimental artifacts
The historical multimodal_fusion.py random-projection experiment and the Phase 1 sandbox are not part of the current inference pipeline. The prior sandbox's synthetic 84.5% / PROVED output is not supported by a labelled, group-disjoint held-out evaluation and must not be represented as a measured project result. test_clip_calib.py is a deprecation notice. Current implementation and research claims must be based on the active CLIP/OT/conformal path and the explicitly documented evaluation artifacts in this SRS.
10.2 UCA proxy dataset and interpretation limits
The supplied group-v3 research data is derived from UCF-Crime videos and UCFCrime.json descriptions. It is a proxy dataset for image–description compatibility, not a corpus of independently collected witness statements.

The group-v3 pairing policy is:
- MATCH: an image/frame is paired with the annotation description from its own source recording.
- CONTRADICTION: an image/frame is paired with an annotation description from a different source recording and a different activity category within the same partition.
Accordingly, CONTRADICTION means a constructed cross-recording, cross-category description mismatch. It is not a human-adjudicated finding that a witness statement is factually contradicted by visual evidence. The labels do not establish whether a particular person, object, action, intent, or event is present in the frame.

The group-v3 split contains 3,136 fit pairs from 1,568 source recordings, 140 calibration pairs from 70 recordings, and 140 test pairs from 70 recordings. The supplied CSVs retain both source_recording_id and statement_source_recording_id; the recorded adapter/split audit shows these IDs are disjoint across the fit, calibration, and test partitions. The camera_id field is derived from the source recording ID and is not an independently verified physical camera identifier. environment_id represents activity categories and can overlap across partitions. No physical-camera-disjoint generalization claim is supported.

The source image paths in the supplied CSVs point to separately held UCF-Crime/UCA dataset assets. The complete source image collection is not included in the supplied ZIP, so the real-image CLIP experiment cannot be independently rerun from the ZIP alone. In addition, the existing test split has been used in prior ablations; report the saved results as exploratory/retrospective, not as a fully untouched confirmatory evaluation.
10.3 Capabilities not implemented or not validated
The following items are not claimed as implemented or validated in the current baseline:
1. A CCTV-specific, human-labelled dataset of genuine witness statements and independently adjudicated visual support/contradiction labels, with fit/calibration/test splits designed for the intended generalization claim.
2. Physical-camera-disjoint evaluation based on verified camera identities and broader environmental separation.
3. Object detection, object localization, color/attribute verification, and supervised phrase grounding.
4. Robust video-level temporal modelling beyond N-point sampling and one sharpness-selected representative frame.
5. Calibrated entity resolution for linking suspects, vehicles, weapons, or modus operandi across cases.
6. A grounded local LLM forensic briefing or an LLM-based factual verifier. The frontend analysis panel presents the returned model/audit information; the current analysis endpoint does not invoke an LLM.
7. Durable queuing/replay during Neo4j outages, complete legacy data migration, and tested database disaster recovery.
8. Complete authentication/authorization for all services, signed immutable chain-of-custody infrastructure, trusted acquisition timestamps, and production security review.
9. Independent forensic validation, legal review, and any claim of court admissibility.
These items are future research or production-hardening work, not features delivered by the current repository snapshot.

End of implementation-aligned Software Requirements Specification — CrimeVision AI.