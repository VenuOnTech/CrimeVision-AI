CrimeVision AI
CrimeVision AI is a local research and decision-support prototype for comparing an uploaded image or a representative frame sampled from a video with a supplied witness statement. It combines frozen CLIP image–text representations, a quality-aware Optimal Transport (OT) alignment diagnostic, a split-conformal prediction layer, and a Neo4j case/evidence graph.
Research and safety boundary
CrimeVision AI is not a validated forensic instrument. Its outputs do not establish what physically occurred, identify a person, prove that a witness statement is true or false, or constitute a legal conclusion. The labels MATCH and CONTRADICTION are compatibility assessments evaluated using proxy labels described below. A qualified human reviewer must independently examine the original evidence. Do not use this prototype as the sole basis for investigative, prosecutorial, or judicial action.

Contents
- What the application does
- Current implementation status
- System architecture
- AI and decision pipeline
- Dataset and experimental protocol
- Recorded research results
- Repository layout
- Requirements
- Run with Docker Compose
- Run services locally for development
- Research reproduction commands
- API overview
- Testing and validation status
- Limitations and security
- Team
What the application does
1. An investigator creates or selects a case in the web dashboard.
2. The investigator uploads an image or video and supplies a non-empty witness statement. The statement can be entered as text or uploaded as a UTF-8 .txt file.
3. The research engine validates the media and statement, calculates a SHA-256 digest for the original evidence file, and stores the uploaded bytes separately from the graph metadata.
4. For an image, the image is analysed directly. For a video, the engine samples frames and selects one representative frame using a sharpness heuristic. The current video path does not perform temporal or whole-video understanding.
5. The inference pipeline extracts CLIP features, calculates global similarity and patch/token alignment diagnostics, and applies the configured decision mechanism.
6. The dashboard displays the assessment, heatmap and diagnostics, and exposes the evidence and case relationships for human review.
The dashboard also provides case management, evidence records, case/evidence graph views, and cross-case relationship review. Cross-case suggestions are heuristic keyword-overlap candidates; they are not verified links between people, vehicles, weapons, or events.
Supported uploads
Input	Supported formats / handling
Images	JPEG, PNG, BMP, WEBP and TIFF
Videos	MP4, AVI, MOV and MKV; sampled at nine points by default between 10% and 90% of the clip
Witness statement	Text entered in the form or a UTF-8 .txt file; the statement must be non-empty


The video sampler chooses one decoded frame using an edge-energy-to-noise sharpness score. That single frame is passed to the image–text pipeline. The default maximum video duration is 300 seconds; resource limits can be changed through environment variables.
Current implementation status
Capability	Current implementation
Case management	Create, retrieve, update and delete cases through the FastAPI service and dashboard
Image ingestion	Validates supported image extensions against the decoded image format before analysis
Video ingestion	Samples multiple frames and selects one sharpness-based representative frame; it does not analyse temporal events
Witness-statement input	Accepts text or a .txt upload; rejects empty statements and does not insert a synthetic statement
CLIP feature extraction	Frozen openai/clip-vit-base-patch32 backbone; extracts global image/text features, visual patch features and text-token features
Optimal Transport	Log-domain Sinkhorn implementation with full-OT and partial-OT modes; the shipped runtime configuration selects partial OT
Image-quality measurements	Used to derive patch-level trust for partial OT; they are not themselves model epistemic uncertainty
Conformal decision mechanism	Active in the supplied configuration through the schema-3, recording-group-calibrated artifact research-engine/models/uca_conformal_v3.json
Evidential classifier	Alternative trainable head and calibration path exist in the code, but they are not configured as the active decision mechanism in the supplied .env.example
Evidence integrity	SHA-256 digest, evidence manifest, file retrieval and hash-verification endpoints are implemented
Case/evidence graph	Neo4j integration with a process-local memory fallback and optional snapshot support
Cross-case suggestions	Heuristic suggestions based on shared text/keywords; investigator review is required
Frontend	Next.js/React dashboard with case tabs, evidence table, analysis panel, heatmap and graph visualizations
Real-world forensic validation	Not established; the supplied experiments use UCF-Crime-derived proxy pairs


If neither a valid conformal artifact nor a trained evidential head plus its matching calibration file is loaded, the engine returns INDETERMINATE while still exposing available diagnostics. OT or CLIP similarity values must not be converted manually into a fallback verdict.
System architecture
The repository contains one FastAPI research-engine service. It is both the API layer and the host for the inference pipeline; it is not a separate business API plus a second independent ML server.
Investigator browser
       |
       v
Next.js 13 / React 18 dashboard
       |
       |  NEXT_PUBLIC_RESEARCH_API_URL
       v
FastAPI research engine (port 8001)
       |
       +--> Request, statement and media validation
       |
       +--> OpenCV / Pillow
       |      +--> image validation, or
       |      +--> video frame sampling and representative-frame selection
       |
       +--> SHA-256 hashing and evidence-file storage
       |
       +--> Frozen Hugging Face CLIP
       |      +--> global image/text representations
       |      +--> visual patch and text-token representations
       |
       +--> Image-quality measurements and patch trust
       |
       +--> Sinkhorn OT alignment diagnostics
       |
       +--> Configured decision mechanism
       |      +--> split-conformal prediction (active in supplied configuration), or
       |      +--> calibrated evidential head (alternative path)
       |
       +--> Neo4j case/evidence graph
       |      +--> process-local memory fallback when Neo4j is unavailable
       |
       v
Assessment, diagnostics, evidence metadata and graph data
       |
       v
Investigator dashboard
Docker services and ports
Service	Technology	Host address
frontend-web	Next.js / React	http://127.0.0.1:3000
research-engine	FastAPI, CLIP and OT pipeline	http://127.0.0.1:8001
neo4j	Neo4j Community 5.11	Browser: http://127.0.0.1:7474; Bolt: bolt://127.0.0.1:7687


The Docker Compose configuration binds the published ports to loopback (127.0.0.1). Do not expose these services on a public interface without adding appropriate authentication, authorization, network controls and security review.
AI and decision pipeline
1. Frozen CLIP representations
The configured backbone is openai/clip-vit-base-patch32. Its parameters are put into evaluation mode and frozen; the project does not fine-tune the CLIP backbone as part of normal inference.

For each image–statement pair, the engine obtains:
- Global image and text representations for an overall similarity measurement.
- Visual patch representations. With the ViT-B/32 input configuration, the visual patch grid is 7 × 7 (49 patches).
- Text-token representations after removing special tokens.
The global cosine similarity is one of the features used by the conformal scorer. The patch and token representations are used to calculate local alignment diagnostics and the heatmap.
2. Image quality and partial Optimal Transport
The engine calculates image-quality measurements and patch sharpness. Under the supplied TRUST_MODE=heterogeneity configuration, patch trust is derived from relative patch sharpness. The trust values influence the partial-OT mass allocation.

The OT solver builds a patch-to-token dissimilarity matrix and uses a log-domain Sinkhorn procedure. Its partial-transport implementation includes a dustbin mechanism so that some mass can remain unmatched. The engine reports diagnostics such as Sinkhorn divergence, discarded mass, the transport plan and the patch/token cost matrix.

The heatmap is a feature-alignment diagnostic. It is not an object detector, an object-location explanation, or proof that a particular word refers to a particular physical object in the image.

Full OT is also implemented and is used in comparative experiments. The supplied runtime artifact records force_full_ot=false, so the shipped inference configuration uses partial OT.
3. Active split-conformal decision mechanism
The supplied root .env.example selects the conformal path:
EVIDENTIAL_HEAD_PATH=
EVIDENTIAL_CALIBRATION_PATH=
CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json
The conformal predictor uses three features, in this order:
1. sinkhorn_divergence
2. discarded_mass
3. one_minus_global_clip_similarity
The fit split is used to fit class-conditional Gaussian feature-density models for MATCH and CONTRADICTION. The calibration split is then used to calculate nonconformity scores. In schema 3, calibration is performed at the source_recording_id group level: the maximum true-label score within each calibration recording is used as that recording's calibration score. The finite-sample conformal quantile is then calculated across recording groups.

The supplied artifact records:
Artifact setting	Value
Artifact	research-engine/models/uca_conformal_v3.json
Schema	3
Alpha	0.10
Nominal group-level coverage target	90%
Calibration groups	70 source recordings
Calibration pairs	140
q_hat	0.6086335403
Feature count	3


The predictor returns a set of labels:
- {MATCH} becomes MATCH.
- {CONTRADICTION} becomes CONTRADICTION.
- {MATCH, CONTRADICTION} becomes INDETERMINATE.
- An empty prediction set also becomes INDETERMINATE.
This is a selective decision mechanism: it can abstain rather than force a binary assessment. The conformal guarantee is a simultaneous-within-recording-group coverage statement under exchangeability of recording groups and the same group-construction protocol. It is not class-conditional, does not guarantee that a singleton prediction is correct, does not bound the abstention rate, and does not protect against distribution shift.
4. Alternative evidential route
The repository includes an evidential pair head and associated calibration code under research-engine/src/. FastAPI supports an alternative configuration using both EVIDENTIAL_HEAD_PATH and EVIDENTIAL_CALIBRATION_PATH together. The evidential route and conformal route are mutually exclusive in the current inference initialization. The supplied configuration does not load the evidential checkpoint, so the current application results should be described as conformal results, not as outputs from the evidential head.
Dataset and experimental protocol
Source and label construction
The grouped-v3 adapter uses UCF-Crime video frames and descriptions from UCFCrime.json to construct image–description pairs:
- MATCH: an image is paired with the annotation description from its own source recording.
- CONTRADICTION: an image is paired with a description from a different source recording and a different activity category within the same partition.
CONTRADICTION is therefore a synthetic cross-category description-mismatch proxy label. It is not a human-verified contradiction between real witness testimony and evidence. The project must not describe these labels as forensic ground truth.
Supplied grouped-v3 partitions
The current conformal artifact's recorded fit and calibration CSV hashes match the grouped-v3 dataset files. Their recorded split sizes are:
Partition	Pairs	MATCH	CONTRADICTION	Source recordings
Fit	3,136	1,568	1,568	1,568
Calibration	140	70	70	70
Test	140	70	70	70


The grouped-v3 CSVs contain these columns:
image_path,statement,label,camera_id,environment_id,
source_recording_id,statement_source_recording_id,
statement_source_environment_id
The supplied grouped-v3 CSVs have no overlap between fit, calibration and test for either source_recording_id or statement_source_recording_id. This prevents the same source recording from being used across those partitions under the recorded construction. It does not establish independent physical cameras: camera_id is derived from the source-recording identifier, not measured from camera hardware. environment_id represents a UCF-Crime activity category and may overlap across partitions.
Dataset availability and reproduction boundary
The ZIP includes the grouped-v3 CSVs and experiment artifacts, but it does not include the complete original UCF-Crime/UCA image assets and source annotation dataset referenced by the CSV paths. A small number of selected review frames and contact sheets are present in the reports directory, but they are not a replacement for the full source dataset. Consequently, the metadata, split definitions and saved results can be inspected, but the full CLIP feature extraction and model evaluation cannot be independently rerun from this ZIP alone. Restore the original dataset and preserve its directory structure before running the research scripts.
Recorded research results
The numbers below are transcribed from the supplied grouped-v3 reports. They are exploratory/retrospective results on this proxy dataset, not real-world forensic performance estimates.
Active three-feature conformal artifact
The following result corresponds to the three-feature all_three grouped ablation and the supplied uca_conformal_v3.json artifact (alpha=0.10, q_hat=0.6086335403):
Metric	Recorded result
Test pairs	140
Test recording groups	70
Pair-level empirical coverage	93.57%
Singleton prediction rate	30.00%
Selective accuracy	78.57%
Abstention rate	70.00%
Average prediction-set size	1.70
Recording-group simultaneous coverage	61 / 70 groups = 87.14%


The pair-level coverage and group-level simultaneous coverage are different summaries. The nominal 90% conformal target applies to simultaneous inclusion within an exchangeable recording group, not to per-pair correctness. The observed group-level coverage on this finite test split was 87.14%; it should be reported as observed, not rewritten as 90% measured accuracy or guaranteed per-case correctness.
Separate full-OT versus partial-OT comparison
The grouped-v3 two-feature comparison below uses sinkhorn_divergence and one_minus_global_clip_similarity. It is a separate ablation and not the same three-feature configuration as the active artifact above.
Metric	Full OT	Partial OT
Pair-level empirical coverage	94.29%	93.57%
Singleton prediction rate	32.86%	27.14%
Selective accuracy	82.61%	76.32%
Abstention rate	67.14%	72.86%
Recording-group simultaneous coverage	88.57%	87.14%


For this particular two-feature experiment, Full OT performed better on singleton rate and selective accuracy. This is a comparison result, not evidence that Full OT will always outperform Partial OT or that either method is forensically valid.
Evaluation caveats
- The supplied test split appears in earlier ablation and robustness artifacts. Results on this split are therefore retrospective/exploratory rather than a completely untouched confirmatory evaluation.
- The grouped-v3 label semantics are synthetic proxy labels, not human-adjudicated witness-statement contradictions.
- Recording-disjoint evaluation is not physical-camera-disjoint evaluation.
- Controlled image-degradation experiments are stress tests on the proxy data; they do not establish robustness to real CCTV conditions or semantic reliability.
- The recorded results cannot be independently recomputed from this archive because the full referenced image assets are absent.
Repository layout
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
│   │   ├── evidential_fusion.py           # Alternative evidential head and OT implementation
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
Requirements
- Docker Engine/Desktop with Docker Compose (recommended for the complete application).
- For local development: Python 3.10 or newer (the research-engine Docker image uses Python 3.11), Node.js 18 or newer, and npm.
- Sufficient disk space for the CLIP model cache, Docker images/volumes and any locally restored research dataset.
- Internet access on first model load to download the configured Hugging Face CLIP files, unless they are already cached.
- A CUDA-capable GPU is optional. DEFAULT_DEVICE=auto selects CUDA when available to the process and otherwise uses CPU. GPU passthrough must be configured separately if running inside Docker.
Run with Docker Compose (recommended)
Run these commands from the repository root.
1. Create and configure the environment file
PowerShell:
Copy-Item .env.example .env
macOS / Linux:
cp .env.example .env
Edit .env and replace YOUR_ACTUAL_NEO4J_PASSWORD with a strong local development password. Keep .env private and do not commit it. The supplied .env.example already points the research engine to the packaged schema-3 artifact:
CONFORMAL_ARTIFACT_PATH=/app/models/uca_conformal_v3.json
The Compose file mounts research-engine/models/ read-only at /app/models/ inside the research-engine container.
2. Build and start all services
docker compose up --build -d
Check service status:
docker compose ps
Follow the research-engine logs if model loading or startup fails:
docker compose logs -f research-engine
The CLIP backbone may be downloaded from Hugging Face during the first model initialization. Initial startup can therefore take longer than later starts.
3. Open the application
- Investigator dashboard: http://127.0.0.1:3000
- FastAPI interactive API documentation: http://127.0.0.1:8001/docs
- FastAPI health endpoint: http://127.0.0.1:8001/health
- FastAPI readiness endpoint: http://127.0.0.1:8001/ready
- Neo4j Browser: http://127.0.0.1:7474
Use the Neo4j username and password configured in .env to access Neo4j Browser.
4. Stop the services
docker compose down
To remove the named database/evidence volumes as well, use docker compose down -v. This deletes persisted local project data; do not run it if you need to keep cases or evidence.
Run services locally for development
Docker Compose is the simplest way to run the complete application. For code development, the services can also be run separately. Start Neo4j first, for example with docker compose up -d neo4j, and configure the research-engine environment variables in the shell that launches Uvicorn.
1. Research engine
cd research-engine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Set the local environment values in PowerShell (use your own Neo4j password):
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "YOUR_LOCAL_NEO4J_PASSWORD"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
$env:DEFAULT_DEVICE = "auto"
$env:CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
$env:CLIP_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
$env:TRUST_MODE = "heterogeneity"
$env:CONFORMAL_ARTIFACT_PATH = "models/uca_conformal_v3.json"

python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
For bash/zsh, use the equivalent export VARIABLE=value commands. The CONFORMAL_ARTIFACT_PATH above is relative to research-engine/.
2. Frontend
In a second terminal:
cd frontend-web
npm ci
Create frontend-web/.env.local:
NEXT_PUBLIC_RESEARCH_API_URL=http://127.0.0.1:8001
Start the development server:
npm run dev
Open http://127.0.0.1:3000.
Research reproduction commands
The following commands require the original UCA/UCF-Crime image assets to be restored and the image_path entries in the CSV files to resolve correctly. The supplied archive does not contain the complete image dataset, so these commands are workflow examples rather than a claim that the full experiment can be rerun from the ZIP alone.

Run research commands from research-engine/.
Refit a schema-3 recording-group conformal artifact
Use separate recording-disjoint fit and calibration CSVs:
python calibrate_conformal.py \
  --fit-csv datasets/uca_group_v3/fit.csv \
  --calibration-split-csv datasets/uca_group_v3/calibration.csv \
  --output models/uca_conformal_rebuilt.json \
  --alpha 0.10 \
  --model-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --trust-mode heterogeneity
Evaluate on the separate test partition
python evaluate_research_engine.py \
  --conformal models/uca_conformal_rebuilt.json \
  --test-csv datasets/uca_group_v3/test.csv \
  --output reports/uca_group_v3_test_metrics.json \
  --sweep-alpha 0.05,0.10,0.20,0.30 \
  --model-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
Do not tune the model, select features or choose trust parameters using the test partition. Retain the CSV hashes, experiment manifest and raw metrics report with any result presented to reviewers.

Related experiment scripts include:
- ablate_conformal_scores.py — compares conformal feature subsets.
- ablate_full_vs_partial_ot.py — compares Full OT and quality-weighted Partial OT.
- evaluate_image_robustness.py and evaluate_image_robustness_bootstrap.py — evaluate selected image-degradation settings and recording-group resampling uncertainty.
- datasets/adapt_uca.py and datasets/prepare_ucf_crime_proxy.py — dataset adaptation and proxy-pair preparation.
See RESEARCH_PROTOCOL.md for the fuller protocol, experiment manifests, resampling units and claim boundaries.
API overview
The complete endpoint schemas are available through the FastAPI Swagger UI at /docs. Important routes include:
Method	Route	Purpose
GET	/health	Service health
GET	/ready	Readiness status
GET, POST	/cases	List and create cases
PUT, DELETE	/cases/{case_id}	Update or delete a case
GET	/graph/{case_id}	Retrieve a case graph
GET	/graph/all	Retrieve the global graph
POST	/cross_case_relation	Record a cross-case relationship for review
GET	/cross_case_relations	Retrieve recorded cross-case relationships
GET	/cross_case_suggestions	Retrieve heuristic keyword-overlap candidates
POST	/analyze_evidence/	Ingest and analyse evidence with a statement
GET	/evidence/{case_id}	Retrieve evidence metadata
GET	/evidence/{case_id}/manifest	Retrieve the evidence manifest
GET	/evidence/{case_id}/{sha256}/file	Retrieve a stored evidence file after integrity checking
POST	/evidence/{case_id}/verify	Compare an uploaded file's SHA-256 with the stored evidence record
GET	/admin/memory	Inspect memory-fallback status
POST	/admin/memory/flush	Flush the configured memory snapshot


Testing and validation status
Research-engine tests
From research-engine/:
python -m pytest -q
Frontend tests and build
From frontend-web/:
npm ci
npm test
npm run build
Existing validation record in this archive
VALIDATION_STATUS.md records the following checks from the supplied remediation-validation run:
- Research-engine tests: 103 passed, 1 skipped. The skipped test requires a live Neo4j test server.
- Python source syntax and compileall: passed for the checked research-engine files.
- Synthetic UCA adapter smoke run and source-recording split audit: passed.
- Docker Compose host-published ports: checked as loopback-bound.
That same validation record says the following were not verified in that run:
- Full real-UCA CLIP inference and updated model metrics, because the referenced image assets were absent.
- Live Neo4j transaction/integration behaviour.
- Frontend npm ci, Vitest execution and Next.js production build.
- Public-network security, authentication, authorization, penetration testing, production backups and production evidence governance.
These are results recorded in the archive's validation document, not a claim that those checks were rerun while updating this README.
Limitations and security
- Not a forensic instrument: the system is a research prototype and must not be treated as proof or legal evidence of what occurred.
- Proxy labels: MATCH and CONTRADICTION in the supplied experiment are constructed UCF-Crime description-pair labels, not human-verified witness-statement judgments.
- No identity recognition: the project does not identify suspects or perform face recognition.
- No object detector or grounded explanation: patch/token heatmaps show feature alignment diagnostics, not verified object locations or word-to-object explanations.
- Limited video analysis: one representative frame is selected from sampled frames; temporal actions and events across the video are not modelled.
- Heuristic cross-case candidates: shared words are candidates for investigator review, not proof of a shared entity or event and not calibrated probabilities.
- No active LLM analysis: the LLMAnalysisPanel is a frontend presentation component for analysis and diagnostics; its name does not mean that an LLM service is part of the active inference pipeline.
- Authentication is not implemented: the current login screen is a client-side demonstration gate; it does not validate a username/password against a server. There is no production user authentication, role-based access control or case-level authorization.
- Local prototype only: do not expose the current application on a shared or public network. The loopback-bound Compose ports reduce accidental network exposure but do not replace authentication or a security review.
- Evidence integrity boundary: SHA-256 can verify that retrieved bytes match a stored digest. By itself it does not provide signed, append-only chain of custody, immutable evidence storage, operator attribution or legal admissibility.
- Persistence boundary: Neo4j is the intended graph store. If it is unavailable, the fallback is process-local memory with optional snapshot support; this is not equivalent to durable production database persistence.
- Conformal boundary: the group-level coverage statement relies on exchangeability of recording groups and the same grouping process. It is not class-conditional, per-pair correctness, an abstention-rate guarantee or protection against distribution shift.
- Evaluation boundary: the supplied test split has already been used in prior development ablations; treat its results as exploratory/retrospective.
Before sharing the repository, remove any local .env file and other secrets. Share .env.example as the configuration template, and rotate any credential that may already have been exposed.
Team
- Shyamal — Computer Science & Engineering
- Moukthika — Computer Science & Engineering
- Venu — Computer Science & Engineering
- Niteesha — Computer Science & Engineering
Developed as a B.Tech capstone research prototype.