# Literature Review: CrimeVision AI

**Project:** CrimeVision AI — Multimodal Evidence Comparison and Investigation Decision-Support Prototype  
**Document Type:** Comprehensive Literature Review and Theoretical Foundation  
**Review Basis:** Original peer-reviewed literature, repository source code, research protocol, experiment manifests, and recorded evaluation reports  
**Date:** 18 November 2026  

---

> [!WARNING]
> ### Research Scope and Claim Boundary
> CrimeVision AI is an academic decision-support prototype, **not** a validated forensic instrument. The evaluations in the repository utilize synthetic proxy labels (`MATCH` and `CONTRADICTION`) derived from UCF-Crime description pairings; they do not constitute ground-truth witness verification or legal evidence. Outputs must always be subjected to independent qualified human examination.

---

## Abstract

CrimeVision AI investigates whether visual information in an image or a representative frame sampled from a surveillance video is compatible with an associated natural-language description. Its research pipeline combines a frozen Contrastive Language–Image Pre-training (CLIP) model, patch-to-token feature comparison, entropic Optimal Transport (OT), a quality-weighted Partial OT variant, and split-conformal prediction. The system is designed to provide a decision-support signal with three possible outcomes: `MATCH`, `CONTRADICTION`, and `INDETERMINATE`.

This literature review examines the theoretical and empirical foundations relevant to the methods employed by the project. It covers surveillance-video anomaly datasets, Vision Transformers (ViT), vision-language pretraining, image–text entailment, Optimal Transport, Sinkhorn divergences, Partial and unbalanced OT, image-quality metrics, conformal prediction, selective prediction, and evidential deep learning. It relates these foundational approaches to the concrete implementation in the repository and analyzes the strengths and limitations of the reported experiments.

The review concludes that CrimeVision AI integrates established theoretical methods into an end-to-end engineering pipeline, rather than proposing a novel foundation model architecture, an original OT theorem, or a new conformal coverage guarantee. Its active runtime decision mechanism is a three-feature, recording-group-calibrated split-conformal predictor. An evidential neural head is present in the repository as an alternative implementation but remains unselected in the shipped configuration. The dataset labels are synthetic cross-recording, cross-category description mismatches rather than human-adjudicated forensic contradictions. Consequently, reported metrics must be interpreted as exploratory benchmarks on a proxy dataset rather than demonstrations of forensic accuracy or real-world CCTV generalization.

**Keywords:** *CLIP, Vision-Language Learning, Image–Text Compatibility, Optimal Transport, Sinkhorn Divergence, Partial Optimal Transport, Conformal Prediction, Uncertainty Quantification, Selective Prediction, Surveillance Video.*

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Surveillance-Video Analysis and the UCF-Crime Dataset](#2-surveillance-video-analysis-and-the-ucf-crime-dataset)
  - [2.1 Surveillance Anomaly Detection](#21-surveillance-anomaly-detection)
  - [2.2 Dataset Adaptation and Proxy Label Construction](#22-dataset-adaptation-and-proxy-label-construction)
- [3. Vision Transformers and Vision–Language Representation Learning](#3-vision-transformers-and-visionlanguage-representation-learning)
  - [3.1 Vision Transformer Representations](#31-vision-transformer-representations)
  - [3.2 CLIP and Contrastive Image–Text Learning](#32-clip-and-contrastive-imagetext-learning)
  - [3.3 Global Matching vs. Fine-Grained Visual Entailment](#33-global-matching-vs-fine-grained-visual-entailment)
- [4. Optimal Transport for Local Cross-Modal Alignment](#4-optimal-transport-for-local-cross-modal-alignment)
  - [4.1 Optimal Transport Background](#41-optimal-transport-background)
  - [4.2 Entropic Regularisation and the Sinkhorn Algorithm](#42-entropic-regularisation-and-the-sinkhorn-algorithm)
  - [4.3 Debiased Sinkhorn Divergence](#43-debiased-sinkhorn-divergence)
  - [4.4 Partial and Unbalanced Optimal Transport](#44-partial-and-unbalanced-optimal-transport)
  - [4.5 Image-Quality-Based Patch-Trust Weighting](#45-image-quality-based-patch-trust-weighting)
  - [4.6 Semantic Interpretation of the Alignment Heatmap](#46-semantic-interpretation-of-the-alignment-heatmap)
- [5. Uncertainty Quantification and Selective Prediction](#5-uncertainty-quantification-and-selective-prediction)
  - [5.1 Split-Conformal Prediction](#51-split-conformal-prediction)
  - [5.2 Active Group-Calibrated Conformal Decision Mechanism](#52-active-group-calibrated-conformal-decision-mechanism)
  - [5.3 Selective Prediction and the Abstention Mechanism](#53-selective-prediction-and-the-abstention-mechanism)
- [6. Evidential Deep Learning as an Alternative Research Direction](#6-evidential-deep-learning-as-an-alternative-research-direction)
- [7. Synthesis: Methodological Mapping to CrimeVision AI](#7-synthesis-methodological-mapping-to-crimevision-ai)
- [8. Empirical Findings Available in the Repository](#8-empirical-findings-available-in-the-repository)
  - [8.1 Active Three-Feature Conformal Configuration](#81-active-three-feature-conformal-configuration)
  - [8.2 Comparative Evaluation: Full OT vs. Partial OT](#82-comparative-evaluation-full-ot-vs-partial-ot)
  - [8.3 Controlled Image-Degradation Experiments](#83-controlled-image-degradation-experiments)
  - [8.4 Evaluation Status and Test-Set Reuse](#84-evaluation-status-and-test-set-reuse)
- [9. Research Gaps and Project Positioning](#9-research-gaps-and-project-positioning)
- [10. Conclusion](#10-conclusion)
- [References](#references)
- [Repository Sources Reviewed](#repository-sources-reviewed)

---

## 1. Introduction

Surveillance systems produce vast quantities of video footage, while criminal investigations routinely rely on textual witness statements, investigative narratives, and dispatch logs. Comparing natural-language accounts with visual evidence presents substantial computational and semantic challenges:
- Visual scenes contain dense, multifaceted elements, only a fraction of which are referenced in a given witness statement.
- Critical descriptive attributes (e.g., clothing color, held objects, vehicle type) may be small, partially occluded, blurred, poorly illuminated, or absent from a particular camera perspective.
- Video feeds from closed-circuit television (CCTV) frequently suffer from low resolution, compression artifacts, and extreme noise.

Consequently, an automated decision-support system cannot rely solely on a single scalar image–text similarity metric. To be practically useful and ethically responsible, it must quantify uncertainty, expose spatial and lexical alignment diagnostics, and explicitly abstain from issuing a binary verdict when evidence is ambiguous or incomplete.

CrimeVision AI investigates this challenge by formulating image–statement compatibility assessment through a multi-stage pipeline:
1. Extracting frozen global representations and local token/patch sequences via a Vision-Language foundation model (CLIP).
2. Constructing an Optimal Transport alignment cost matrix between visual patches and text tokens.
3. Solving for transport plans using log-domain Sinkhorn iterations, incorporating image-quality-aware trust weighting and a partial-transport dustbin.
4. Feeding divergence, discarded mass, and global cosine similarity into a split-conformal prediction layer.
5. Emitting one of three well-defined output states: `MATCH`, `CONTRADICTION`, or `INDETERMINATE`.

Crucially, the labels `MATCH` and `CONTRADICTION` represent proxy compatibility metrics derived from synthetic dataset pairings. They do not represent forensic proof, do not identify suspects, and do not constitute legal conclusions. This review systematically analyzes the peer-reviewed literature underpinning these components and evaluates how CrimeVision AI positions itself within current research.

---

## 2. Surveillance-Video Analysis and the UCF-Crime Dataset

### 2.1 Surveillance Anomaly Detection
Surveillance-video research has historically focused on identifying temporal outliers and criminal activities without exhaustive frame-level supervision. Sultani, Chen, and Shah [1] introduced the landmark **UCF-Crime** dataset alongside a weakly supervised anomaly-detection framework utilizing multiple-instance learning (MIL) and ranking loss. UCF-Crime contains 1,900 untrimmed surveillance videos (128 hours) captured from real-world CCTV cameras, spanning 13 anomaly categories (e.g., Abuse, Arrest, Arson, Assault, Burglary, Robbery, Road Accidents, Stealing, Vandalism) as well as normal operational footage.

While Sultani et al. [1] established a vital benchmark for surveillance analysis, their core objective—ranking anomalous temporal segments within untrimmed video bags—differs fundamentally from cross-modal witness verification. An anomaly-detection model identifies *when* an unusual event transpires; it does not determine whether an arbitrary natural-language proposition accurately corresponds to visual elements within a given frame.

### 2.2 Dataset Adaptation and Proxy Label Construction
The CrimeVision AI repository adapts UCF-Crime via a dataset parsing utility (`adapt_uca.py`) paired with descriptive annotations from `UCFCrime.json`. It synthesizes image–statement pairs under the following protocol:
- **`MATCH`:** A selected video frame is paired directly with the textual annotation associated with its own source recording.
- **`CONTRADICTION`:** A video frame is paired with an annotation description drawn from a completely different source recording and a distinct activity category within the same partition.

This protocol generates a controlled, synthetic cross-category mismatch benchmark. However, it imposes critical interpretative constraints:
- **Semantic Imperfection of Proxy Labels:** A description from a differing activity category is not guaranteed to be strictly contradictory in a formal linguistic sense. Broad descriptions (e.g., *"a person walking near a street corner"*) may remain physically compatible with footage from another recording. Hence, `CONTRADICTION` denotes a synthetic proxy mismatch, not ground-truth perjury or forensic conflict.
- **Partition Disjointness:** The repository's `uca_group_v3` partition enforces strict separation of `source_recording_id` and `statement_source_recording_id` across the Fit (3,136 pairs; 1,568 source recordings), Calibration (140 pairs; 70 recordings), and Test splits (140 pairs; 70 recordings).
- **Camera Independence Limitations:** The dataset field `camera_id` is an alias derived from the source recording identifier rather than verified hardware metadata. Additionally, `environment_id` represents activity classes and overlaps across splits. Thus, while the partition guarantees recording-disjoint evaluation, it does **not** establish physical-camera-disjoint or environment-disjoint generalization.

---

## 3. Vision Transformers and Vision–Language Representation Learning

### 3.1 Vision Transformer Representations
Dosovitskiy et al. [2] introduced the **Vision Transformer (ViT)**, demonstrating that treating an image as a sequence of flattened spatial patches processed by a standard Transformer encoder achieves competitive or superior performance compared to traditional Convolutional Neural Networks (CNNs).

In CrimeVision AI, the visual backbone is `openai/clip-vit-base-patch32`. For standard $224 \times 224$ pixel input images, the architecture divides the scene into non-overlapping $32 \times 32$ pixel patches:
$$N = \left(\frac{224}{32}\right) \times \left(\frac{224}{32}\right) = 7 \times 7 = 49 \text{ patches}$$
After discarding the class token (`[CLS]`), the model yields an explicit spatial sequence of 49 localized visual feature vectors. This patch-based structure allows the downstream pipeline to perform fine-grained spatial comparisons against individual words/tokens, rather than compressing the entire image into a single global vector.

### 3.2 CLIP and Contrastive Image–Text Learning
Radford et al. [3] introduced **CLIP (Contrastive Language–Image Pre-training)**, training dual encoders (Vision Transformer and Text Transformer) across 400 million internet image–caption pairs using a symmetric infoNCE contrastive loss. CLIP projects visual and textual representations into a shared latent embedding space where matching pairs exhibit high cosine similarity:

$$\operatorname{sim}(\mathbf{g}_v, \mathbf{g}_t) = \frac{\mathbf{g}_v^\top \mathbf{g}_t}{\|\mathbf{g}_v\|_2 \|\mathbf{g}_t\|_2}$$

CrimeVision AI freezes the parameters of `clip-vit-base-patch32` (pinned to commit revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`). Inference extracts:
1. Normalized global image vector $\mathbf{g}_v \in \mathbb{R}^{512}$.
2. Normalized global text vector $\mathbf{g}_t \in \mathbb{R}^{512}$.
3. Sequence of normalized patch embeddings $\mathbf{v}_i \in \mathbb{R}^{512}$ for $i \in \{1, \dots, 49\}$.
4. Sequence of normalized token embeddings $\mathbf{t}_j \in \mathbb{R}^{512}$ for $j \in \{1, \dots, M\}$ (excluding special delimiter tokens).

While CLIP provides strong zero-shot transfer capabilities, global cosine similarity is an uncalibrated proxy for truth. High similarity indicates general semantic affinity, not factual consistency; conversely, low similarity can stem from complex framing or rare vocabulary rather than explicit contradiction. The project correctly avoids using raw CLIP similarity as a sole deterministic verdict.

### 3.3 Global Matching vs. Fine-Grained Visual Entailment
Image–text retrieval differs fundamentally from visual entailment. Xie et al. [4] formalized **Visual Entailment (VE)** via the **SNLI-VE** benchmark, framing the task as determining whether a natural-language hypothesis is *entailed*, *contradicted*, or *neutral* with respect to an image premise.

Visual entailment models must capture fine-grained asymmetric relationships (e.g., verifying attributes, numbers, and spatial relationships) where missing information implies neutrality rather than contradiction. CrimeVision AI does not implement a trained SNLI-VE three-way entailment classifier; instead, it relies on unsupervised feature distance metrics paired with conformal set-valued decision rules. Adapting genuine visual-entailment architectures remains an acknowledged area for future research.

---

## 4. Optimal Transport for Local Cross-Modal Alignment

### 4.1 Optimal Transport Background
Optimal Transport (OT) provides a principled mathematical framework for comparing probability distributions by finding a transport plan that minimizes the total cost of moving probability mass from a source distribution to a target distribution. In cross-modal vision-language alignment, visual patch features and text token features are treated as discrete distributions over latent feature space.

Rather than forcing a one-to-one mapping, OT computes a joint coupling that accounts for many-to-one and one-to-many relationships (e.g., multiple visual patches representing a single described vehicle, or a compound phrase describing a localized visual entity).

### 4.2 Entropic Regularisation and the Sinkhorn Algorithm
Solving exact Kantorovich OT via linear programming scales cubically with sample size, rendering it computationally prohibitive for real-time inference. Cuturi [5] introduced **entropic regularisation**, transforming OT into a strictly convex optimization problem solvable via the highly parallelizable **Sinkhorn algorithm**:

$$\operatorname{OT}_{\varepsilon}(a, b; \mathbf{C}) = \min_{\mathbf{P} \ge 0, \; \mathbf{P}\mathbf{1} = a, \; \mathbf{P}^\top\mathbf{1} = b} \left[ \langle \mathbf{P}, \mathbf{C} \rangle + \varepsilon \, \mathrm{KL}(\mathbf{P} \,\|\, a b^\top) \right]$$

where:
- $\mathbf{C} \in \mathbb{R}^{N \times M}$ is the pairwise cost matrix, defined in CrimeVision AI as:
  $$C_{ij} = \max\left(0, 1 - \mathbf{v}_i^\top \mathbf{t}_j\right)$$
- $a \in \Delta^N$ and $b \in \Delta^M$ are the marginal mass distributions over patches and tokens.
- $\varepsilon > 0$ is the regularization parameter controlling entropy smoothness.
- $\mathrm{KL}(\mathbf{P} \,\|\, a b^\top)$ is the Kullback–Leibler divergence.

The Sinkhorn algorithm solves this problem through iterative matrix-scaling operations:
$$\mathbf{P} = \operatorname{diag}(u) \, \mathbf{K} \, \operatorname{diag}(v), \qquad \mathbf{K}_{ij} = \exp\left(-\frac{C_{ij}}{\varepsilon}\right)$$
CrimeVision AI implements this solver in the log domain to guarantee numerical stability and prevent floating-point underflow.

### 4.3 Debiased Sinkhorn Divergence
While entropic OT is computationally efficient, adding entropy introduces an entropic bias: $\operatorname{OT}_\varepsilon(a, a; \mathbf{C}) \ne 0$. Feydy et al. [6] resolved this through the **Sinkhorn divergence**, which subtracts the self-transport costs to yield a positive-definite, debiased divergence that interpolates between Optimal Transport and Maximum Mean Discrepancy (MMD):

$$S_{\varepsilon}(V, T) = \operatorname{OT}_{\varepsilon}(V, T) - \frac{1}{2}\operatorname{OT}_{\varepsilon}(V, V) - \frac{1}{2}\operatorname{OT}_{\varepsilon}(T, T)$$

CrimeVision AI incorporates $S_{\varepsilon}(V, T)$ as a primary diagnostic alignment score and as an input feature to the conformal prediction layer.

### 4.4 Partial and Unbalanced Optimal Transport
Standard OT strictly conserves total mass ($\sum_{ij} P_{ij} = 1$), requiring every visual patch to find a corresponding text token. In real surveillance scenarios, this assumption fails: background scenery (sky, asphalt, walls) has no counterpart in a focused witness statement, and conversely, statements may mention unobservable details.

- **Partial Optimal Transport (POT):** Formalized by Figalli [7], POT transports only a fraction of mass $m < \min(\|a\|_1, \|b\|_1)$, leaving extraneous mass unassigned.
- **Unbalanced Optimal Transport (UOT):** Chizat et al. [8] relaxed hard equality marginal constraints using soft $\mathrm{KL}$ or Rényi divergence penalties.

CrimeVision AI adopts a discrete **dustbin-based partial transport** formulation. The cost matrix $\mathbf{C}$ is augmented with virtual dustbin rows and columns with fixed cost $c_{\text{dustbin}} = 10.0$. Patches or tokens whose pairwise matching costs exceed the dustbin cost deposit their mass into the dustbin. The solver reports `discarded_mass` as a quantitative measure of unaligned scene content.

### 4.5 Image-Quality-Based Patch-Trust Weighting
To prevent degraded or blurry patches from skewing alignment, CrimeVision AI weights the marginal mass vector $a$ using patch sharpness. Under the active `TRUST_MODE=heterogeneity`, local Laplacian variance is computed across the $7 \times 7$ grid:

$$\sigma_{\text{Laplace}}^2 = \operatorname{Var}\left(\nabla^2 I_{\text{patch}}\right)$$

Trust scores $\tau_i$ are normalized relative to the maximum patch sharpness within the same image:
$$\tau_i = \frac{\sigma_{\text{Laplace}, i}^2}{\max_k \sigma_{\text{Laplace}, k}^2}$$

Patches with low relative trust are allocated less mass or directed into the dustbin. While this provides a useful heuristic for downweighting severely blurred regions, relative within-image sharpness does not indicate absolute physical visibility: if an entire frame is degraded, relative normalization will artificially assign high trust to the least-degraded patch.

### 4.6 Semantic Interpretation of the Alignment Heatmap
The interactive dashboard visualizes the cost matrix $\mathbf{C} \in \mathbb{R}^{49 \times M}$ as a spatial heatmap. **This visualization must be interpreted cautiously:**
- It reflects mathematical cosine dissimilarity between deep feature embeddings.
- It does **not** represent a supervised bounding-box detector, an object segmentation mask, or proof that an identified object matches real-world entities.

---

## 5. Uncertainty Quantification and Selective Prediction

### 5.1 Split-Conformal Prediction
Conformal prediction, pioneered by Vovk, Gammerman, and Shafer [9], and popularized by Angelopoulos and Bates [10], provides a distribution-free framework for constructing prediction sets with rigorous finite-sample coverage guarantees.

Given a user-specified significance level $\alpha \in (0, 1)$, a split-conformal classifier guarantees that the true label $Y_{n+1}$ is contained within the prediction set $\mathcal{C}(X_{n+1})$ with marginal probability at least $1 - \alpha$:

$$P\left(Y_{n+1} \in \mathcal{C}(X_{n+1})\right) \ge 1 - \alpha$$

This guarantee holds for any black-box scoring function, provided the calibration examples and the test example satisfy the assumption of **exchangeability** (e.g., independently and identically distributed data).

### 5.2 Active Group-Calibrated Conformal Decision Mechanism
CrimeVision AI implements split-conformal prediction via `uca_conformal_v3.json` using three nonconformity features:
$$\mathbf{x} = \begin{bmatrix} S_\varepsilon(V, T) \\ \text{discarded\_mass} \\ 1 - \operatorname{sim}(\mathbf{g}_v, \mathbf{g}_t) \end{bmatrix}$$

1. **Density Fitting:** Class-conditional Gaussian feature density models $\mathcal{N}(\boldsymbol{\mu}_y, \boldsymbol{\Sigma}_y)$ are fitted on the disjoint Fit split ($N = 3,136$).
2. **Nonconformity Scoring:** Nonconformity scores are computed as negative log-likelihoods:
   $$s(x, y) = -\log f_y(x)$$
3. **Recording-Group Aggregation:** In surveillance video, multiple frames sampled from the same video are statistically dependent, violating simple i.i.d. exchangeability. The project addresses this by aggregating calibration scores at the recording-group level (70 calibration recording groups, 140 pairs):
   $$s_{\text{group}} = \max_{i \in \text{Recording}} s(x_i, y_i)$$
4. **Quantile Calculation:** The conformal threshold $\hat{q}$ is computed at the finite-sample adjusted level:
   $$\hat{q} = \text{Quantile}\left(\frac{\lceil (K+1)(1-\alpha) \rceil}{K}; \; \{s_{\text{group}}^{(k)}\}_{k=1}^K\right)$$
   For the active artifact, $\alpha = 0.10$ (90% nominal coverage target), yielding $\hat{q} = 0.6086335403$.

#### Operational Set-to-Status Mapping:
- $\mathcal{C}(X) = \{\text{MATCH}\} \longrightarrow$ **`MATCH`**
- $\mathcal{C}(X) = \{\text{CONTRADICTION}\} \longrightarrow$ **`CONTRADICTION`**
- $\mathcal{C}(X) = \{\text{MATCH}, \text{CONTRADICTION}\} \longrightarrow$ **`INDETERMINATE`** (Ambiguous)
- $\mathcal{C}(X) = \emptyset \longrightarrow$ **`INDETERMINATE`** (Out-of-Distribution)

The theoretical guarantee applies to **simultaneous coverage across all pairs within an exchangeable source recording**. It does not guarantee conditional coverage per camera, per-class correctness, or stability under domain shifts.

### 5.3 Selective Prediction and the Abstention Mechanism
Selective prediction models [12] optimize an explicit risk–coverage trade-off by allowing the model to abstain from uncertain predictions. CrimeVision AI achieves selective prediction naturally through conformal set sizing:
- **Singleton Rate:** Fraction of test samples resulting in $|\mathcal{C}(X)| = 1$.
- **Abstention Rate:** Fraction resulting in $|\mathcal{C}(X)| \ne 1$ (labeled `INDETERMINATE`).
- **Selective Accuracy:** Accuracy evaluated exclusively over accepted singleton predictions.

A high empirical coverage (e.g., 93.57%) is often coupled with a high abstention rate (e.g., 70.00%). Abstaining on difficult cases preserves the theoretical coverage guarantee while declining to issue unreliable binary verdicts.

---

## 6. Evidential Deep Learning as an Alternative Research Direction

Sensoy, Kaplan, and Kandemir [13] introduced **Evidential Deep Learning (EDL)** to place Dirichlet prior distributions over categorical probabilities, enabling neural networks to quantify both aleatoric (data) and epistemic (model) uncertainty in a single forward pass.

CrimeVision AI includes an implemented evidential classification head (`evidential_fusion.py`):
1. **Feature Input:** Concatenates 1,025 pair features ($512$ absolute differences, $512$ Hadamard products, $1$ Sinkhorn divergence) with $3$ standardized image-quality metrics ($1,028$ total dimensions).
2. **Dirichlet Formulation:**
   $$\mathbf{e} = \operatorname{softplus}(f_\theta(\mathbf{x})), \qquad \boldsymbol{\alpha} = \mathbf{e} + 1$$
   $$p_k = \frac{\alpha_k}{S}, \qquad u = \frac{K}{S}, \quad \text{where } S = \sum_{j=1}^K \alpha_j, \; K=2$$
   Here, $u \in [0, 1]$ represents Dirichlet-based epistemic uncertainty.

> [!IMPORTANT]
> **Configuration Reality:** While fully implemented and accompanied by test suites, the evidential head is **not active** in the supplied `.env.example`. The inference engine enforces mutual exclusivity between the evidential and conformal engines. Therefore, current repository results stem entirely from the split-conformal pipeline, and must not be reported as evidential uncertainty predictions.

---

## 7. Synthesis: Methodological Mapping to CrimeVision AI

| Research Area | Foundational Literature | Role in CrimeVision AI | Key Distinction / Limitation |
| :--- | :--- | :--- | :--- |
| **Surveillance Video** | Sultani et al. (CVPR 2018) [1] | Provides source CCTV footage and activity categories via UCF-Crime. | Original task is temporal anomaly detection, not natural-language statement verification. |
| **Vision Transformers** | Dosovitskiy et al. (ICLR 2021) [2] | Provides the patch sequence ($7 \times 7 = 49$) via ViT-B/32. | Patch embeddings represent latent spatial features, not verified semantic objects. |
| **Vision-Language Pretraining** | Radford et al. (ICML 2021) [3] | Supplies frozen dual image/text encoders and global cosine similarity. | Global CLIP similarity measures general semantic affinity, not factual truth or legal proof. |
| **Visual Entailment** | Xie et al. (arXiv 2019) [4] | Conceptual framework for support vs. contradiction vs. neutrality. | Project uses synthetic mismatch heuristics rather than a trained 3-way SNLI-VE model. |
| **Entropic Optimal Transport** | Cuturi (NeurIPS 2013) [5] | Log-domain Sinkhorn solver enables fast patch-to-token coupling. | OT cost serves as an alignment diagnostic, not an autonomous contradiction threshold. |
| **Sinkhorn Divergence** | Feydy et al. (AISTATS 2019) [6] | Debiased cross- and self-transport formulation $S_\varepsilon(V, T)$. | Divergence is one feature among three in the conformal decision vector. |
| **Partial & Unbalanced OT** | Figalli (ARMA 2010) [7]; Chizat et al. (Math. Comp. 2018) [8] | Motivates virtual dustbins and unmatched mass discarding. | Project utilizes a customized discrete dustbin implementation rather than continuous UOT theory. |
| **Conformal Prediction** | Vovk et al. (2005) [9]; Angelopoulos & Bates (2023) [10] | Provides finite-sample set-valued coverage guarantees. | Guarantees hold only under exchangeability at the recording-group aggregation unit. |
| **Non-Exchangeable Settings** | Oliveira et al. (JMLR 2024) [11] | Theoretical basis for managing grouped/dependent observations. | Recording-disjoint splits do not prevent distribution shift across unseen physical camera hardware. |
| **Selective Classification** | Geifman & El-Yaniv (ICML 2019) [12] | Conceptual framing for abstention (`INDETERMINATE`) and selective accuracy. | Uses conformal prediction set cardinality rather than SelectiveNet's rejection head. |
| **Evidential Deep Learning** | Sensoy et al. (NeurIPS 2018) [13] | Implements alternative Dirichlet uncertainty pair-head classifier. | Head is implemented but unselected in active deployment configuration. |
| **Hierarchical Conformal** | Dunn et al. (JASA 2023) [14] | Informs calibration over nested observations (frames in recordings). | Project applies a max-score heuristic rather than exact two-layer hierarchical conformalization. |

---

## 8. Empirical Findings Available in the Repository

The findings summarized below reflect saved evaluation reports (`research-engine/reports/group_v3/`) included in the project repository. Because raw video assets are omitted from the archive, these metrics represent verified historical records rather than newly executed benchmarks.

### 8.1 Active Three-Feature Conformal Configuration
Evaluated on the held-out `uca_group_v3` test partition ($N = 140$ pairs across 70 source recordings) using $\alpha = 0.10$ and $\hat{q} = 0.6086335403$:

| Metric | Recorded Value | Theoretical / Operational Meaning |
| :--- | :--- | :--- |
| **Test Image–Statement Pairs** | 140 | 70 MATCH pairs, 70 CONTRADICTION pairs |
| **Test Recording Groups** | 70 | 2 evaluated pairs per source recording |
| **Pair-Level Empirical Coverage** | **93.57%** | Percentage of test pairs where true label $\in \mathcal{C}(X)$ |
| **Singleton Prediction Rate** | **30.00%** | Percentage of test pairs resulting in single-label output |
| **Selective Accuracy** | **78.57%** | Accuracy evaluated exclusively over singleton predictions ($33 / 42$) |
| **Abstention Rate** | **70.00%** | Percentage of pairs yielding `INDETERMINATE` ($98 / 140$) |
| **Average Prediction-Set Size** | **1.70** | Mean cardinality $|\mathcal{C}(X)|$ across all test pairs |
| **Recording-Group Simultaneous Coverage**| **87.14%** | 61 of 70 recording groups had all pairs covered simultaneously |

> [!NOTE]
> Pair-level coverage (93.57%) and recording-group coverage (87.14%) represent distinct mathematical quantities. The nominal 90% target applies to simultaneous inclusion across recording groups under exchangeability; the observed 87.14% represents finite-sample realization.

### 8.2 Comparative Evaluation: Full OT vs. Partial OT
A separate two-feature ablation study (`sinkhorn_divergence` + `one_minus_global_clip_similarity`) compared Full OT against Quality-Weighted Partial OT:

| Metric | Full OT (Two-Feature) | Partial OT (Two-Feature) | Empirical Difference |
| :--- | :--- | :--- | :--- |
| **Pair-Level Empirical Coverage** | 94.29% | 93.57% | +0.71% |
| **Singleton Prediction Rate** | 32.86% | 27.14% | +5.71% |
| **Selective Accuracy** | **82.61%** | **76.32%** | **+6.29%** |
| **Abstention Rate** | 67.14% | 72.86% | -5.71% |
| **Recording-Group Simultaneous Coverage**| 88.57% (62/70) | 87.14% (61/70) | +1.43% |

A paired recording-level bootstrap analysis (2,000 replicates, seed 42) revealed an observed selective-accuracy advantage of 6.29 percentage points for Full OT, with a 95% percentile bootstrap confidence interval of $[2.02, 11.67]$. While Partial OT provides theoretical appeal by isolating unmatched mass into dustbins, it did **not** demonstrate empirical superiority over Full OT on this specific proxy benchmark.

### 8.3 Controlled Image-Degradation Experiments
The repository records stress tests evaluating model stability under synthetic image corruptions:
- Gaussian blur ($\sigma \in \{1.0, 2.0, 3.0\}$)
- Gaussian additive noise ($\sigma \in \{0.05, 0.10, 0.15\}$)
- Brightness shifts ($\pm 20\%, \pm 40\%$)
- JPEG compression artifacts ($Q \in \{50, 30, 15\}$)

The experimental protocol fits densities and calibrates thresholds exclusively on clean data, subsequently evaluating corrupted test frames without recalibration. While demonstrating resilience to moderate noise, synthetic corruptions cannot fully emulate optical lens distortion, low-light sensor gain, heavy motion blur, or severe rain/fog found in operational CCTV feeds.

### 8.4 Evaluation Status and Test-Set Reuse
As disclosed in `RESEARCH_PROTOCOL.md`, the `uca_group_v3` test partition was previously consulted during hyperparameter tuning and ablation studies. Consequently, these metrics must be reported as **exploratory and retrospective**, rather than a strictly confirmatory double-blind validation.

---

## 9. Research Gaps and Project Positioning

A critical synthesis of the literature reveals substantial gaps between current theoretical methodologies and operational investigative tooling:

1. **Proxy vs. Real Forensic Statements:** UCF-Crime provides anomaly classes, not authentic witness testimonies. Operational deployment requires curated CCTV benchmarks with multi-annotator witness statements and legal ground-truth adjudications.
2. **Feature Alignment vs. Semantic Grounding:** CLIP patch-level transport plans reveal feature similarity, not causal semantic grounding. The system cannot verify whether a detected *"blue shirt"* belongs to the suspect or an adjacent bystander.
3. **Temporal Reasoning Deficit:** The video ingestion pipeline selects a single representative frame via an edge-energy sharpness heuristic. It possesses zero temporal understanding of sequence, speed, event duration, or causal action progression.
4. **Exchangeability under Distribution Shift:** Conformal guarantees collapse when test distributions diverge from calibration data. Variations in camera angle, resolution, lighting, and geographic scene structure violate exchangeability assumptions.
5. **Camera Independence Verification:** Dataset splits partition recording IDs, but lack physical camera telemetry. True cross-camera generalization remains unverified.
6. **Confirmatory Data Independence:** The test partition has experienced reuse during exploratory research phases; fresh, untouched confirmatory benchmarks are required.
7. **Chain of Custody & Legal Governance:** Computing SHA-256 digests verifies byte integrity against file tampering; it does not constitute a legally admissible, cryptographic chain-of-custody log.

### Defensible Project Positioning
CrimeVision AI represents a rigorous **engineering integration and empirical exploration** of multimodal decision support. Its contributions lie in uniting frozen CLIP representations, entropic Optimal Transport, quality-aware dustbin mechanics, and group-calibrated split-conformal abstention into an operational architecture. It makes no claims of establishing new mathematical theorems, achieving forensic certainty, performing automated biometric identification, or producing court-ready evidence.

---

## 10. Conclusion

The academic literature supports the core design choices implemented in CrimeVision AI:
- **CLIP and ViT** provide robust zero-shot feature extraction across modalities.
- **Entropic Optimal Transport and Sinkhorn Divergences** enable structured, debiased local feature alignment without requiring full fine-tuning.
- **Partial OT Dustbins** offer a principled mechanism for handling unaligned visual and textual elements.
- **Split-Conformal Prediction** provides a mathematically grounded framework for selective abstention under exchangeability.
- **Evidential Deep Learning** represents a viable, implemented alternative for single-pass epistemic uncertainty estimation.

The system is best understood as a decision-support prototype designed to assist human investigators by filtering obvious mismatches, highlighting potential feature correspondences, and explicitly abstaining when uncertainty is elevated. Future work must bridge remaining gaps through genuine forensic video-statement datasets, multi-frame temporal modeling, supervised phrase grounding, and rigorous cross-camera domain adaptation.

---

## References

1. **W. Sultani, C. Chen, and M. Shah**, "Real-World Anomaly Detection in Surveillance Videos," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 6479–6488. DOI: [10.1109/CVPR.2018.00678](https://doi.org/10.1109/CVPR.2018.00678).
2. **A. Dosovitskiy et al.**, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *International Conference on Learning Representations (ICLR)*, 2021. arXiv: [2010.11929](https://arxiv.org/abs/2010.11929).
3. **A. Radford et al.**, "Learning Transferable Visual Models From Natural Language Supervision," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR, vol. 139, 2021, pp. 8748–8763. arXiv: [2103.00020](https://arxiv.org/abs/2103.00020).
4. **N. Xie, F. Lai, D. Doran, and A. Kadav**, "Visual Entailment: A Novel Task for Fine-Grained Image Understanding," *arXiv preprint*, arXiv: [1901.06706](https://arxiv.org/abs/1901.06706), 2019.
5. **M. Cuturi**, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 26, 2013, pp. 2292–2300.
6. **J. Feydy, T. Séjourné, F.-X. Vialard, S. Amari, A. Trouvé, and G. Peyré**, "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences," in *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 89, 2019, pp. 2681–2690.
7. **A. Figalli**, "The Optimal Partial Transport Problem," *Archive for Rational Mechanics and Analysis*, vol. 195, no. 2, pp. 533–560, 2010. DOI: [10.1007/s00205-008-0212-7](https://doi.org/10.1007/s00205-008-0212-7).
8. **L. Chizat, G. Peyré, B. Schmitzer, and F.-X. Vialard**, "Scaling Algorithms for Unbalanced Optimal Transport Problems," *Mathematics of Computation*, vol. 87, no. 314, pp. 2563–2609, 2018. DOI: [10.1090/mcom/3303](https://doi.org/10.1090/mcom/3303).
9. **V. Vovk, A. Gammerman, and G. Shafer**, *Algorithmic Learning in a Random World*, New York: Springer, 2005. DOI: [10.1007/b106715](https://doi.org/10.1007/b106715).
10. **A. N. Angelopoulos and S. Bates**, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023. DOI: [10.1561/2200000101](https://doi.org/10.1561/2200000101).
11. **R. I. Oliveira, P. Orenstein, T. Ramos, and J. V. Romano**, "Split Conformal Prediction and Non-Exchangeable Data," *Journal of Machine Learning Research (JMLR)*, vol. 25, no. 225, pp. 1–38, 2024.
12. **Y. Geifman and R. El-Yaniv**, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option," in *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR, vol. 97, 2019, pp. 2151–2159.
13. **M. Sensoy, L. Kaplan, and M. Kandemir**, "Evidential Deep Learning to Quantify Classification Uncertainty," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018, pp. 3179–3189.
14. **R. M. Dunn, L. Wasserman, and A. Ramdas**, "Distribution-Free Prediction Sets for Two-Layer Hierarchical Models," *Journal of the American Statistical Association (JASA)*, vol. 118, no. 544, pp. 2491–2502, 2023. DOI: [10.1080/01621459.2022.2060112](https://doi.org/10.1080/01621459.2022.2060112).

---

## Repository Sources Reviewed

The technical analysis and implementation mapping in this review were conducted directly against the CrimeVision AI repository artifacts:

- **Core Documentation:** `README.md`, `CRIMEVISION_SRS_DOCUMENTATION.md`, `RESEARCH_PROTOCOL.md`, `SECURITY_BOUNDARY.md`, and `VALIDATION_STATUS.md`.
- **Inference & Model Pipelines:**
  - `research-engine/src/inference.py`: CLIP feature extraction, cost computation, and decision routing.
  - `research-engine/src/evidential_fusion.py`: Log-domain Sinkhorn iterations, partial-transport dustbins, and Dirichlet evidential head.
  - `research-engine/src/conformal.py`: Class-conditional Gaussian densities, group-level calibration, and set-prediction mapping.
  - `research-engine/src/image_quality.py`: Laplacian variance, high-frequency residual, and patch trust generation.
- **Data Engineering & Manifests:**
  - `research-engine/src/data_utils.py` and `research-engine/datasets/adapt_uca.py`: UCA dataset adaptation.
  - `research-engine/datasets/uca_group_v3/`: Fit, Calibration, and Test splits with `uca_pairing_manifest.json`.
- **Artifacts & Reports:**
  - `research-engine/models/uca_conformal_v3.json`: Production schema-3 conformal calibration model.
  - `research-engine/reports/group_v3/uca_ablation_grouped.json`: Multi-feature ablation report.
  - `research-engine/reports/group_v3/full_vs_partial_ot_grouped.json` and `full_vs_partial_ot_grouped_bootstrap.json`: OT comparison metrics and 2,000-sample bootstrap distribution.
  - `research-engine/reports/group_v3/image_robustness_bootstrap.json`: Controlled corruption evaluation data.
- **Application & Service Architecture:**
  - `research-engine/main.py`: FastAPI endpoints and orchestration.
  - `research-engine/evidence_store.py`: SHA-256 content-addressed evidence preservation.
  - `research-engine/neo4j_connector.py`: Neo4j graph driver with volatile memory fallback.
  - `frontend-web/`: Next.js 13 / React 18 dashboard and visual components.

> [!NOTE]
> Third-party libraries, virtual environments (`.venv`), Git metadata, temporary test caches, and compiled binaries were excluded from analysis. Review was restricted strictly to author-written code, research documentation, and experiment manifests. Raw source video frames referenced in the CSVs were absent from the archive; hence feature extraction was audited from code and saved reports rather than recomputed from raw media.
