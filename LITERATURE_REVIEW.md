Literature Review – CrimeVision AI
Project: CrimeVision AI – Multimodal Evidence Comparison and Investigation Decision-Support Prototype
Document type: Literature Review
Review basis: Original research literature, the uploaded CrimeVision AI repository, its research protocol, source code, experiment manifests and saved evaluation reports
Prepared: 18 November 2026
Abstract
CrimeVision AI investigates whether visual information in an image or a representative frame sampled from a surveillance video is compatible with an associated natural-language description. Its research pipeline combines a frozen Contrastive Language–Image Pre-training (CLIP) model, patch-to-token feature comparison, entropic Optimal Transport (OT), a quality-weighted Partial OT variant and split-conformal prediction. The system is intended to provide a decision-support signal with three possible outcomes: MATCH, CONTRADICTION and INDETERMINATE.

This review examines the literature relevant to the methods used by the project. It covers surveillance-video anomaly datasets, Vision Transformers, vision-language pretraining, image–text entailment, Optimal Transport, Sinkhorn divergences, Partial and unbalanced OT, image-quality measurements, conformal prediction, selective prediction and evidential deep learning. It then relates those approaches to the implementation in the supplied repository and discusses the strengths and limitations of the reported experiments.

The review finds that the project brings together established methods in a particular engineering pipeline, rather than proposing a new CLIP architecture, a new Optimal Transport theorem or a new conformal-prediction guarantee. Its active decision path uses a three-feature, recording-group-calibrated split-conformal predictor. The evidential neural head is present as an alternative implementation but is not selected by the supplied runtime configuration. The dataset labels are synthetic cross-recording, cross-category description mismatches and are not human-adjudicated forensic contradictions. Consequently, the current results should be interpreted as exploratory measurements on a proxy dataset, not evidence of forensic accuracy or real-world CCTV generalisation.

Keywords: CLIP, vision-language learning, image–text compatibility, Optimal Transport, Sinkhorn divergence, Partial Optimal Transport, conformal prediction, uncertainty quantification, selective prediction, surveillance video.
1. Introduction
Surveillance systems generate large amounts of visual material, while investigation workflows may also contain written descriptions, statements and other textual information. Comparing a textual account with visual evidence is challenging because an image contains many visual elements, only some of which may be relevant to a statement. The relevant details may be small, partly occluded, blurred, poorly illuminated or absent from a particular frame. A useful decision-support system must therefore do more than calculate one global image–text similarity value. It should also communicate uncertainty and avoid forcing a binary decision when the available evidence does not support one.

CrimeVision AI studies this problem as image–description compatibility assessment. Given an image and a supplied statement, the research engine extracts global image and text representations, compares local visual patch and text-token representations, calculates Optimal Transport diagnostics, and uses a calibrated prediction-set mechanism to decide whether to return a single label or abstain. Its outputs are MATCH, CONTRADICTION or INDETERMINATE.

These labels must be interpreted carefully. In the supplied research dataset, MATCH and CONTRADICTION are proxy labels created by a pairing procedure. They are not human-adjudicated conclusions about a genuine witness statement. The intended role of the application is therefore research and decision support. It does not identify a person, verify a suspect's identity, establish what physically occurred, or provide a legal or forensic conclusion.

The literature review is organised around the methodological components that underpin this pipeline and the differences between the research tasks in the cited literature and the task implemented by CrimeVision AI.
2. Surveillance-Video Analysis and the UCF-Crime Dataset
2.1 Surveillance anomaly detection
A major research direction in surveillance-video analysis is automated detection of anomalous events. Sultani, Chen and Shah introduced the UCF-Crime dataset and a weakly supervised anomaly-detection method based on multiple-instance learning and ranking. Their dataset contains 1,900 long, untrimmed surveillance videos, covering 13 anomaly categories and normal activities. The method treats videos as bags and video segments as instances, learning to assign higher anomaly scores to anomalous segments from video-level labels [1].

This work is relevant to CrimeVision AI because it provides the surveillance-video domain from which the project's image material is derived. However, the original UCF-Crime task is not the same as image–text contradiction assessment. Its central objective is to rank or localise anomalous video segments, whereas CrimeVision AI compares a visual frame with a textual description. A dataset designed for anomaly detection does not automatically provide human-verified image–statement entailment or contradiction labels.
2.2 How the project adapts the dataset
The repository's UCA adapter uses UCF-Crime recordings together with a companion JSON annotation file named UCFCrime.json. It constructs image–description pairs as follows:
- MATCH: an image frame is paired with the annotation description associated with its own source recording.
- CONTRADICTION: an image frame is paired with a description from a different source recording and a different activity category, selected from the same dataset partition.
This creates a controlled, synthetic cross-category description-mismatch dataset. It is useful for testing whether feature-based methods distinguish some matched and mismatched pairs. Nevertheless, a different activity category does not prove that every pair is semantically contradictory. Some descriptions may be broad, incomplete or compatible with more than one visual situation. The negative label is therefore a proxy mismatch label, not a human-verified contradiction label.

The updated adapter records both source_recording_id and statement_source_recording_id, allowing the evidence-source and description-source recording provenance to be audited. The supplied uca_group_v3 manifest reports 3,136 fit pairs, 140 calibration pairs and 140 test pairs. The class distribution is balanced within each partition: 1,568 MATCH and 1,568 CONTRADICTION fit pairs, and 70 examples of each class in both calibration and test partitions. Evidence-source recording IDs and statement-source recording IDs are reported as disjoint across the three partitions.

There are still important limits to the split. The camera_id values are aliases derived from recording IDs, not verified physical camera identifiers. The environment_id values represent activity categories and overlap between partitions. Therefore, the split supports a recording-disjoint claim, but not an unseen-physical-camera or unseen-activity-category generalisation claim. The original UCF-Crime research [1] and the project adapter serve different purposes; this distinction should remain explicit in reports and presentations.
3. Vision Transformers and Vision–Language Representation Learning
3.1 Vision Transformer representations
Dosovitskiy et al. demonstrated that an image can be represented as a sequence of patches and processed by a Transformer, providing the basis for the Vision Transformer (ViT) architecture [2]. Instead of relying only on convolutional feature maps, a ViT divides an image into fixed-size patches, embeds them and models relationships between the resulting tokens.

This representation is relevant to CrimeVision AI because the project uses CLIP with a ViT-B/32 visual backbone. At the configured 224 × 224 input size and 32 × 32 patch size, the visual patch grid is 7 × 7, producing 49 visual patch features after the classification token is excluded. Text is represented by token-level features. These sequences provide more detailed comparison inputs than a single pooled vector.
3.2 CLIP and contrastive image–text learning
Radford et al. introduced CLIP, a vision-language model trained on a large collection of image–text pairs using a contrastive objective [3]. The image encoder and text encoder map their inputs into a shared representation space. Training encourages the representation of a matched image and caption to be more similar than representations of mismatched examples in the same training batch. The resulting model can transfer to a range of image-recognition tasks through natural-language prompts without requiring a separately trained classifier for every category [3].

CrimeVision AI uses the pretrained openai/clip-vit-base-patch32 model through Hugging Face Transformers. The model is frozen during inference; the project does not train a new large-scale vision-language foundation model. It extracts:
- a global image embedding;
- a global text embedding;
- local visual patch features;
- local text-token features.
The global embeddings are L2-normalised and compared using cosine similarity. This gives a compact measure of overall image–text compatibility. The local patch and token features are used to construct a more detailed cross-modal cost matrix.

CLIP is a useful feature backbone, but its global similarity score is not a calibrated probability that a statement is true. High similarity does not prove that all attributes in a statement are visible, and low similarity does not by itself prove contradiction. The project accordingly uses global similarity as one feature in a broader pipeline rather than as a standalone forensic decision threshold.
3.3 Global matching versus fine-grained visual entailment
Image–text matching and visual entailment are related but distinct tasks. In visual entailment, a model evaluates whether a sentence is entailed by, contradicts or is neutral with respect to an image. Xie et al. introduced the SNLI-VE dataset and formulated visual entailment as a three-way image–sentence reasoning problem [4]. This task is conceptually closer to determining whether a statement is supported by an image than generic image–text retrieval.

The comparison is useful for positioning CrimeVision AI. CLIP provides a general shared embedding space, while visual-entailment research addresses finer semantic relations between an image and a sentence. CrimeVision AI does not currently implement a supervised SNLI-VE-style entailment model. Its current UCA-derived labels are generated from recording and activity-category pairing rules. The repository also identifies a vetted SNLI-VE adapter and its neutral-label treatment as future work rather than part of the current baseline.

A future dataset with human-annotated support, contradiction and insufficient-evidence labels would be more directly aligned with the intended investigative use. Such a dataset should include annotation guidelines, multiple independent annotators, disagreement records and clear rules for cases where a statement contains several attributes but only some are observable.
4. Optimal Transport for Local Cross-Modal Alignment
4.1 Optimal Transport background
Optimal Transport compares distributions by finding a transport plan that moves mass from one distribution to another while minimising a cost. In image–text comparison, visual patch features and text-token features can be treated as two sets of feature vectors. A cost matrix can then represent the dissimilarity between each visual patch and each text token.

This is attractive for multi-part inputs because it does not require every feature to be represented by a single pooled vector. The transport plan can distribute mass across multiple patch–token pairs, providing a structured alignment diagnostic. However, a transport plan over learned feature vectors is not automatically a semantic explanation of which physical object appears in a patch.
4.2 Entropic regularisation and the Sinkhorn algorithm
Classical Optimal Transport can require solving a linear-programming problem. Cuturi proposed adding entropic regularisation, making the transport problem amenable to efficient matrix-scaling iterations known as the Sinkhorn algorithm [5]. Entropic regularisation smooths the transport plan and makes computation more practical for machine-learning applications.

For source and target marginals \(a\) and \(b\), and cost matrix \(C\), a common entropic OT formulation is:
\[
\operatorname{OT}_{\varepsilon}(a,b;C)=
\min_{P\geq 0,\;P\mathbf{1}=a,\;P^\top\mathbf{1}=b}
\left[\langle P,C\rangle+\varepsilon\,\mathrm{KL}(P\,\|\,ab^\top)\right].
\]
Here, \(P\) is the transport plan and \(\varepsilon\) controls the strength of entropic regularisation. The CrimeVision AI implementation uses a log-domain Sinkhorn solver, checks marginal errors and treats non-convergence as an error rather than silently accepting an invalid result.
4.3 Debiased Sinkhorn divergence
Entropic regularisation can introduce a bias into the regularised OT cost. Feydy et al. studied Sinkhorn divergences, which combine cross-distribution and self-distribution transport terms to obtain a debiased comparison [6]. The project implements the form:
\[
S_{\varepsilon}(V,T)=\operatorname{OT}_{\varepsilon}(V,T)
-\frac{1}{2}\operatorname{OT}_{\varepsilon}(V,V)
-\frac{1}{2}\operatorname{OT}_{\varepsilon}(T,T),
\]
where \(V\) denotes visual patch features and \(T\) denotes text-token features. In the application, this divergence is an alignment diagnostic and one of the features used by the conformal predictor. It is not a universal contradiction threshold. A particular Sinkhorn value cannot, by itself, establish whether a witness statement is true or false.
4.4 Partial and unbalanced Optimal Transport
Full OT generally enforces transport of the complete prescribed mass. Partial OT formulations allow only a specified fraction of mass to be transported, which is useful when some parts of one distribution have no meaningful counterpart in the other [7]. Unbalanced OT relaxes marginal constraints and provides another family of methods for handling unequal or uncertain mass [8]. These ideas motivate approaches that avoid forcing every visual patch to align with a text token.

CrimeVision AI implements a dustbin-based partial transport mechanism. The transport problem is augmented with dustbin row and column entries so that some visual or textual mass can remain unmatched. The solver reports transported mass and discarded mass, and the project compares this configuration with Full OT in its experiments. The implementation is related to the broader Partial and Unbalanced OT literature, but the project's precise dustbin construction and feature weighting should be understood from its own code rather than assumed to be identical to every formulation in the cited papers.
4.5 Image-quality-based trust weighting
The project uses image-quality measurements to derive patch-level visual trust. Its quality module calculates descriptive indicators including high-frequency residual, Laplacian variance and luminance variation. Patch sharpness is estimated from local Laplacian variance on the 7 × 7 visual patch grid.

In the supplied runtime configuration, TRUST_MODE=heterogeneity. The visual trust for each patch is based on its sharpness relative to the sharpest patch in the same image. Text-token trust is uniform. The resulting trust values influence the transport marginals and the amount of mass allowed to be transported. An alternative absolute sigmoid trust mode is implemented, but it is not the configured default in the supplied environment.

This approach provides a practical heuristic for testing whether image-quality variation can inform partial alignment. It should not be described as a learned or independently calibrated estimate of semantic importance. Relative, within-image sharpness is also not equivalent to absolute image quality: if all patches are degraded similarly, normalising by the maximum patch sharpness may not reflect the absolute loss of image detail. The project’s degradation experiments are therefore important, but they do not establish robustness to all real surveillance conditions.
4.6 Interpretation of the heatmap
The frontend heatmap displays the patch-to-token cost matrix computed from the CLIP representations. It is an exploratory visualisation of feature dissimilarity. It is not a supervised phrase-grounding model, an object detector, an attribute-localisation system or proof that a particular word refers to a particular region of the image. The distinction between a feature-alignment diagnostic and an explainable semantic localisation result is important when presenting the system to evaluators or investigative-domain reviewers.
5. Uncertainty Quantification and Selective Prediction
5.1 Split-conformal prediction
Conformal prediction is a framework for constructing prediction sets with finite-sample coverage properties under an exchangeability assumption [9, 10]. Rather than returning only one class, a conformal classifier can return a set of labels that remain plausible at a chosen significance level. A prediction set containing one class supports a single-label output; a set containing both classes or no classes can be treated as an abstention condition, depending on the system design.

A standard split-conformal workflow separates data into a fitting partition and a calibration partition. A model or scoring function is fitted using the fitting data. Nonconformity scores are then calculated on the separate calibration examples, and a finite-sample corrected quantile is computed. At inference, a candidate label is included when its nonconformity score is below the calibrated threshold.

The finite-sample coverage statement is conditional on the relevant exchangeability assumption. It does not establish that a particular prediction is correct, that accepted predictions have a specified accuracy, or that the model is robust to distribution shift. Work on conformal prediction under non-exchangeable data highlights why the exchangeability assumption needs careful consideration in applications with temporal, spatial or grouped structure [11].
5.2 CrimeVision AI's active conformal mechanism
The supplied configuration selects the schema-3 conformal artifact, uca_conformal_v3.json. Its nonconformity features are:
1. Sinkhorn divergence;
2. discarded transport mass;
3. one minus global CLIP similarity.
The fitting partition is used to fit class-conditional Gaussian density models for the MATCH and CONTRADICTION feature distributions. The calibration examples are scored using these fitted densities. For each source recording, the maximum nonconformity score across the examples in that recording is used as the recording-level calibration score. The finite-sample conformal quantile is then computed across the calibration recording groups.

The active artifact uses \(\alpha=0.10\), corresponding to a nominal coverage target of 90%, and records \(\hat q=0.6086335403\). It contains 3,136 fit pairs and 140 calibration pairs, with 70 calibration recording groups. The runtime prediction set is mapped as follows:
- {MATCH} → MATCH;
- {CONTRADICTION} → CONTRADICTION;
- both labels or an empty set → INDETERMINATE.
The group-maximum calibration design targets simultaneous inclusion of the labels for all evaluated pairs within a future recording group, conditional on exchangeability of calibration and future recording groups and on using the same group-construction procedure. This is not the same as group-conditional coverage across predefined populations (for example, guaranteeing a separate coverage level for each camera type or activity category). Research on hierarchical data studies the statistical consequences of observations nested within groups and develops alternative group-aware conformal procedures [14]. The project's maximum-score procedure should therefore be understood according to its own declared recording-group coverage event. It does not provide class-conditional coverage, a per-example correctness guarantee, a guaranteed abstention rate or protection against distribution shift. These boundaries are explicitly documented in the project's conformal implementation and research protocol.
5.3 Selective prediction and abstention
Selective prediction systems can abstain on examples where the model is insufficiently confident. Geifman and El-Yaniv's SelectiveNet, for example, jointly trains a predictor and a selection mechanism to optimise a risk–coverage trade-off [12]. CrimeVision AI uses a different approach: the set-valued conformal predictor abstains when its prediction set is not a singleton.

This difference matters. The project is not implementing SelectiveNet, and its reported selective accuracy is not a conformal coverage guarantee. Coverage, singleton rate, selective accuracy, abstention rate and prediction-set size describe different aspects of the system and should be reported together. A high coverage result can coexist with a high abstention rate because returning both classes preserves coverage while declining to make a single-class decision.
6. Evidential Deep Learning as an Alternative Research Direction
Sensoy, Kaplan and Kandemir proposed evidential deep learning for classification uncertainty by placing a Dirichlet distribution over class probabilities and learning evidence values from data [13]. In this approach, a neural network produces non-negative evidence, which is converted into Dirichlet concentration parameters. The resulting concentration controls both class probabilities and an uncertainty measure.

CrimeVision AI contains an alternative evidential pair-head implementation, training script and supporting tests. The head is designed to consume pair-level CLIP/OT features and image-quality features and to produce Dirichlet-based estimates for the two classes. It can be paired with a separate calibration mechanism and abstention checks.

However, the supplied runtime environment leaves the evidential-head and evidential-calibration paths unset and loads the schema-3 conformal artifact instead. Therefore, the evidential neural head is not the active decision mechanism in the supplied configuration. It should be described as an implemented alternative research component, not as the model currently generating the application's MATCH, CONTRADICTION and INDETERMINATE results. The distinction also prevents the project from implying that the active conformal predictor combines evidential uncertainty with conformal scores; it does not.
7. Synthesis: How the Literature Informs CrimeVision AI
The cited research contributes different pieces to the project's design. CLIP provides a general image–text representation; ViT supplies a patch-based visual sequence; Optimal Transport compares sets of feature vectors; Partial OT motivates leaving unmatched mass; conformal prediction provides calibrated set-valued outputs under exchangeability; and evidential learning informs an alternative uncertainty-aware classifier. The project integrates these established ideas into a prototype for image–description compatibility analysis.
Literature area	Representative work	Relevance to CrimeVision AI	Important distinction
Surveillance anomaly detection	Sultani et al. [1]	Provides the source surveillance-video domain and activity categories used by the dataset adaptation.	The original task is video anomaly detection, not witness-statement entailment.
Vision Transformers	Dosovitskiy et al. [2]	Provides the patch-sequence visual representation used by the ViT-based CLIP backbone.	Patch tokens are feature units, not object detections.
Vision-language pretraining	Radford et al. [3]	Supplies frozen global and local image–text representations.	Global similarity is not a calibrated truth score or forensic verdict.
Visual entailment	Xie et al. [4]	Provides a closer research framing for image–sentence support, contradiction and neutrality.	The project does not currently train a supervised visual-entailment model.
Entropic OT	Cuturi [5]	Motivates efficient Sinkhorn computation for patch/token distributions.	OT cost is an alignment diagnostic, not a contradiction threshold.
Sinkhorn divergence	Feydy et al. [6]	Motivates the debiased cross-/self-transport comparison.	The project uses a particular implementation and feature pipeline.
Partial and unbalanced OT	Figalli [7]; Chizat et al. [8]	Provides theoretical context for transporting only part of the mass or relaxing marginal constraints.	The project's dustbin and trust-weighting design is a specific implementation, not a new OT theory.
Conformal prediction	Vovk et al. [9]; Angelopoulos and Bates [10]	Provides the set-valued uncertainty mechanism and finite-sample calibration framework.	Coverage depends on exchangeability at the declared calibration unit.
Non-exchangeable settings	Oliveira et al. [11]	Highlights the challenge of applying conformal guarantees when observations are dependent or structured.	Recording-disjoint splits alone do not prove exchangeability of all future recording groups.
Selective prediction	Geifman and El-Yaniv [12]	Provides context for abstention and the risk–coverage trade-off.	CrimeVision AI uses conformal prediction sets, not SelectiveNet.
Evidential learning	Sensoy et al. [13]	Provides the methodological basis for the alternative Dirichlet evidential head.	This head is not activated by the supplied runtime configuration.
Hierarchical conformal prediction	Dunn et al. [14]	Provides related work on prediction when examples are nested within groups.	The project uses its own recording-level maximum-score calibration rule, not the specific procedures proposed in this paper.


8. Experimental Findings Available in the Repository
This section summarises saved reports included in the archive. These are repository-reported experimental results, not results independently reproduced during preparation of this literature review. The CSV files reference image paths that are not included in the ZIP, so the original CLIP feature extraction and end-to-end evaluation cannot be rerun from the supplied archive alone.
8.1 Three-feature conformal configuration
The active conformal artifact uses all three features: Sinkhorn divergence, discarded mass and one minus global CLIP similarity. Its saved grouped-v3 ablation report records the following test metrics at \(\alpha=0.10\):
Metric	Saved result
Test image–text pairs	140
Pair-level empirical coverage	93.57%
Singleton prediction rate	30.00%
Selective accuracy	78.57%
Abstention rate	70.00%
Average prediction-set size	1.70
Recording-group simultaneous coverage	61 / 70 groups (87.14%)


These metrics must be interpreted together. The system abstains on 70% of test pairs, and selective accuracy is calculated only on singleton predictions. Pair-level empirical coverage and recording-group simultaneous coverage describe different events and must not be interchanged. The nominal conformal target is 90%; the observed 87.14% group coverage is the descriptive result in this finite test sample, not a guarantee that every future sample will achieve exactly 90%.
8.2 Full OT versus Partial OT
The saved Full OT versus Partial OT experiment uses the two-feature sinkhorn_plus_clip configuration, not the active three-feature artifact. In that comparison, the report records:
Metric	Full OT	Partial OT
Pair-level empirical coverage	94.29%	93.57%
Singleton prediction rate	32.86%	27.14%
Selective accuracy	82.61%	76.32%
Abstention rate	67.14%	72.86%
Recording-group simultaneous coverage	62 / 70 (88.57%)	61 / 70 (87.14%)


The paired recording-level bootstrap report uses 2,000 replicates and seed 42. For the Full OT minus Partial OT comparison, the observed selective-accuracy difference is approximately 6.29 percentage points, with a 95% percentile interval of approximately [2.02, 11.67] percentage points. The comparison therefore does not support a claim that Partial OT outperformed Full OT on these reported metrics. Partial OT remains a method investigated by the project, but its intended ability to handle unmatched mass should not be confused with demonstrated superiority on this experiment.
8.3 Image-robustness experiments
The repository includes controlled image-corruption experiments involving Gaussian blur, brightness changes, Gaussian noise and JPEG compression. The reported design fits and calibrates on clean data, then reuses the clean-data threshold for clean and distorted test images without recalibrating on corrupted test data. Paired bootstrap intervals resample complete source-recording groups.

This is a useful stress-test design because it examines changes in prediction behaviour under controlled perturbations while keeping the threshold fixed. It is not equivalent to validation on naturally degraded CCTV footage. Synthetic corruptions do not capture every source of surveillance degradation, such as motion blur, compression pipelines, camera sensors, lighting transitions, occlusion, weather, perspective or scene-specific effects.
8.4 Evaluation status and test-set reuse
The project's RESEARCH_PROTOCOL.md notes that the held-out test split has been used in earlier ablation work. The saved group-v3 metrics should consequently be described as retrospective or exploratory rather than as results from a completely untouched confirmatory test set. A final confirmatory evaluation should use a newly reserved set of recordings, with all modelling, feature selection and threshold choices fixed before that final evaluation.
9. Research Gap and Project Positioning
The literature establishes strong foundations for image–text representation learning, optimal transport, uncertainty quantification and surveillance-video analysis. However, these areas do not automatically solve the specific problem of deciding whether a witness statement is supported or contradicted by a particular CCTV frame. Several gaps remain between general-purpose methods and a reliable investigative application:
1. Task-specific labels: UCF-Crime is an anomaly-detection dataset, not a collection of human-verified witness statements paired with CCTV frames. A task-specific dataset with expert-adjudicated support, contradiction and insufficient-evidence labels is needed.
2. Semantic grounding: CLIP patch/token dissimilarity is not supervised phrase grounding. More direct object, attribute and relation verification would be required to explain which visual facts support or conflict with a statement.
3. Temporal reasoning: The current video workflow samples a limited number of frames and selects one sharpness-based representative frame. It does not model event duration, action sequence, temporal order or interactions across frames.
4. Calibration under domain shift: Recording-group conformal calibration depends on exchangeability of calibration and future recording groups. New cameras, locations, event categories or image-processing pipelines may violate that assumption.
5. Generalisation evidence: The dataset's recording identifiers are separated, but physical camera identities are not verified and activity categories overlap. Camera-disjoint and location-disjoint evaluations remain necessary.
6. Independent testing: The available test set has been used in prior ablations. A new locked test set is required for confirmatory claims.
7. Evidence governance: SHA-256 file integrity checks do not by themselves provide signed acquisition records, trusted timestamps, immutable storage, operator attribution or a complete chain-of-custody process.
Within these boundaries, the defensible contribution of CrimeVision AI is the engineering and empirical examination of a pipeline combining frozen CLIP representations, OT alignment diagnostics, a quality-weighted Partial OT variant and recording-group split-conformal abstention. The project can report what this pipeline does on the constructed proxy task, compare Full OT and Partial OT, and investigate controlled image degradation. It should not claim a novel OT theorem, forensic-grade contradiction detection, suspect identification, legal admissibility or verified real-world accuracy.
10. Conclusion
The literature supports the methodological choices used in CrimeVision AI, while also highlighting the need for cautious interpretation. CLIP offers transferable global image–text features; ViT exposes visual patch representations; Sinkhorn methods make entropic OT computationally practical; Partial OT provides a principled motivation for leaving some mass unmatched; and split-conformal prediction supplies a set-valued mechanism for uncertainty-aware abstention under exchangeability assumptions. Evidential deep learning provides an alternative uncertainty-modelling approach that exists in the repository but is not active in the supplied deployment configuration.

The current implementation is best understood as a research prototype for selective image–description compatibility analysis. Its most important limitations concern the proxy nature of the labels, absence of the original image assets from the archive, prior reuse of the test split, lack of physical-camera-disjoint evaluation and the gap between patch/token feature alignment and human-interpretable semantic grounding. Future research should prioritise human-adjudicated CCTV–statement datasets, independent recording and camera splits, temporal video reasoning, grounded visual verification, calibration under distribution shift and stronger evidence-governance mechanisms.
References
[1] W. Sultani, C. Chen, and M. Shah, “Real-World Anomaly Detection in Surveillance Videos,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6479–6488. Paper (CVF Open Access). UCF project page. DOI: 10.1109/CVPR.2018.00678.

[2] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” International Conference on Learning Representations (ICLR), 2021. Google Research publication page. arXiv:2010.11929.

[3] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” Proceedings of the 38th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763, 2021. PMLR paper. arXiv:2103.00020.

[4] N. Xie, F. Lai, D. Doran, and A. Kadav, “Visual Entailment: A Novel Task for Fine-Grained Image Understanding,” 2019. arXiv:1901.06706. Institutional record.

[5] M. Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport,” Advances in Neural Information Processing Systems, vol. 26, 2013. NeurIPS paper.

[6] J. Feydy, T. Séjourné, F.-X. Vialard, S. Amari, A. Trouvé, and G. Peyré, “Interpolating between Optimal Transport and MMD using Sinkhorn Divergences,” Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, vol. 89, pp. 2681–2690, 2019. PMLR paper.

[7] A. Figalli, “The Optimal Partial Transport Problem,” Archive for Rational Mechanics and Analysis, vol. 195, no. 2, pp. 533–560, 2010. DOI: 10.1007/s00205-008-0212-7.

[8] L. Chizat, G. Peyré, B. Schmitzer, and F.-X. Vialard, “Scaling Algorithms for Unbalanced Optimal Transport Problems,” Mathematics of Computation, vol. 87, no. 314, pp. 2563–2609, 2018. DOI: 10.1090/mcom/3303.

[9] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. Springer, 2005. DOI: 10.1007/b106715.

[10] A. N. Angelopoulos and S. Bates, “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,” Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023. DOI: 10.1561/2200000101. Paper record.

[11] R. I. Oliveira, P. Orenstein, T. Ramos, and J. V. Romano, “Split Conformal Prediction and Non-Exchangeable Data,” Journal of Machine Learning Research, vol. 25, no. 225, pp. 1–38, 2024. JMLR paper.

[12] Y. Geifman and R. El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option,” Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 97, pp. 2151–2159, 2019. PMLR paper.

[13] M. Sensoy, L. Kaplan, and M. Kandemir, “Evidential Deep Learning to Quantify Classification Uncertainty,” Advances in Neural Information Processing Systems, vol. 31, 2018. NeurIPS paper.

[14] R. M. Dunn, L. Wasserman, and A. Ramdas, “Distribution-Free Prediction Sets for Two-Layer Hierarchical Models,” Journal of the American Statistical Association, vol. 118, no. 544, pp. 2491–2502, 2023. DOI: 10.1080/01621459.2022.2060112. Author publication page.
Repository Sources Reviewed
The project-specific descriptions and implementation mapping in this review are based on the supplied CrimeVision AI archive, including the following principal files and artifacts:
- README.md – current implementation summary, architecture and setup.
- CRIMEVISION_SRS_DOCUMENTATION.md – implementation-aligned requirements and system description.
- RESEARCH_PROTOCOL.md – research question, label semantics, split assumptions, claim boundaries and reproduction instructions.
- SECURITY_BOUNDARY.md and VALIDATION_STATUS.md – deployment limitations and validation status.
- research-engine/src/inference.py – CLIP feature extraction, patch/token costs, OT configuration and decision-path selection.
- research-engine/src/evidential_fusion.py – log-domain Sinkhorn, dustbin partial transport and alternative evidential head.
- research-engine/src/conformal.py – Gaussian class-density scoring, conformal quantile, prediction sets and assessment mapping.
- research-engine/src/image_quality.py – image-quality and patch-sharpness measurements.
- research-engine/src/data_utils.py and research-engine/datasets/adapt_uca.py – dataset parsing, pair construction and split validation.
- research-engine/datasets/uca_group_v3/{fit,calibration,test}.csv and uca_pairing_manifest.json – grouped dataset metadata and provenance.
- research-engine/models/uca_conformal_v3.json – active schema-3 conformal artifact and feature/configuration hashes.
- research-engine/reports/group_v3/uca_ablation_grouped.json – feature-ablation results.
- research-engine/reports/group_v3/full_vs_partial_ot_grouped.json and full_vs_partial_ot_grouped_bootstrap.json – OT comparison and paired recording-level bootstrap results.
- research-engine/reports/group_v3/image_robustness_bootstrap.json – controlled image-corruption experiment design and results.
- research-engine/main.py, research-engine/evidence_store.py, research-engine/neo4j_connector.py and docker-compose.yml – API, evidence persistence, graph integration and deployment architecture.
- frontend-web/ – investigator dashboard, upload interface, heatmap, graph and analysis-result components.
Archive inspection note: The archive contains extensive third-party dependencies, Git internals, virtual-environment files, generated build outputs, model/cache artifacts and binary assets. These were not treated as authored literature or project logic. The review focused on the project-authored source, configuration, documentation, tests and research artifacts. The archive does not include the original UCA/UCF-Crime image files referenced by the grouped CSV paths or the original companion annotation JSON, so the saved feature-level experiments could not be independently rerun from this ZIP.