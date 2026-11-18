# Quality-Aware Partial Optimal Transport (Novelty Direction 1)

Full entropic OT assumes every visual patch and text token is trustworthy.
Forensic CCTV violates this: blur, noise, occlusion and irrelevant background
patches should not be forced into a transport plan.

## Mechanism

1. `compute_patch_sharpness` returns Laplacian variance per CLIP ViT-B/32
   patch cell (7×7 = 49 patches on a 224×224 canvas).
2. Trust is `sharpness / max(sharpness)`, clamped to `[1e-6, 1]`. A uniform
   image (max sharpness ≈ 0) falls back to uniform trust.
3. Non-uniform marginals: `a_i = trust_i / sum(trust)`, `b_j` similarly.
4. Transported mass `m = min(mean(visual_trust), mean(text_trust)) ∈ (0, 1]`.
5. Partial OT with a dustbin: augmented cost `(N+1)×(M+1)` with dustbin
   entries at cost `λ` and a zero dustbin–dustbin cell. Source and target
   marginals are scaled by `m`; `(1−m)` mass is discarded on each side.
6. Debiased divergence uses cross, visual-self and text-self partial solves
   at the same `m`, so the comparison is at a fixed transported-mass level.
7. `discarded_mass = 1 − m` is appended to the 1025-d pair feature vector,
   producing 1026 pair features (1029 with quality features) for the
   evidential head.

## Why this is a defensible contribution

- It is a targeted mechanism for **forensically degraded evidence**, not a
  generic CLIP+OT+EDL combination.
- It exposes a **quality-derived abstention signal** (discarded mass) that is
  separate from the Dirichlet uncertainty and from image-quality metrics.
- It is falsifiable: report `discarded_mass` vs. error rate on the held-out
  test split, and compare against full OT (trust = 1) as an ablation.

## Required ablation in the report

Train and calibrate two models on the same splits:
- **Full OT**: force `visual_trust = text_trust = 1`.
- **Partial OT**: use the sharpness-derived trust.

Report balanced accuracy, coverage, and discarded-mass distributions. If
partial OT does not improve over full OT on your proxy, report that honestly;
a negative result with a clean ablation is still a valid capstone finding.