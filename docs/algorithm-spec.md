# Algorithm specification: spatial feature reconstruction
Date: 2026-11-12. Approved design; starting hyperparameters below must be implemented and checked in CPU preparation, then frozen before evaluation. Implementation is in the four src modules; see configs/freeze.json and docs/p2-selftest.json. Main experiments have not run.

## Shared preprocessing and cache contract
RGB, 448x448 bilinear image resize, nearest-neighbour masks, ImageNet normalization. Preserve no-crop behavior for comparability; document aspect-ratio distortion. Derive feature grids from actual tensors: remove existing hardcoded 224/16x16 assumptions in a future implementation change.
Frozen DINOv2 ViT-S/14 produces 384-dimensional tokens on 32x32 grid; L2-normalize per token as in the existing variant. PatchCore uses the existing WideResNet50-2 layer2/layer3 patch aggregation and 1024-dimensional vectors, expected 56x56 grid at 448; assert actual dimensions in smoke.
Extract once per unique image/backbone/preprocessing/checkpoint fingerprint. Persist float16 feature tensors and cast to float32 for scoring/training, consistently across methods. This is a new, disclosed precision policy; do not compare it as numerically identical to the historical 224px run. Masks/labels are NOT inputs to features or learning. Metadata and label access remain role-gated even when caches share storage.

## Conv AE
Input: 384 token channels plus fixed x/y coordinates in [-1,1].
Encoder: Conv3x3(386,128,padding=1), GroupNorm(8,128), GELU; Conv3x3(128,32,padding=1), GroupNorm(8,32), GELU.
Decoder: Conv3x3(32,128,padding=1), GroupNorm(8,128), GELU; Conv3x3(128,384,padding=1), linear output.
No skip connections, spatial downsampling, pretrained-weight updates or batch running statistics.
Denoising: independently mask 10% of feature coordinates during training; never corrupt x/y coordinates. Reconstruct original clean frozen features. Loss: mean squared reconstruction error over tokens/channels, with equal image weighting. No defect masks/labels enter the loss.
Source training: AdamW lr=1e-3, weight_decay=1e-4, up to 1,000 updates. Start batch size two full token grids; use gradient accumulation if needed without changing effective image weighting. Check source-normal validation loss every 100 updates; select minimum validation-loss checkpoint with earliest-step ties. Five independent initialization/training seeds for K=0. Record training/validation curves.
Adaptation: 100 fixed updates, AdamW lr=1e-4, weight_decay=1e-4. Freeze encoder convolutions and encoder norms. Train decoder convolutions and decoder norms. Each update contains equal numbers of target-normal and source-fit-normal grids; sample images uniformly with replacement. Source validation never enters replay. No target validation/test labels select stopping times.
Source threshold is calibrated once after K=0 checkpoint selection, using clean held-out source-normal scores; freeze it for that seed at K>0.

## Scoring
At inference use clean features. Patch score = channel-mean squared reconstruction error. Image score = mean of the highest ceil(0.01 * number_of_patches) patch scores. No test-fitted normalization. Upsample maps bilinearly to the common evaluation grid. DINO memory uses cosine nearest-neighbour distances with existing top-1%-patch image aggregation; PatchCore variant uses squared-L2 nearest-neighbour distances and maximum patch score.
Do not imply these three full pipelines differ only in their backbone. Report scoring differences.

## Adaptation ablations, K=20, seeds 0/1/2
- No replay: same source checkpoints, decoder-only updates and target exposure per update; remove source loss. Report lower total replay compute rather than doubling target exposure.
- FiLM/norm-only: adapt ONLY existing GroupNorm affine scale/bias parameters throughout AE; freeze all convolutions. GroupNorm affine adaptation is the chosen FiLM-like implementation, not an additional unidentified FiLM network. Same 1:1 replay, steps and learning rate as core.
- Per-patch MLP: concatenate x/y with each feature, 386->128->32->128->384, GELU hidden activations, no spatial mixing or skip. Independently source-train seeds 0/1/2; decoder comprises last two linear layers. Same loss, corruption, replay and adaptation budget. Report differing parameter counts; no architecture-only causal claim.

## Memory baselines and alignment
Both memory variants retain total bank budget 2048 and existing candidate/coreset policy (128 candidate patches/image, 8192 global candidates, coreset seed 12345). Source+K support fit the same-size bank; source contribution may change. Log actual bank owners and target allocation. No hidden capacity increase with K.
Mean-alignment ablation: compute per-image mean features, then average equally across authorized source-fit and K target-normal images. At K=20 shift source feature vectors by mu_target - mu_source before the otherwise unchanged source+support bank fit. Target features remain unchanged. For DINO cosine features, renormalize translated source vectors. Declare source-retention evaluation policy: apply the same deployed detector unchanged to source test. No covariance alignment or target-only refit.
Primary memory K=0 has a source-fixed threshold; retain it after support/mean alignment.

## Controlled 10+10 threshold table
For each seed split its 20 selected target normals deterministically into 10 adaptation and 10 calibration, disjoint by reviewed capture group. The core detector adapts/fits with 10 only. Evaluate identical predictions using (1) original source-fixed threshold and (2) empirical 95th-percentile threshold, method=higher, from ten held-out target-normal scores. Positive decision: score > threshold. Report calibration denominators and tie behavior. This table is separate from all-20 adaptation and does not claim guaranteed FPR.

## Risks and checks
Local execution contract: Codex performs only the P3 correctness smoke on the user's 4 GB GPU, with batch one, at most two source-normal examples and two training updates. Load one frozen backbone at a time; never lower the main 448px resolution to disguise memory failure. If it does not fit, record the blocker and stop for a revised smoke arrangement. Smoke does not establish final convergence or Kaggle throughput. The user runs the full matrix on Kaggle using docs/local-workflow.md; benchmark there before expanding. Main training batch/steps above are unchanged.
Frozen feature reconstruction discourages constant outputs but does not prevent defect reconstruction, identity-like behavior, oversmoothing, adaptation overfit or source forgetting. Check finite features/loss/gradients, frozen parameter hashes, changing authorized trainable weights, nonconstant outputs, shapes and masks before execution. Test using synthetic tensors/normal validation only before scoring held-out defects.
Checkpoint metadata must include protocol/source/code/feature fingerprint, support IDs, seed, trainable parameter count and update count. GPU timing must synchronize and distinguish extraction, fitting and scoring; report peak allocated/reserved memory and full session time.
