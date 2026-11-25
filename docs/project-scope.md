# Approved project scope
Approved: 2026-11-12. Status 2026-11-13: P0/P1 complete; MetalParts + PCB verified on CPU. PCB NO-GO under the conservative capture audit; see docs/data-audit.md. P2 implementation is tracked by configs/freeze.json and docs/p2-selftest.json; no main-study experiment is authorized in this step.

## Objective
Evaluate whether a small spatial convolutional autoencoder over frozen DINOv2 features can adapt to real imaging shifts using verified-normal images, compared with frozen DINOv2 memory and budgeted PatchCore variants. Primary metric: image AUROC. Working title: Spatial Feature Reconstruction for Normal-Only Few-Shot Adaptation under Industrial Image Shift.

## Authority and historical evidence
This scope supersedes the next-work sections in project-plan.md; protocol.md governs evaluation and docs/algorithm-spec.md governs the proposed implementation. Historical 224px results and recorded configs remain immutable. They are development evidence, not results of the proposed 448px method. Existing implementations are variants, not official author reproductions.
No syllabus/rubric was found during the permitted text audit. No original PROMPT.md was found; the new root prompt reconstructs approved chat rules, not an exact copy of an unknown document.

## Approved matrix
| Experiment | Coverage | K | Seeds |
|---|---|---|---|
| Conv AE, decoder-only + source replay | MetalParts test1-test6 | 0,5,20 | 0,1,2,3,4 |
| Frozen DINOv2 memory variant | Same | 0,5,20 | Same paired support draws |
| Budgeted PatchCore variant | Same | 0,5,20 | Same paired support draws |
| No replay; FiLM/norm-only; per-patch MLP | Same, separately controlled ablations | 20 | 0,1,2 |
| Mean alignment for each memory variant | Same | 20 | 0,1,2 |
| Controlled threshold comparison, three core methods | Same: 10 adaptation + 10 disjoint calibration | total 20 | 0,1,2 |
| PCB confirmation, conditional | All CPU-audited feasible documented target domains; declare exclusions before inference | 0,5,20 | 0,1,2 |

Conv AE K=0 means FIVE independent source-training initializations per condition; do not reuse one trained checkpoint as five runs. Each K>0 starts independently from the matching K=0 checkpoint, not sequentially from K=5. Memory K=0 is deterministic with fixed coreset seed and is not artificially replicated as independent evidence.

## Hypotheses and acceptable outcomes
H1: Spatial reconstruction improves mean target image AUROC relative to the DINOv2 memory variant at K=20. This is the single planned inferential comparison, not a promised win.
H2: Replay reduces loss of source performance relative to decoder adaptation without replay.
H3: Spatial context contributes relative to a per-patch MLP; parameter counts differ and must be reported.
H4: Threshold recalibration changes false alarms/recall under score shift; ten calibration images give imprecise estimates.
Own method does not beat the baselines is an acceptable outcome. Report negative results, uncertainty, failures and cost. Never retune using PCB outcomes.

## Scope exclusions
No official PatchCore 512px, covariance alignment, target-only refit, extra backbone sizes, trainable backbone, McNemar, Holm, vision-language methods or PiledBags.
No claim of a novel reconstruction principle, state of the art, causal isolation of backbone effects, guaranteed calibration, physical-object independence, or guaranteed publication. Any manuscript novelty needs a separate literature assessment.

## Three-day budget
25 working hours; 12 total GPU hours including local smoke and GPU idle/initialization; at most two Kaggle GPU sessions. Codex works locally in Desktop\DS; the user runs Kaggle cells. CPU-only preparation does all downloads, dependency setup, PCB audit, manifests and protocol freezing. See docs/local-workflow.md for ownership, pasteable handoff cells and estimates. No remote is created; commit locally after every completed phase.
| Day | Work hours | CPU work | GPU allocation (estimate, concurrent where possible) |
|---|---:|---|---|
| 1 | 9 | Scripts/checks, downloads later, MetalParts contract, PCB audit and recorded go/no-go, CPU freeze | Local 4 GB smoke: <=0.25 h; user Kaggle window 1 timing/extraction: 1.25 h |
| 2 | 9 | Output validation, analysis preparation | Continue session 1: extraction, core, affordable ablations: 6.5 h |
| 3 | 7 | Confirmation freeze in CPU, statistics/report/reproducibility | Session 2 PCB if feasible: 2 h |
| Reserve | Included within total GPU cap | Recovery margin, not extra working hours | 2 h |
GPU allocations are estimates, not benchmarks. Keep session 1 continuous only during useful work; no overnight idle GPU. If the first window cannot cover planned work, use session 2 for remaining priorities and cut PCB rather than open a third session.

## Gates and cut order
1. CPU readiness: runnable scripts, data/weight checksums, source/support/calibration/test separation, mask geometry, domain enumeration, PCB audit and frozen protocol hash must pass before main work.
2. Local smoke: batch one on the 4 GB GPU checks correctness only. Re-measure representative extraction, fitting, scoring, map evaluation and storage on the FIRST Kaggle run; never use local throughput as a Kaggle estimate. Project the ENTIRE remaining matrix, including fixed overhead, all consumed GPU time and reserve; persist measured costs. Stop expansion if projection exceeds 12 GPU hours. Cut PCB, then ablations (including alignment), then main seeds to three. Keep the controlled threshold table unless a revised scope is explicitly approved. If the three-seed core still cannot fit, stop and report infeasibility.
3. PCB: Day-1 CPU audit go/no-go, followed by a final CPU confirmation freeze before its one evaluation batch. No new PCB settings after seeing results. No-go means development-only evidence.
Budget-only reductions before unrun work are logged/versioned without changing splits, algorithms or test-set rules. Never cut leakage checks or silently alter the frozen scientific protocol.

## Deliverables and next step
Reuse existing report/figure/table locations. Add only required method code, immutable run configs, machine-readable summaries and updated report. Git stores code/protocol/config/small summaries only; feature caches/checkpoints/datasets stay external. P0/P1 are complete; P2 and restricted P3 are the current authorization; stop after the smoke. Current handoff is the audit/manifests and a local commit. Later execution follows PROMPT.md P0-P7. User extracts returned results.zip into Desktop\DS\results\; Codex validates locally. No pilot files need copying from the absent Documents\Codex\DS folder.
