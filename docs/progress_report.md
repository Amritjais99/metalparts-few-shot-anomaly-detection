# Completed MetalParts study at 448 px

**2026-11-14.** All 414 scheduled jobs verified (270 main comparisons, 90 ablations, 54 controlled threshold comparisons). Q4 was excluded by the frozen PCB gate. The historical 224 px report is retained in the separate Desktop backup `DS.zip`.

The follow-up export passed ZIP CRC, frozen fingerprint, ledger and completion-seal checks. All 270 Q1 rows and completion seals match the previous validated export. Target image AUROC was recomputed from sealed scores for all 414 jobs, and each of the 54 calibration runs used its ten assigned calibration normals. Both shards show clean completion. The follow-up used 0.841 GPU-session hours. Logged cumulative use was 3.046 hours, including the prior usage entered in the notebook. [Validation record](../comparison/tables/validation.json)

## Primary result

Mean target image AUROC. Seed means are taken within condition and the six conditions have equal weight. The test manifests contain 207 normal and 290 defect image entries across conditions.

| Method | K=0 | K=5 | K=20 | K20 minus K0 |
|---|---:|---:|---:|---:|
| Convolutional autoencoder | 0.6499 | 0.6658 | 0.6660 | +0.0161 |
| DINOv2 memory | 0.8458 | 0.8857 | 0.9060 | +0.0603 |
| PatchCore variant | 0.8293 | 0.8342 | 0.8690 | +0.0397 |

DINOv2 memory led at K=20 in all six conditions. The largest DINOv2 gain was on test5 (0.7235 to 0.9349). PatchCore gained most on test6 (0.6690 to 0.8629). All three methods fell on test3 from K=0 to K=20. The autoencoder improved by 0.0161 AUROC and was nearly flat from K=5 to K=20.

The predeclared K20 comparison gave autoencoder minus DINOv2 AUROC **-0.2400** (paired 95% CI **[-0.2926, -0.1869]**). Two-sided permutation **p < 1e-4 (minimum attainable with 10,000 permutations)**. The image-level fallback applies to both procedures. Inference is **conditional on image-level independence, potentially optimistic**. Physical capture IDs are unverified, and the interval is conditional on the trained seeds. [Inference record](../comparison/statistics/primary-inference.json) · [Per-condition table and seed SD](../comparison/tables/q1-per-condition.csv)

## Ablations and operating thresholds

On the matched three seeds, the per-patch MLP reached 0.8676 AUROC versus 0.6674 for the convolutional autoencoder (+0.2002). This comparison is a post-hoc finding from the development test results. Removing replay gave +0.0239 target AUROC and slightly worse source retention. Norm-only adaptation gave −0.0171. Mean alignment gave +0.0097 for PatchCore and −0.0014 for DINOv2. The MLP differs in both architecture and parameter count, so the ablation cannot isolate the effect of spatial context. [Paired comparison table](../comparison/tables/q2-macro-ablations.csv)

In Q1, the autoencoder reached 0.8499 AUPRO at K=20 (0.8290 at K=0), while defect recall at the source-fixed threshold fell from 35.7% to 13.1%. At K=20, PatchCore had slightly higher AUPRO (0.9115 vs 0.9106) and higher source-threshold recall (66.6% vs 53.0%) than DINOv2. Source image-AUROC retention at K=20 was +0.0050, +0.0085 and +0.0035 for the autoencoder, DINOv2 and PatchCore, respectively. [Macro metrics](../comparison/tables/q1-macro.csv) · [Runtime and memory](../comparison/tables/q1-runtime-memory.csv)

At K>0, balanced accuracy uses the frozen source threshold and can change with score scale and ranking. Conv AE was near chance at about 0.54. [Summary with macro pixel AUROC and AUPRO](../comparison/summary.md)

Q3 compared two thresholds on the **same** 10-adapted detector, using ten disjoint target normals for calibration. Ranking and localization scores are identical across the two thresholds. FPR@95% TPR is a separate retrospective ranking diagnostic, not an operating threshold selected by deployment data.

| Method | False alarms, source → calibrated | Recall, source → calibrated |
|---|---:|---:|
| Convolutional autoencoder | 4.0% → 7.3% | 11.4% → 20.5% |
| DINOv2 memory | 15.2% → 6.0% | 54.7% → 48.0% |
| PatchCore variant | 29.0% → 7.9% | 72.4% → 65.4% |

Ten normal calibration images offer a coarse threshold estimate and no guaranteed future false-alarm rate. The source threshold also rests on only 11–14 source normals per condition. [Controlled threshold table](../comparison/tables/q3-macro-thresholds.csv)

## Figures and reproduction

- [AUROC and recall by support count](../comparison/figures/metalparts-q1.png)
- The shared K=20 heatmap panels select by Conv AE outcome and compare the same image and seed across methods. Each method-specific reporting panel takes the first eligible saved-map image ID per condition and outcome, breaking ties by seed. The saved map subset contains no false-alarm heatmaps. The [earlier false-alarm panel](../comparison/figures/false-alarm-examples.png) shows original images only.
- A shared false-alarm heatmap panel would use up to six eligible normals, one per condition, and 18 maps across the three detectors. Estimate: 2–5 GPU minutes for a targeted batch, including initialization and export, with data, weights and saved detector states staged. Approval is pending.
- Frozen experiment bundle source commit `5c12000`. Five Q1 autoencoder runs retain their original pre-export-fix provenance in `session_Q1_recovery.json`. Run `python common/analyze.py --out comparison` to reproduce the paired analysis from the preserved model results. The exact frozen bundle and original notebook are in `comparison/archive/`.

The study uses six known MetalParts conditions and five main seeds, with three matched seeds for Q2/Q3. The PatchCore and DINOv2 memory detectors are project-specific, budgeted variants. Support draws overlap across seeds and unknown repeated captures may affect uncertainty. The results are development evidence. The existing study remains frozen.
