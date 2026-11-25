# Active scope as of 2026-10-27

Follow project-guide.md for MetalParts test1–test6 only. D067 supersedes prior quota-study and other-category next-step decisions below; older entries remain a historical log.

# Current project decision log

Updated: 2026-10-18 IST. Root: `C:\Users\Amrit Jaiswal\Documents\Codex\DS`.

Historical decisions D001–D053 are preserved in `archive/history-2026-10-18/decision.md`. That historical log is not the current project status. Recorded data, metrics and protocol hashes are preserved.

## D054 — Consolidate the active study and retire pilot material

- Date: 2026-10-18. Status: Adopted, requested by the user.
- Decision: Keep the active expanded study in DS; move pilot-only code/config/data/caches/predictions/figures/reports to `archive/pilot-2026-10-18`. Move superseded interim notes, old plan/log, original code snapshot and duplicate import staging to `archive/history-2026-10-18`.
- Why: Pilot results and current results were mixed, with stale status claims across several files.
- Evidence: 546 current-study artifacts passed SHA-256 checks across consolidation; the recorded expanded protocol is unchanged. Shared Python environment and pretrained assets are retained. Two legacy files held open elsewhere have archived copies and original source files retained; they are explicitly excluded from current analysis. See maintenance audit and relocation map.
- Consequence: No permanent deletion of research evidence. Active README/plan/report become the navigation and status sources; `.venv` library count is not project code count.

## D055 — Correct the description of completed work

- Date: 2026-10-18. Status: Adopted.
- Decision: Describe completed runs as memory fitting/adaptation with frozen CNN and DINOv2 features. Each MetalParts condition starts from its own recorded source/reference configuration; the six tests are not sequential gradient training of one model.
- Evidence: Run provenance attests `backbone_frozen: true`; detector fitting selects feature vectors and applies no optimizer/backpropagation.
- Scope: Full repository staged on Kaggle; only MetalParts test1–test6 evaluated so far, using 224px/custom filtered splits and capped memory. Do not call this a complete official RobustAD benchmark or all-category study.
- Limitation: Existing source-fit/validation memberships vary by target condition after duplicate exclusions, so cross-condition AUROC differences are descriptive rather than a pure causal measure of shift.

## D056 — Adopt a focused semester research question

- Date: 2026-10-18. Status: Adopted as direction; proposed method remains untested.
- Decision: Study when few verified-normal target images help or hurt under a fixed reference-memory budget. Compare source-only memory, existing pooled selection, fixed source/target quotas, and an experimental normal-only selection of quotas.
- Why: Existing means show benefits at k=20 for four of six MetalParts conditions and declines for two for both detectors. Running more categories measures breadth; it does not establish method novelty.
- Candidate contribution: A budget-controlled, leakage-audited empirical study, plus a target-memory allocation rule if it demonstrably improves reliability. Do not claim new backbone training or invent a novelty claim.
- Sources: Current focused literature review in `report/literature.md`; AnomalyDINO, RoDA and SPARC already cover closely related few-shot/shift adaptation problems. Novelty of the proposed allocation policy is unverified.

## D057 — Separate exploration from future confirmation

- Date: 2026-10-18. Status: Adopted.
- Decision: Treat the six already-observed MetalParts conditions as exploratory/development evidence. Freeze the follow-up protocol before inspecting defect metrics on new PCB/PiledBags conditions or an external dataset.
- Why: A method designed after seeing current test results cannot claim those same images as an untouched confirmatory test. More support draws on the same images do not repair this.
- Analysis: Match support draws across methods, use fixed evaluation IDs, report all endpoints and negative outcomes, measure harmful-adaptation frequency and runtime/memory, retain per-image predictions for paired/group-level uncertainty. Ten draws are a target if support pools permit, not ten independent datasets.
- Limits: Known-normal support alone cannot guarantee anomaly recall, absence of negative transfer, or a target false-positive rate. Do not claim such guarantees from a normal-score proxy.

## D058 — Set category and execution gates

- Date: 2026-10-18. Status: Adopted.
- Decision: PCB/PiledBags use target test1–test5; MetalParts uses test1–test6. PiledBags metadata may omit masks and its pixel/AUPRO metrics are unavailable. The runner already has an image-only PiledBags evaluation branch; the local metadata loader has been corrected to permit omitted masks for this group, with full-data verification pending.
- PCB: Diagnose why conservative pHash connected components leave too few eligible source groups. Do not lower leakage standards to force results. Exclude with a documented reason if no defensible split exists.
- Execution: Prepare CPU-only diagnostics first. No new GPU experiments are launched during this consolidation/research-planning task.
- Protocols: Preserve `configs/expanded_protocol.json` exactly for existing run provenance. The v2 study specification is a draft and must pass feasibility and implementation checks before execution.

## D059 — Protect reproducibility after cleanup

- Date: 2026-10-18. Status: Adopted.
- Decision: Default shared utilities to expanded-study paths, remove automatic pilot audit fallback, and require an explicit per-condition manifest selection. Preserve original Python sources and original Kaggle bundle in the historical snapshot before edits.
- Why: Archived pilot files must not be selected silently by a new run. Source changes do not retroactively alter historical runs or their hashes.
- Verification: Syntax/import and existing leakage tests are recorded in the maintenance validation file when completed. Shared analysis reads only current per-condition result files.

## D060 — Prepare the next diagnostic without repeating model experiments

- Date: 2026-10-18. Status: Implemented locally; not executed on the other groups' real data.
- Decision: Default manifest enumeration to the documented target list for each group, retain PCB/MetalParts mask-schema requirements, and permit absent mask fields for PiledBags only. An explicitly requested nonexistent group/condition fails with a clear error.
- Why: The earlier test6 errors were protocol enumeration errors; PiledBags mask-key errors were a loader contract error. Neither justifies modifying dataset bytes or weakening leakage checks.
- Artifact: `kaggle/02_category_diagnostics_cell.py` inspects actual pinned schemas and exact/basename/pHash component capacity on CPU. It uses the existing published PCB builder and therefore can be pasted into the current Kaggle notebook without first uploading a new bundle.
- Integrity: Existing recorded v1 config and results remain unchanged. The v2 proposal remains disabled. Local source changes require a deliberately versioned bundle before new model experiments; no new Kaggle runs or dataset downloads were started here.

## D061 — Record final consolidation and review evidence

- Date: 2026-10-18. Status: Completed for this maintenance/design step.
- Evidence: Final validation confirmed the original 546 protected current-study files and recorded protocol unchanged, 15 active Python files syntactically valid, and 24 unit checks passing. New checks cover maskless PiledBags metadata, retained PCB schema requirements, and group-specific target ranges. The original test1 ZIP was also recovered from Downloads, CRC-checked, matched to the current metrics and copied into `results/archives/`.
- Analysis: Recomputed 168 metric rows / 84 cases from the six recorded MetalParts bundles, using no pilot inputs. Saved target endpoints, paired deltas, actual bank allocation, membership and runtime tables under `results/analysis/`, plus an inspected PNG/PDF support-effect figure.
- Decision: Keep this as an exploratory baseline package and proposal. No proposed allocation method, real other-category diagnostic, confirmatory study, new heatmap set or demo is claimed completed. The local Kaggle source bundle is prepared/versioned only; it has not been uploaded or run.
- Cleanup exception: The original legacy pilot CSV and empty monitor log remain held open by another process; archived copies and the exception record are retained. Close the applications holding those files before relocating the remaining originals. Do not rerun the completed historical consolidation script over new current documents.

## D062 — Execute category readiness diagnostics on Kaggle CPU

- Date: 2026-10-18. Status: In progress following the user's “go ahead”.
- Decision: Use a fresh view of the existing experiment notebook, select accelerator None, restore the pinned dataset once, and insert a separate diagnostic cell after setup. Do not run the old GPU check, model-asset downloader, experiment runner or Run All.
- Why: The existing session was off; old displayed outputs were saved notebook history, not proof that temporary images still existed. Category readiness requires the actual pinned images and metadata.
- Diagnostic: Check PCB/PiledBags train/test0/test1–test5 schemas, verify absence/presence of test6, and compare PCB source capacity under exact-content, basename, and full pHash grouping. This is a diagnosis, not permission to weaken the leakage rule.
- Dependencies: The notebook diagnostic installs ImageHash 4.3.2 from the standard package registry only if absent. No model weights or complete dataset re-download is required; the existing attached lossless archives are reused.

## D063 — Keep PCB blocked pending a defensible duplicate-review policy

- Date: 2026-10-18. Status: Adopted from actual CPU diagnostics.
- Evidence: For every PCB target test1–test5, all 180 normal training images remain eligible under exact-content and source-basename checks, while adding pHash Hamming-distance-4 connected components leaves only two eligible source groups. Candidate-pair counts are 10,960 / 11,305 / 9,417 / 9,947 / 8,395. The largest component in test1 contains 574 images.
- Interpretation: The current broad similarity rule is the reason for the split failure. It does not establish that all merged images are duplicate captures, or that exact-hash-only separation would meet the requested near-duplicate rule.
- Decision: Do not lower the minimum source capacity, disable the similarity check, or run PCB as if it passed. Retain the failure report. A separate blinded capture-similarity review or independently justified duplicate criterion is needed before including PCB in the study.
- Evidence location: `results/diagnostics/category-readiness.json`, captured from the notebook's complete rendered JSON. Capture provenance records that the direct download link timed out.

## D064 — Restore an explicit normal-reference input guard

- Date: 2026-10-18. Status: Implemented locally; not a new model run.
- Decision: When callers provide an input label, reject non-normal reference images even if the supplied role is source_fit or target_support_pool. Label-free inference manifests still rely on the required pre-fit/provenance audit.
- Why: The existing detector unit check expected this guard, but the current fit implementation checked roles only. The added guard prevents an authorized role name from overriding a contradictory provided label.
- Consequence: Historical results are unchanged; their fitting inputs were audited normals. The change affects no distance, selection budget, model weight or valid-normal bank result. Core detector/metric checks are run on synthetic CPU inputs.

## D065 — Admit PiledBags to feasibility; preserve the confirmation gate

- Date: 2026-10-18. Status: CPU feasibility complete; model experiments not started.
- Evidence: Actual PiledBags test1–test5 all pass the original pre-fit separation audit with only the optional-mask schema correction. Each uses the same 68 source-fit and 17 source-validation IDs. Support pools are 31/40/40/40/40; target evaluation counts are 127/245/148/161/167.
- Decision: PiledBags is feasible for image-level confirmation after the complete method/protocol is frozen. No pixel metrics or physical-object independence claim is supported. PCB remains blocked under D063.
- Backup: Saved rendered JSON and transferred the notebook-generated 23-file evidence ZIP locally; CRC and config semantic equivalence checked. Exact executed sources and original audit/support records retained. Notebook Quick Save selected explicitly instead of Run All; interactive CPU session stopped after local backup. No GPU or model fitting was used.
- Validation: The additional existing core detector/metric suite passed 9 synthetic CPU checks. Historical recorded results/config remain unchanged. The v2 protocol remains disabled.

## D066 — Stage the next user-run experiment after code preparation

- Date: 2026-10-18. Status: Instructions saved; v2 implementation still pending.
- Decision: Start development with MetalParts/test1 and PatchCore, seed0, source-only/pooled/25%/50% quota comparisons at applicable support counts. Expand only after inspecting provenance and outputs. The first run is a development check, not final evidence.
- Why: Give the user a small visible execution step while preserving matched v2 baselines and untouched PiledBags confirmation. Keep Kaggle stopped while the new runner and notebook are prepared.
- Artifact: `report/next-steps.md` states exact bank allocations, user/engineer responsibilities and the execution gate. No new model run, download or implemented quota selector is claimed.

## D067 — Follow supplied guide; finish the MetalParts comparison

- Date: 2026-10-27. Status: Adopted and local cleanup executed.
- User direction: Keep all six MetalParts experiments/results, delete obsolete pilot work, and follow the supplied guide. The guide is saved verbatim as project-guide.md. Current scope is MetalParts only.
- Decision: Stop the unimplemented v2 quota-study direction. Finish existing 0/5/20-support comparison, current heatmaps/failure examples, report and presentation. Other categories and new methods are future optional work.
- Cleanup: Deleted the retired pilot directory (1,548 files), unused v2 config/proposal, obsolete category-diagnostic cell and stale local source bundle/metadata; deleted the two residual legacy pilot files. All six original MetalParts ZIPs, metrics, runs, manifests, pretrained assets and recorded config are protected.
- Review limitation: Automatic approval review rejected broad deletion of the entire archive/history and other evidence. Narrow pilot deletion succeeded. Historical evidence was retained; other-category diagnostics were moved into inactive archive/out-of-scope-categories. No cloud notebook or Kaggle data was deleted.
- Reproducibility: Preserved original relevant source and pre-cleanup current source in results/reproducibility/original-source-snapshot.zip. Verified hashes for 778 protected files after cleanup.
- Actual progress: 84 normal-memory fits/evaluations and 168 metric rows across six conditions. There were no neural weight updates. Local compact downloads omit per-image predictions/maps and fitted bank arrays. Check saved Kaggle outputs before deciding whether controlled regeneration is necessary.
- Interpretation: Single-category study; averages are across conditions. No invented publication/novelty claim, no post-hoc hypothesis described as preregistered, and no new GPU run during cleanup.

## D068 — Generate the professor progress report from saved evidence

- Date: 2026-10-28. Status: Report and supporting artifacts generated after the user's confirmation.
- Artifacts: `docs/progress_report.md`, `docs/talking_points.md`, source-cited CSVs under `docs/tables/`, six scientific figures in PNG/PDF under `docs/figures/`, and build/source-validation records. Reproduce with `python src/build_progress_report.py`.
- Evidence: Raw CSV/JSON values were matched for all 168 metric rows; 84 case records and six split/audit sets were read. Support ownership/nesting, protocol/manifest hashes and recorded source-code hashes were checked. Report numbers distinguish read values from deterministic derivations and interpretation.
- Analysis: Use equal-condition averages and sample SD across seed-wise averages; condition tables show draw SD separately. K=0 is source-memory fitting, not empty-memory zero-shot inference. Epoch training curves/optimizer settings do not apply to frozen-backbone reference fitting.
- Confusion matrix: Reconstruct integer counts from full-precision FPR/recall and denominators. The displayed maximum-AUROC run is explicitly retrospective, not a configuration chosen by independent validation. No individual image errors or heatmaps are fabricated.
- Limitations: Per-image maps/scores/banks and total parameter counts were not found in active local results; no Git commit exists in this repository context. Five recorded source-file/hash pairs differ from current code. The report records the source drift and different raw/canonical config-hash conventions.
- Execution: No neural training, GPU inference, dataset download or long experiment was launched. Optional isolated recovery commands are documented, not executed. Existing MetalParts result artifacts remain untouched.

## D069 — Build a self-contained six-condition recovery notebook

- Date: 2026-10-29. Status: Generated and locally validated; real Kaggle execution pending.
- User direction: Generate one notebook to reproduce MetalParts test1–test6 and preserve the previously omitted per-image evidence in one ZIP.
- Artifact: `kaggle/02_metalparts_recovery_all_six.ipynb` has five code cells and embeds 73 source/config/manifest/reference files. `kaggle/RECOVERY_START_HERE.md` gives import, input, GPU, download and local ZIP-verification steps. Rebuild with `python kaggle/build_metalparts_recovery_notebook.py`.
- Data: Reuse the saved lossless MetalParts archive only. Verify its original bytes against the pinned input inventory; replay the existing structural separation checks with the original conservative capture grouping. Recorded decoded fingerprints are retained, not recomputed. No new splits/support IDs are generated and physical-object independence remains unknown.
- Execution: Use a separate `/kaggle/working/DS-MetalParts-Recovery` project, all original support counts/draws, one GPU, and shared temporary feature caches. Both backbones remain frozen. Source maintenance changes are declared; every regenerated endpoint/threshold is compared with the original at absolute tolerance 1e-6, with differences retained for review rather than tuned away.
- Export: Keep all 252 prediction NPZs (scores/maps), 84 fitted reference banks, 168 metric rows, per-image CSVs, audits/configurations, original metrics, logs and illustrative seed-0 heatmaps in one Zip64 archive. The exporter checks case/domain coverage, file existence, contents against final image metrics, SHA256 inventory and ZIP CRC. Raw dataset copies, pretrained assets and caches are excluded; actual archive size/runtime are measured during execution.
- Examples: Post-evaluation TP/TN/FP/FN examples are chosen closest to the threshold within each available outcome, with stable ID ties. Missing outcomes are recorded. Display scales are per-image; raw maps remain preserved. No example affects fitting or tuning.
- Validation: All five notebook code cells parse; embedded-file hashes and exact original manifests/metrics/config were checked. Six synthetic CPU integrity tests pass, including a full archive with 252 placeholder prediction files/84 banks, refusal of a missing file, metric drift/count guards, original split replay and synthetic score/PNG export. These fixtures are not model experiment results. No real model, GPU session or dataset download was run locally.

## D070 — Verify and retain the completed six-condition recovery

- Date: 2026-10-29. Status: Received ZIP inspected; verdict PASS.
- Input: `C:\Users\Amrit Jaiswal\Downloads\DS-MetalParts-all-six-recovery.zip`, 4,334,529,972 bytes. Received SHA256: `0610bc17a6e6aaf8243dc0e5052a6ae08adb18240d144af1023bd3dc2b4e4b7e`. No cloud checksum sidecar was supplied; every manifested member was independently hashed and the local archive hash retained.
- Integrity/completeness: All 1,073 ZIP members passed safe-path/CRC checks; all 1,072 manifested member hashes match. Retained under `results/recovered/MetalParts-all-six-2026-10-29/`. Contains 252 full prediction NPZs, 84 banks, 168 metric rows, 22,456 CSV score rows and 132 PNG examples. Twelve unavailable outcome slots were confirmed valid, not missing experiments.
- Independent checks: All 84 case banks were checked for capacity, dimension, finite contents, metadata and authorized patch owners. Every prediction archive was checked for role IDs, metadata, geometry, finite maps/scores and runtime-log hash. Source-normal quantiles and image AUROC/FPR/recall were independently recomputed from scores. Per-image CSVs and example choices match actual NPZ predictions. All 1,008 original/recovered metric and threshold values are exactly equal; no inference/training was executed during local inspection.
- Separation: Original split/support bytes are unchanged; fitting/calibration/test membership and frozen-backbone provenance match. Physical-object independence remains unknown. Source-fit membership varies across conditions; source-test membership is common.
- Limit: Original image/mask files were deliberately excluded from the export. All 341 referenced raw GT masks are absent locally, so pixel AUROC/AUPRO were checked against original results but not recomputed against ground truth. Their predicted maps and rendered true-mask example panels are present. A mask-only retrieval would allow fresh pixel verification without repeated model inference.
- Analysis: Average ranking improves with support, but both methods lose mean AUROC in two conditions at k=20. DINOv2/test6 improves AUROC 0.7548 to 0.8968 while mean recall falls 77.5% to 21.7% under the recorded source-normal calibration. Treat threshold transport and missed defects as key findings; do not tune thresholds on test defects to overwrite the historical outcome.
- Deliverables: `docs/recovery_inspection.md` and `.json`; independent `docs/tables/recovery_*.csv`; actual correct/missed examples and a paired error-transition figure. Historical progress report has a recovery update banner. Check all original 778 protected artifacts unchanged.
- Next: Finish the MetalParts professor report/presentation using real current heatmaps and failure analysis. Other groups/new methods remain deferred. No further model rerun is needed for existing evidence; original mask retrieval is optional for independent localization verification.

## D071 — Prepare three simple project slides in the user's Canva template

- Date: 2026-11-04. Status: Saved after the user explicitly approved the previews with "save the slides". Canva confirmed transaction 2005748838714349460 committed. Saved text and exactly three pages were independently read back.
- Source: User-supplied design DAHZ8UJ1iAQ. Copy pages 2, 3 and 8 into separate design DAHZ8QSA6mA to preserve the original and provide exactly three slides.
- Content: Why choose the project, methodology, and expected outcomes. Use simple language and explicitly state that K counts good target images and pretrained weights stay fixed. Describe reference-memory adaptation rather than gradient training. Expected outcomes do not promise that more support always improves detection.
- Style: Preserve the blue/white template, navy headings, icons and layouts; replace unrelated example names with MetalParts Project and Amrit Jaiswal.
- Artifacts: docs/canva_three_slide_brief.md and docs/canva_project_slides.json retain the content brief and resumable transaction details. Previews were shown in chat. No experiments were run.

## D072 — Clean and relocate the active project to Desktop

- Date: 2026-11-12. User requested removal of unnecessary material and relocation to Desktop.
- Location: C:\Users\Amrit Jaiswal\Desktop\DS. The former Documents\Codex\DS folder was moved, not duplicated.
- Removed: package/model download caches, temporary synthetic-test outputs, stale IDE settings, out-of-scope category diagnostics, empty Word rendering workspace and active-code bytecode caches.
- Preserved: MetalParts test1–test6 code/configs, splits/support draws, all original and recovered metrics, prediction arrays, banks, heatmaps, reports, notebooks, pretrained assets and local Python environment.
- Historical project evidence moved under results/reproducibility/history-2026-10-18. Recorded experimental configurations remain immutable, including historical other-category entries.
- Verification: SHA256 comparison of saved config/split/metric files across relocation; record in results/project-maintenance/desktop-relocation-2026-11-12.json. No model experiment was run.
- Updated README and environment activation paths. Historical log paths remain unchanged as evidence. Use the relocated Python interpreter and python -m pip.

## D073 — Reduce clutter at the user's request

- Date: 2026-11-12. Remove talking points, Canva draft brief, superseded result/next-step summaries, the old manual per-condition notebook, old session instructions and interim historical reports.
- Remove the local virtual environment containing 30,339 dependency files (about 4.9 GB decimal). Save its package list in results/setup/requirements-local.txt. Local execution requires recreating an interpreter; Kaggle inputs/results are unaffected.
- Consolidate literature and references under docs and remove the redundant report folder. Disable talking-points export in the active report builder.
- Keep final Word/Markdown reports, presentation link, complete MetalParts results, scientific source/config/split evidence, notebooks and relevant tests. Preserve original result provenance.
- MetalParts.tar was present in the Desktop project before this cleanup; preserve it. This cleanup does not verify its contents.

## D074 — Minimal active project layout

- Date: 2026-11-12. User requested a stricter cleanup preserving current work and future analysis.
- Removed downloadable model weights/repository, empty raw/cache directories, obsolete full-category downloader, redundant per-condition archives and superseded historical directories.
- Consolidated full recovered evidence into results/MetalParts-full-evidence.zip; copied archive matches previously verified SHA256. Consolidated 492 original records into results/original-study-records.zip and verified every archived file against its original before removing extracted copies.
- Kept summary metrics, all code/checks/configs, exact splits, final reports/tables/figures and the all-six Kaggle notebook. Moved the existing MetalParts.tar into data.
- Added src/prepare_analysis.py to restore evidence on demand. No model run is required. Word report builder now uses a relative project evidence path after restoration.
- Historical report paths refer to the extracted layout; evidence must be restored before detailed analysis/report or notebook rebuild. Next substantive work remains presentation and optional threshold/mask analysis.

D075 | 2026-11-12 | Approve spatial conv-AE comparison at 448px, K=0/5/20, five paired seeds; AE K=0 has independent source training | User approved neural-method scope with 25-hour/three-day budget | Historical 224px results remain development evidence; docs/project-scope.md and docs/algorithm-spec.md govern new work.
D076 | 2026-11-12 | Primary image AUROC; source-fixed threshold under score shift; separate 10+10 table | Avoid mixing changed scores with threshold-selection effects or using fit images for calibration | Retain negative results and source forgetting; tiny calibration counts give imprecise estimates.
D077 | 2026-11-12 | One paired within-condition image-score permutation test and grouped paired bootstrap | User clarified image-level permutation and six-condition mean AUROC statistic | protocol.md fixes pairing, rank-score scale handling, assumptions and uncertainty limitations; no condition-only permutation or multiple testing suite.
D078 | 2026-11-12 | CPU-only setup/audit/freezes; <=12 GPU hours, <=2 sessions; one-time feature cache in private pinned Kaggle Dataset | Prevent setup/idle GPU waste and re-extraction across sessions | Cache is never committed; storage estimate is conditional on audited counts; cut PCB, then ablations, then seeds to three if needed.
D079 | 2026-11-12 | Create new PROMPT.md from approved chat rules | Searches found no original prompt; user does not know its location | Cannot claim exact preservation of missing text; prompt includes explicit CPU/GPU, GitHub, simplicity, token, leakage and P0-P7 rules. Current authorization is documents and commit only.
D080 | 2026-11-12 | Initialize a local Git repository and commit only planning artifacts | User requested commit; DS has no repository/remote and no configured author identity | Use honest Codex agent commit identity locally; no GitHub push or source-code versioning claim. Execution requires subsequent implementation commit.
D081 | 2026-11-12 | Desktop\DS is the single source of truth; Codex local CPU/smoke, user Kaggle full-scale GPU; commit locally after every phase and never create a remote | User clarified local-only ownership and transfer requirements | Documents\Codex\DS is absent; no pilot copies identified; return phase outputs into results/ without replacing historical evidence.
D082 | 2026-11-12 | Add scripts/bundle with text-only src/configs/protocol/manifests allowlist and documented per-phase Kaggle cells | User needs a minimal private upload and one results.zip handoff | Runner and verified dependency/path configs remain P2 work; cells are an explicit future interface, not a claim of executable implementation; cache is separate and never in Git or results.zip.
D083 | 2026-11-12 | Local 4 GB batch-one smoke is correctness-only; first Kaggle run re-measures whole-project cost | Different hardware cannot supply a trustworthy throughput forecast | Count local smoke conservatively within 12 total GPU hours, retain two Kaggle windows and approved cut order; no experiments started.
D084 | 2026-11-12 | Disclose reconstructed prompt additions and extend ignore coverage to cache/ and features/ | No original PROMPT.md was recoverable; user requested transparency and tracking checks | docs/local-workflow.md compares prior reconstruction to approval; no tracked environment/data/big binaries found in initial seven-file commit, so no history rewrite is needed.
D085 | 2026-11-13 | Execute P0/P1 only on CPU in Desktop\DS; use existing Python 3.13.15 and install only missing requested packages with required dependencies | User authorized data readiness, not training or P2 | P0 environment recorded in docs/p0-environment.json; assume 25 work hours/three days available at start, P0 estimate 0.15 h, GPU 0 h.
D086 | 2026-11-13 | Pin AmazonScience/RobustAD to 82c642b2d18be60f2dab104f8dd4dea1f4ccfacb; allow only MetalParts/**, PCB/** and README.md | Dataset listing gives 3659 group files / 6863601708 bytes and card declares CC BY 4.0 | Disk gate passes with >10 GB reserve; reuse only hash-verified existing archive members, resumable hub download into data/raw; never download PiledBags.
D087 | 2026-11-13 | Fix pHash size 8/highfreq 4, candidate distance <=4 plus byte/decoded exact hashes before audit | Required reproducible leakage screen without performance-driven threshold changes | Preserve historical held-out IDs; unresolved cross-role candidates fail closed; pHash components do not prove physical identity. Automated decoding limited to requested audit, few visual samples only.
D088 | 2026-11-13 | Separate evaluation-only similarity links from fit/support/calibration collisions | Full-inventory screening can link source-test and target-test through excluded images; this is dependence, not evidence that fitting accessed a test image | Keep all historical held-out IDs; retain evaluation links for uncertainty review, without relaxing pHash <=4 or permitting fit/calibration collisions.
D089 | 2026-11-13 | Store new small tracked audit/manifests in data/manifests/p1 and map them to manifests/ inside code.zip | User explicitly chose data/manifests for P1; avoid duplicating canonical manifests | Ignore data/raw and historical expanded records; bundle scans only the designated text manifest directory, never raw images.
D090 | 2026-11-13 | Use resumable HTTP with 256 KiB stream writes, 60-second timeouts and bounded transport retries | Xet/cache restrictions and dropped connections repeatedly discarded unwritten large chunks; installed downloader source confirmed buffering behavior | Preserve completed/partial files, unchanged pinned revision/allowlist and repository hash validation; no GPU, weights or PiledBags downloaded.

D091 | 2026-11-13 | P1 integrity/count/manifest checks completed; preserve all six MetalParts held-out sets and extend paired draws to five seeds | 3659 group files verified at pinned revision; 2662 images audited; old experiment counts match exactly, K=20 and 10+10 pass | data/manifests/p1 and docs/data-audit.md are the new audit record; 959 unique MetalParts manifest images, 1350 draw rows, no fit/calibration collisions; disclose evaluation-only candidate dependence.
D092 | 2026-11-13 | PCB NO-GO for all five conditions under the conservative capture-screen gate | Only two source-normal candidate components remain after exclusions, below 20 fit + 10 validation; independent capture identities not documented despite valid labels/masks | Report development-only MetalParts evidence; do not claim PCB confirmation, true physical-object counts, or relax the hash threshold.
D093 | 2026-11-13 | Stop before P2 and conservatively charge approximately 1.50 work hours for P0/P1 including transfer waits | User limited authorization to P0/P1; baseline 25 h / three calendar days assumed at start | P0 estimate 0.15 h, P1 estimate 1.35 h, approximately 23.50 work hours remain; GPU/training/feature extraction 0; no remote or push.

D094 | 2026-11-13 | Save one PCB regrouping proposal without applying it | Exact-only and pHash<=1 sensitivity checks separate filename records which pHash<=4 transitively joins; physical identity remains unknown | docs/pcb-regroup-proposal.md records evidence; PCB remains NO-GO and manifests are unchanged.
D095 | 2026-11-14 | Replace CPU torch/torchvision with matching 2.11.0/0.26.0 CUDA 12.8 builds in the existing interpreter | User explicitly approved the download and shared-environment replacement | Installation succeeded; import and CPU selftests must pass before freeze or GPU smoke.
D096 | 2026-11-14 | Use the requested reviewed-group/giant-component fallback rule before freeze | No P1 physical capture identities were verified; candidate components are not reviewed captures | Label six-condition inference conditional on image-level independence, potentially optimistic; preserve all separation gates and PCB NO-GO.
D097 | 2026-11-14 | Implement exact tied-score region-weighted AUPRO, 8-connectivity, FPR<=0.30; bilinear 448px maps without smoothing | New algorithm spec requires shared upsampling; avoid copying historical histogram/Gaussian behavior without disclosure | Verify AUPRO against independent brute-threshold synthetic reference; these new variants are not numerically identical to historical 224px variants.

D098 | 2026-11-14 | Freeze P2 protocol/config/code/manifests after passing synthetic CPU checks and 120 role checks | Required pre-GPU executable freeze | protocol.md SHA256 86b5ad38dd868a1cde421c29d74dd30983102e7c0177faf897aedd569acd3ef1; configs/config.yaml SHA256 adf9975559a394551a43009a9f2f30c1c4eb0213f29b0abea6681964a94e05c7; full fingerprint fe78f6dcb782112e6a750204b5dac0e6c3c7a33331bf7a56e8f7bb6dd3afabe4; subsequent changes require a dated decision and affected-output invalidation.

D099 | 2026-11-14 | Supersede initial D098 freeze before any GPU work: exclude annotation files from smoke-time manifest hashing | Review caught that generic CSV fingerprinting would open annotation bytes during P3, despite not parsing labels | Fingerprint only label-free identity/roles/draws during fitting/smoke; pin annotation SHA256 in config and validate it only at final evaluation. Re-run CPU selftest and an annotation-open guard; no experimental outputs exist to invalidate.

D100 | 2026-11-14 | Final P2 executable freeze supersedes D098 | Synthetic CPU checks and annotation-open guard passed after D099 fix | protocol.md SHA256 86b5ad38dd868a1cde421c29d74dd30983102e7c0177faf897aedd569acd3ef1; configs/config.yaml SHA256 efc0d3b91a3df91554697949cf42dc566e4d55244e4d1ab6fd3714dc27f97f1b; full fingerprint b7c73ee93a93b382f7749b513eb3589f32eee97af7aeab8f7829d1a332bf2bfe; no GPU or main-study output exists before this freeze.

D101 | 2026-11-14 | Add explicit dataset-group keys to future metric records and summary grouping; refreeze bookkeeping code | Final review found that future PCB summaries could otherwise be combined with same-named MetalParts conditions | No main metrics exist to invalidate. Re-run CPU selftests. P3 timings remain tied to the D100 freeze saved in docs/p3-smoke-freeze.json; data.py/model.py and all smoke computations are unchanged. PCB stays NO-GO.

D102 | 2026-11-14 | Refreeze D101 metadata fix after CPU selftests pass | Preserve dataset separation in future summaries without changing any scientific setting | protocol SHA256 86b5ad38dd868a1cde421c29d74dd30983102e7c0177faf897aedd569acd3ef1; config SHA256 efc0d3b91a3df91554697949cf42dc566e4d55244e4d1ab6fd3714dc27f97f1b; current full fingerprint c0145691a5fb7f283acec6b64063a60f5cb8c6d6a223eb5a6032f3f42c3416fc; prior smoke fingerprint b7c73ee93a93b382f7749b513eb3589f32eee97af7aeab8f7829d1a332bf2bfe remains its honest measurement provenance.

D103 | 2026-11-14 | P3 local normal-only GPU smoke passed on RTX 2050 | Three authorized image files opened; no annotations/test metrics; highest reserved VRAM 1.40 GiB; DINO/CNN 448px grids verified | docs/p3-smoke.json and p3-access-audit.json preserve measured timings; conservatively count 21.913 seconds toward the 12-hour GPU cap; no full run.
D104 | 2026-11-14 | Retain Kaggle full-run location; do not propose a local execution exception | Full local wall projection 12.21 h including CPU metrics and 25% reserve exceeds approximately 10 h; 6.44 GiB feature cache and 2.29 GiB output estimate fit disk | Kaggle 5-9.5 h is an UNMEASURED prior planning range, not local extrapolation. Remeasure there before expansion; no scope cuts yet. CPU notebook handoff remains the next gate; stop now.
D105 | 2026-11-14 | Record active-work estimate approximately 4.0 of 25 hours including P0/P1; 21.0 remain | Active work estimated, paused calendar interval excluded; see docs/p3-budget.json | PCB remains NO-GO; no main-study performance result or publication claim; no remote or push.

D106 | 2026-11-14 | Apply approved PCB-only grouping and one final CPU gate | Exact grouping increases capacity but unresolved candidate review remains mandatory | All domains final NO-GO; no model scores; see data-audit.md and config.yaml.
D107 | 2026-11-14 | Replace persistent cache workflow with two ephemeral-cache Kaggle sessions, two GPU workers, Q1 then Q2/Q3/Q4 | Latest user authorization supersedes previous platform/cache rules | Total 12-hour wall cap includes setup and preflights; no test metric inspection before Q1 validation; local CPU only.

D108 | 2026-11-14 | Interpret the two-session limit as two full committed runs plus short timing preflights | User explicitly clarified this after the Save & Run All session distinction was explained | Every preflight/setup/full-run minute still counts toward the common 12-hour wall budget.

D109 | 2026-11-14 | Implement Q1/Q2/Q3/Q4 job-index sharding, ephemeral extraction, source-only timing and atomic shard outputs | State 1 preparation authorized; no full results exist | Queue sizes 270/90/54/0; separate checkpoint/output directories prevent concurrent writers; serialized incremental ZIP snapshots; budget watchdog stops workers.

D110 | 2026-11-14 | Re-freeze before any queued test run after CPU selftests, 120 separation checks, queue partitions and synthetic export/resume checks pass | Required new queue/notebook contract changes code and execution policy | protocol SHA256 ac0cb8aae37a48b91459ccb2bbc8a07525104e5cf9f556361f3a33efb4b82d3d; config SHA256 49d8b15917c160372508d8533d7a52fd385306147246090543aea2a9e4d2524b; fingerprint d20db7c54dee9796f6fb25d64475c314e9cdcbe31afac3d98b93353a33241a45; GPU/test experiments not run locally.

D111 | 2026-11-14 | Final packaging freeze after lossless uncompressed snapshots and notebook checks | Avoid repeated checkpoint compression overhead; record elapsed wall time and retain strict budget watchdog | Protocol SHA256 ac0cb8aae37a48b91459ccb2bbc8a07525104e5cf9f556361f3a33efb4b82d3d; config SHA256 49d8b15917c160372508d8533d7a52fd385306147246090543aea2a9e4d2524b; full fingerprint 574553750850f508b7a36fb5194378d467118c197c2703dc3acec796a3cac6e1; synthetic tests passed, no queued test run or metric inspection.

D112 | 2026-11-14 | Repair notebook input discovery for ds-code1 and renamed or extracted bundles | The old exact-one-code.zip assertion did not diagnose the mounted input | Prefer the configured slug, verify the inventory hashes, and report missing/ambiguous inputs. Notebook-only setup change; existing code.zip and scientific freeze retained.

D113 | 2026-11-14 | Fix duplicate alignment keyword when exporting DINOv2/PatchCore metrics and surface worker tracebacks | Kaggle Q1 passed verification/timing/extraction but queue workers returned nonzero. Static inspection found a deterministic result-export crash; original worker logs pending. | Prior freeze 574553750850f508b7a36fb5194378d467118c197c2703dc3acec796a3cac6e1; repaired freeze de4a10c2ea7058f590d2ec30bd657cef1e3554344b1969b8d444ebe365441404. Config, protocol, manifests and metric calculations retain their existing hashes/definitions. Preserve partial Q1 archives for validation and assessment of unaffected completed rows before restart.

D114 | 2026-11-14 | Recover five verified Q1 autoencoder records from failed Kaggle output | Both worker logs confirm D113 duplicate alignment error. Outer archive SHA256 09c4a7ce767ac6f682427616c62a78fbe7f1893c1e2956edaca6354d1b593745; five completion seals and checkpoint/detector state pairs verified. | Preserve original archive. Bind the five unaffected K0 test1 results and source checkpoints to the D113 export-only repair, retain original provenance and unchanged prediction/metric values, exclude unfinished DINO runs. Recovery ZIP SHA256 afdf315bf0d0bb639c032b358070423272d027e0442d659cb312dc7a135b756e; notebook resumes 265 outstanding rows, and uses 1.03 h as the approximate prior total through the failure logs pending actual Kaggle end time.

D115 | 2026-11-14 | Validate successful resumed Q1 and prepare the Q2/Q3 Kaggle handoff | Inner archive SHA256 4e4460af2c684cde951dcf46e7491e437fb9f67126a27130e7311562d29d3fef matches the Kaggle output, all 270 scheduled job IDs and frozen test memberships match, both shards finished, 3240 sealed files verified, no missing or nonfinite required ledger metrics. | Retain wrapper and standalone results_Q1.zip. Reuse existing repaired code.zip. Follow-up notebook verifies Q1 input as ZIP or extracted tree, runs Q2=90 and Q3=54 MetalParts jobs, Q4=0. Logged prior GPU wall time 2.2049 h through Q1, subject to any additional actual session time. No metric values used to change the frozen schedule or protocol.

D116 | 2026-11-14 | Attach saved Q1 notebook output directly for session 2 | Browser upload of the 1.7 GB archive stalled. Kaggle supports notebook outputs as inputs. | Choose checksum-matched archive even when saved notebook output exposes loose siblings. Verify and stage before dependency downloads. Q1 frozen code, notebook schedule, and results retain existing hashes. CPU fixture checks cover saved output, standalone ZIP, extracted layout, and mismatches.

D117 | 2026-11-14 | Validate Q2/Q3 and close planned GPU work | kaggle_run2Q2Q3Q4results.zip SHA256 6e9d8685ca951f3f6f709677e9a1eb06bf18cb41638064db603b060a0545c410: 414 ledger jobs, 4554 sealed files, no missing or nonfinite scheduled metrics; all 414 target AUROCs independently reproduced from sealed scores | Q1/Q2/Q3/Q4 = 270/90/54/0. Logged GPU time 3.0457 h; primary analysis used frozen paired test. Results tables, figures and examples are under results/analysis448; existing progress report updated.

D118 | 2026-11-14 | Path-only refactor, no scientific change | All 12,130 members across three result archives verified against preserved files, 450 completion seals resolve, and all 414 scheduled rows plus 36 auxiliary runs retained. Pilot ZIP has 182 verified members, SHA256 d674d530cd8fa51b091445f056ad8cecc113798e69f4d7aa4c407f660c9e14c8. | Frozen protocol SHA256 before and after ac0cb8aae37a48b91459ccb2bbc8a07525104e5cf9f556361f3a33efb4b82d3d; original code.zip retained unchanged, new layout fingerprint 6216f8af01b6d87d88192a1ccd9035f248b79e9d5ffd8f654dde254bfea0e1aa. CPU selftests, 120 role checks, all Q1/Q2/Q3 summaries, paired CI and seed-20261112 permutation passed. Workspace caches and three installer/download logs removed after verification. Original raw prediction, checkpoint and ledger bytes preserved. Removed archive SHA256 values: {"kaggle_run2Q1results.zip": "d45eb17a419b71849014b9a35023fdaab5e517d2c8894cfdc4148003ac2c3f26", "kaggle_run2Q2Q3Q4results.zip": "6e9d8685ca951f3f6f709677e9a1eb06bf18cb41638064db603b060a0545c410", "results_Q1.zip": "4e4460af2c684cde951dcf46e7491e437fb9f67126a27130e7311562d29d3fef"}.

D119 | 2026-11-14 | Post-hoc reporting additions; metric definitions unchanged | The nine macro image AUROCs and ten sampled table cells match saved predictions. | Added CPU-only common/report.py --report, kept the frozen analysis entry point and hashes intact, and recorded unavailable qualitative examples. Four report PNGs of at least 1 MB remain local.
D120 | 2026-11-14 | Keep scope MetalParts only | Exclude PiledBags, PCB and all other datasets | All MetalParts findings are development-only evidence.
D121 | 2026-11-14 | Clean active DS for final MetalParts submission | PCB images and superseded archives were byte-verified against Desktop DS.zip; v2 draft and PCB regrouping proposal are recoverable from Git history | Keep all 450 sealed runs, current figures, all 1,912 pinned MetalParts files and frozen run inputs. The downloader now requests only MetalParts.
