> **Current paths (2026-11-14).** Use [README](../README.md) and [the relocated notebook](../common/kaggle_run.ipynb). The instructions below describe the completed run. Its exact notebook and code bundle are preserved in `comparison/archive/`. Recorded result fingerprints remain historical.

# Current Kaggle handoff — 2026-11-14
Desktop/DS remains the single source of truth. Codex performs CPU preparation/validation locally. User runs kaggle_run.ipynb; two full committed runs plus short timing preflights, all charged to the same 12-hour GPU wall-clock budget. No remote/push. This replaces old --phase and persistent-cache instructions.

| Run | Queue | Expected rows |
|---|---|---:|
| First full committed run | Q1 | 270 |
| Second full committed run | Q2/Q3/Q4 | 90/54/0 (PCB final NO-GO) |

Upload code.zip as a PRIVATE Dataset ds-code. Import kaggle_run.ipynb, attach ds-code, select T4 x2 and Internet On. First use QUEUE="Q1", MODE="timing"; allow roughly 15-20 minutes for preparation (unmeasured, transfer may take longer). Send the projection before a full run. All extraction/training waits count, even CPU work in an attached GPU session.
For the approved full run, update GPU_H_USED with actual prior timing/setup usage, set MODE="full", and Save Version -> Save & Run All. The committed run remeasures and refuses a forecast above 9h for the session or above the remaining 12h total. Download results_Q1.zip to the DS root and request validation. Do not inspect test metrics before Q1 passes.
For session 2, attach validated results_Q1.zip as a separate private input (to restore source checkpoints), set QUEUE="Q2Q3Q4", and update GPU_H_USED to all previous actual GPU wall-clock time. Repeat timing if more than one day elapsed; the full notebook always performs a short source-only timing check. Download results_Q2Q3Q4.zip to the DS root afterward.

Raw data, pinned models and cache live under /kaggle/temp; features are rebuilt every session. No cache Dataset. Cache missing during queue execution fails closed. Each GPU writes a distinct shard directory/CSV; ZIP snapshots are serialized and include completed outputs only. Checkpoints are atomically replaced. Outputs plus ZIP must remain below 15 GiB. Workloads use the frozen 448px settings and paired manifests.
The data cell verifies the pinned file listing/content hashes and p1_audit image byte hashes. Byte-identical raw files preserve the audited decoded pixels/masks; it does not rerun p1_audit.main or regenerate splits. The bundle embeds the original pinned listing and only the four current Python modules, configs, protocol and text manifests.
No new test measurements were produced during notebook preparation. Later code bugs require a dated decision/refreeze and rerun of affected rows only. Budget reductions require a declared schedule update before execution, not notebook cell edits that silently omit jobs.

Snapshots use ZIP_STORED (lossless, uncompressed) to avoid repeatedly compressing growing checkpoints after every job. Reserve space for output plus ZIP; final snapshot also records session elapsed time. GPU_H_USED must include actual Kaggle session/setup/export time, not just training timers. Structural non-applicable calibrated columns in non-Q3 ledger rows are not missing experimental results.

## Reuse saved Q1 output directly
In a separate follow-up notebook, use Add Input to attach the successful Q1 notebook version whose Output contains out/results_Q1.zip. Keep ds-code1 at the repaired Q1 version. The follow-up notebook matches the Q1 archive against the validated SHA256 before installing dependencies or downloading data. Kaggle may expose both the archive and the loose output files. The checksum-matched archive supplies the results and checkpoints. A private Dataset made from the verified local standalone ZIP remains an alternative.
