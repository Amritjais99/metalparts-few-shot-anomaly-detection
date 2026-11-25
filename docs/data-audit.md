# P1 data audit — 2026-11-13

| item | value |
|---|---|
| Dataset | AmazonScience/RobustAD @ 82c642b2d18be60f2dab104f8dd4dea1f4ccfacb |
| License | CC BY 4.0; pinned README.md, common/configs/p1-dataset-listing.json |
| Integrity | 3659 listed group files verified by repository hash; README separate |
| Splits | Official train/test0 and MetalParts test1..6, PCB test1..5. Few-shot roles are custom, not official support splits. |
| Image access | CPU decoding for audit hashes/resolution/masks only; no model inference/training/GPU |
| Evidence | data/manifests/{MetalParts,PCB,shared}/*.csv; metadata paths retained in image_inventory.csv |

## Official raw split counts

| group | split | normal | anomalous | masks | missing_anomaly_masks | bad_geometry | nonbinary_masks | resolutions |
|---|---|---|---|---|---|---|---|---|
| MetalParts | test0 | 120 | 51 | 51 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test1 | 120 | 80 | 80 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test2 | 120 | 80 | 80 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test3 | 120 | 80 | 80 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test4 | 120 | 80 | 80 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test5 | 60 | 40 | 40 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | test6 | 60 | 40 | 40 | 0 | 0 | 0 | 2681x1500 |
| MetalParts | train | 180 | 51 | 51 | 0 | 0 | 0 | 2681x1500 |
| PCB | test0 | 120 | 75 | 75 | 0 | 0 | 0 | 2048x2048,2560x2560 |
| PCB | test1 | 120 | 90 | 90 | 0 | 0 | 0 | 2048x2048 |
| PCB | test2 | 120 | 60 | 60 | 0 | 0 | 0 | 2048x2048 |
| PCB | test3 | 80 | 60 | 60 | 0 | 0 | 0 | 2048x2048 |
| PCB | test4 | 80 | 60 | 60 | 0 | 0 | 0 | 2048x2048 |
| PCB | test5 | 80 | 60 | 60 | 0 | 0 | 0 | 2048x2048 |
| PCB | train | 180 | 75 | 75 | 0 | 0 | 0 | 2048x2048,2560x2560 |

## Domains and file structure

| group | split | shift_inferred_from_filename | normal_filename_example | object_id |
|---|---|---|---|---|
| MetalParts | test0 | source | normal/Live_Normal_None_None_75.jpg | NOT DOCUMENTED |
| MetalParts | test1 | lighting | normal/Live_Normal_Lighting_Ambient_Light_Var_250.jpg | NOT DOCUMENTED |
| MetalParts | test2 | position | normal/Live_Normal_Position_+5%_to_-5%_28.jpg | NOT DOCUMENTED |
| MetalParts | test3 | rotation | normal/Live_Normal_Rotation_+5_deg_and_-5_deg_38.jpg | NOT DOCUMENTED |
| MetalParts | test4 | scale | normal/Live_Normal_Scale_+5%_to_-5%_7.jpg | NOT DOCUMENTED |
| MetalParts | test5 | background 1 | normal/Live_Normal_Background_Background_1_57.jpg | NOT DOCUMENTED |
| MetalParts | test6 | background 2 | normal/Live_Normal_Background_Background_2_65.jpg | NOT DOCUMENTED |
| MetalParts | train | source | normal/Live_Normal_None_None_156.jpg | NOT DOCUMENTED |
| PCB | test0 | source Set2 | normal/Test_PCB_resized_None_None_Normal_Set2_222.jpg | NOT DOCUMENTED |
| PCB | test1 | lighting | normal/Test_PCB_Lighting_Morning_Normal_35.jpg | NOT DOCUMENTED |
| PCB | test2 | white balancing | normal/Test_PCB_White Balancing_Evening + AWB_Normal_25.jpg | NOT DOCUMENTED |
| PCB | test3 | rotation | normal/Test_PCB_Rotation_-5%_Normal_10.jpg | NOT DOCUMENTED |
| PCB | test4 | position | normal/Test_PCB_Position_-10%_Normal_26.jpg | NOT DOCUMENTED |
| PCB | test5 | shadow | normal/Test_PCB_Shadow_Object Alternate Location_Normal_2.jpg | NOT DOCUMENTED |
| PCB | train | source Set1 | normal/Train_PCB_resized_None_None_Normal_Set1_36.jpg | NOT DOCUMENTED |

| Review | Limitation |
|---|---|
| visual_review.csv | Three samples visually inspected; PCB layout similarity is not evidence of physical-object identity. Domain names above are filename-based interpretations, not undocumented identity annotations. |


## Preserved MetalParts experiment counts

| condition | source_fit | source_val | support | target_normal | target_anomalous | support_components | cross_role_components | forbidden_role_components | k20_and_10plus10 | failed_seeds | excluded_target_normals | excluded_target_anomalies |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test1 | 41 | 11 | 38 | 20 | 50 | 38 | 1 | 0 | PASS |  | 62 | 30 |
| test2 | 53 | 14 | 31 | 28 | 41 | 31 | 1 | 0 | PASS |  | 61 | 39 |
| test3 | 52 | 14 | 40 | 41 | 46 | 40 | 1 | 0 | PASS |  | 39 | 34 |
| test4 | 53 | 14 | 40 | 76 | 77 | 40 | 1 | 0 | PASS |  | 4 | 3 |
| test5 | 52 | 14 | 36 | 21 | 36 | 36 | 0 | 0 | PASS |  | 3 | 4 |
| test6 | 53 | 14 | 39 | 21 | 40 | 39 | 0 | 0 | PASS |  | 0 | 0 |

| Comparison | Finding |
|---|---|
| Old data_splits.csv | No differences in source-fit, source-val, support or held-out class counts. |
| Raw vs experiment counts | Raw official test folders include reserved supports and excluded duplicates; raw counts should not equal held-out test counts. |
| Sampling | Seeds 0..2 draws match historical files; seeds 3..4 extend the same normal pool. K=5 nested in K=20; controlled split first 10 adapt, last 10 calibrate. K=0 has no support rows. |
| Threshold pool | source_val role in MetalParts_roles.csv; never source replay. Five independent source-training seeds are future P2/P4 work. |
| Annotation barrier | Training roles CSV and draws have no defect labels/masks; final labels in separate evaluation_annotations.csv. Audit inventory is not a training manifest. |

## Duplicate/capture audit

| check | value |
|---|---|
| near_duplicate_pairs | 24065 |
| candidate_clusters | 164 |
| affected_official_test_images | 1374 |
| exact_byte_pairs | 1 |
| exact_rgb_pairs | 3 |
| cross_group_candidate_pairs | 0 |

| Evidence | Interpretation |
|---|---|
| near_duplicate_pairs.csv / near_duplicate_clusters.csv | Exhaustive within-group all-split pHash <=4 candidates and component member IDs; exact flags separate. Test affected means official test0..N, not necessarily experiment leakage. |
| cross_role_candidates.csv | Fit/calibration collisions block a condition. Links solely between source-test and target-test are evaluation-dependence warnings, retained for grouped uncertainty review; they do not expose test data to fitting. |
| Filenames | MetalParts names encode normal/defect text, shift and an integer; PCB names encode capture settings/index and sometimes Set1/Set2. No documented physical-object ID; do not parse defect words for sampling. |
| Capture identity | Similarity/filename IDs do not prove physical independence; global components identify possible dependence across condition comparisons. |

## PCB confirmation gate

| condition | raw_target_normal | raw_target_anomalous | anomaly_masks | mask_ok | source_fit_plus_val_components | eligible_target_normal_components | support_capacity_after_test_reserve | reviewed_independent_support | near_duplicate_pairs | candidate_clusters | target_images_affected | status | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test1 | 120 | 90 | 90 | True | 2 | 2 | 0 | NOT ESTABLISHED | 12723 | 2 | 206 | NO-GO | Conservative source/support component capacity fails; physical identity unverified |
| test2 | 120 | 60 | 60 | True | 2 | 1 | 0 | NOT ESTABLISHED | 12990 | 2 | 178 | NO-GO | Conservative source/support component capacity fails; physical identity unverified |
| test3 | 80 | 60 | 60 | True | 2 | 0 | 0 | NOT ESTABLISHED | 10436 | 2 | 135 | NO-GO | Conservative source/support component capacity fails; physical identity unverified |
| test4 | 80 | 60 | 60 | True | 2 | 2 | 0 | NOT ESTABLISHED | 10857 | 3 | 140 | NO-GO | Conservative source/support component capacity fails; physical identity unverified |
| test5 | 80 | 60 | 60 | True | 2 | 54 | 34 | NOT ESTABLISHED | 9340 | 17 | 70 | NO-GO | Conservative source/support component capacity fails; physical identity unverified |

| Decision | Consequence |
|---|---|
| NO-GO | No PCB model run. Report MetalParts development-only evidence; no unseen-domain confirmation or broad industrial generalization claim. Giant pHash clusters need independently justified capture review, not a relaxed threshold chosen to pass. |

## Reproduce and status

| Item | Value |
|---|---|
| Download | Existing Python: `python common/p1_download.py` (resumable; pinned listing and disk gate) |
| Audit | `python common/p1_audit.py` (CPU only; cached image hash inventory reused if present) |
| Hash rerun | The protected inventory is reused. Fresh audit output goes to comparison/verification/p1-audit. |
| Next | Stop before P2. Any blocked MetalParts condition needs explicit resolution before training. |


## Completion and budget

| Item | Verified status / estimate |
|---|---|
| P0/P1 | Complete on 2026-11-13; P2 not started. |
| Budget assumption | 25 working hours / three-day window at P0 start. Conservatively charge approximately 1.50 h including transfer waits: P0 0.15 h, P1 1.35 h; approximately 23.50 h remain. These are work-budget estimates, not measured GPU timings. |
| GPU | 0 h; no training or feature extraction. |
| Disk | 6,866,085,552 bytes in data/raw including Hub metadata; final recorded C: free 174,273,536,000 bytes. Network traffic was not measured; retries exceed unique payload bytes. |
| Acquisition | 6,863,601,708 unique selected-group bytes; 1,912 existing MetalParts files reused after pin/hash checks. PCB unique payload 6,179,226,271 bytes. README is separate. |
| Manifest verification | docs/p1-validation.json: 2,132 role rows, 1,350 support/calibration draw rows, 108 scheduled condition/seed arms, 959 unique MetalParts manifest images; five-seed nesting/disjointness PASS. |
| Audit caveat | 24,065 pHash candidate pairs are not 24,065 confirmed duplicate captures. Only 1 exact-byte pair and 3 exact-decoded-RGB pairs were found. Candidate grouping is deliberately conservative; physical-object identities remain unknown. |
| PCB | No condition meets the source-fit/validation capacity gate. test5 has target support capacity 34 under the conservative screen, but still fails the source gate. No PCB confirmation is authorized. |
| Replay inputs | Existing data/manifests/expanded/MetalParts/*/manifest.jsonl and support_draws.jsonl define immutable historical membership; retained locally. common/p1_audit.py checks their byte hashes against the pinned raw data. |
| Next | P2 implementation and CPU checks only after authorization. Do not attach GPU. |

## Final PCB gate after approved regrouping — 2026-11-14
Exact-byte/decoded duplicates define groups; no documented capture IDs found. pHash edges are review flags only, not transitive identity. Unresolved cross-role candidates remain quarantined under the approved proposal. No scores/features inspected. Gate run once; final NO-GO, development-only evidence.

| Condition | Exact source | Source after quarantine | Exact support | Support after quarantine | Unresolved groups | Final |
|---|---:|---:|---:|---:|---:|---|
|test1|180|7|40|25|597|NO-GO|
|test2|180|7|40|0|595|NO-GO|
|test3|180|7|40|1|541|NO-GO|
|test4|180|7|40|0|548|NO-GO|
|test5|180|7|40|40|466|NO-GO|
