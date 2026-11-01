[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23851748.svg)](https://doi.org/10.5281/zenodo.23851748)

# Data-integrity duplicate audit of the CP-AnemiC conjunctival anaemia dataset

Code, duplicate maps and out-of-fold predictions for the article
*"Class-conditional duplicates inflate cross-validated performance on a paediatric conjunctival anaemia dataset: a data-integrity audit of CP-AnemiC"*
(Calizaya Apaza F, Arce Apaza B, Apaza Quispe JH, Condori Coaquira AR; Universidad Peruana Unión).

The audit precedes a prospective study in children aged 6–59 months in Puno, Peru (3 800 m), which is described in the article.
No Peruvian data are included here.

## What is here

| Folder | Content |
|---|---|
| `codigo/` | Container pipeline (CPU): `common.py` (loading, dedup, folds, metrics, bootstrap CI), `classic.py` (63 colour features, nested repeated CV for LR/SVM/RF/XGBoost), `cnn.py` (TensorFlow/Keras CNNs + TFLite), `fusion.py`, `probe.py`, `hb_reg.py`, `xai_classic.py` (SHAP), `gradcam.py`, `make_table.py`, `who2024.py` |
| `gpu/` | PyTorch/CUDA replica used for the reported CNN results: `train_torch.py` (7 architectures, 2-phase fine-tuning, ONNX FP32/FP16/INT8 export, onnxruntime latency), `gradcam_torch.py`, `run_all.bat`, `run_seeds.bat`; post hoc: `train_torch_grouped.py` (grouped protocol, StratifiedGroupKFold on the 708 rows), `gradcam_review.py` (fold-wise Grad-CAM with in-mask ratio, bootstrap CI and Wilcoxon test), `run_review.bat`, `run_strict.bat`; their outputs are in `gpu/out_review/` (grouped CNNs, Grad-CAM) and `gpu/out_strict/` (seven CNNs re-run on the strict 452-image map); `codigo/review/integrate_gpu.py` merges them with the CPU results into `resultados/review/gpu_review.json` (leakage CIs, DeLong, Grad-CAM statistics by predicted class and without ratio-0 maps) ; the figure scripts of the current manuscript are in `codigo/review/figs/` (`make_fig1.py`, `make_fig2.py`, `make_fig6.py`, `make_figS7.py`, `make_figs_a.py`, `make_figs_b.py`) |
| `codigo/meta_groups.csv` | **The duplicate map.** One row per CP-AnemiC image: `image_id, image_path, anemia, sha, grp` (Hb, age, sex and facility of the children are not redistributed; see *Data*). Rows sharing `grp` are byte-identical or perceptual near-duplicates (pHash-16 with mirror, Hamming ≤ 20, connected components). 710 rows → 472 unique groups → 471 after removing one label-conflict group. |
| `resultados/` | `classic_results.csv`, `cnn_gpu_results.csv`, `fusion_results.csv`, `tabla_comparativa.csv`, `hb_reg.json`, `who2024.json`, `xai_classic.json`, SHAP and Grad-CAM figures, out-of-fold predictions (`*_oof.json` / `cnn_*_clean.json`) |
| `figuras/` | Figures 1–4 of the first draft (PNG + PDF); `figuras/review/` holds the figures of the current manuscript (main text: `Fig1_overview`, `Fig2_dedup_flow`, `Fig3_leakage`, `Fig5_shap`, `Fig6_gradcam`, printed as Figures 1–5; supplement: `figS3`–`figS8` and `Fig4_roc`, printed as Figure S9; `Fig7_size_vs_auc` is the same plot as `figS8_size_vs_auc`) |
| `codigo/review/` | Post hoc analyses after the first internal review (CPU): `review_analyses.py` (naive / grouped / deduplicated protocols with paired bootstrap and DeLong tests, per-repetition AUC, nested screening thresholds, calibration, PPV/NPV, sex subgroups, singleton-only and random-representative sensitivity), `review_analyses_b.py` (feature-space duplicate audit, 1-NN/5-NN, shape-only and brightness ablations, leave-one-hospital-out with random effects, DeLong + Holm comparisons against the CNNs, SHAP), `review_strict.py` (strict 452-image set), `figs/` (figure scripts) |
| `codigo/dedup_audit_v2.py` | **Third stage of the duplicate audit (post hoc).** Perceptual hash under the eight dihedral transforms (rotations and mirror) plus feature-vector identity. It found 19 copies rotated by 90°/180°/270° that the mirror-only hash had missed. |
| `codigo/meta_groups_strict.csv` | The strict duplicate map produced by `dedup_audit_v2.py`: 710 rows → 453 components → 452 after removing the label-conflict group (218 anaemic / 234 non-anaemic). The pre-specified analyses use `meta_groups.csv` (471); the strict map is used in the sensitivity analysis. |
| `codigo/review2/` | Post hoc analyses after the second internal review (CPU): `review2_stats.py` (class-conditional duplication with permutation tests, Obuchowski clustered-ROC and cluster-bootstrap tests, paired 471-vs-452 comparison, PPV/NPV at 42/47/70 % prevalence, REML + Hartung–Knapp meta-analysis of leave-one-hospital-out, Brier skill score, Riley et al. sample size), `review2_images.py` (Hamming-distance histogram, Grad-CAM ceiling at 7 × 7, mask-area classifier, feature stability ICC under JPEG / rescaling / white balance / mask erosion, singleton-only LOHO), `review2_orb.py` (**fourth audit stage**: SIFT keypoints + RANSAC on every pair of the 472 representatives of the pre-specified map, mirror included), `review2_geomverify.py` (photometric verification: NCC of the aligned green channels and overlap), `review2_build_geo_map.py` (accept rule and union with the strict groups), `review2_geo.py` (all classical analyses on the geometric set and grouped protocol with the geometric groups), `review2_geo_clustered.py` (cluster-aware tests with the geometric groups as clusters), `review2_montage_pairs.py` (contact sheets for visual checks) |
| `codigo/meta_groups_geo.csv` | **The geometric duplicate map** (fourth stage): 710 rows → 338 distinct images → 337 for classification (179 anaemic / 158 non-anaemic). Accept rule: ≥ 10 RANSAC inliers, NCC ≥ 0.70, overlap ≥ 0.30, scale 0.5–2. |
| `resultados/review2/` | Result files of the second round (`review2_stats.json`, `review2_images.json`, `geo_map_summary.json`, `orb_results.json`, candidate pairs `orb_pairs.npy` and verification `geomverify.npy`, `clustered_geo.json`, `geo/` for the geometric-set models, `gpu/` for Grad-CAM of the predicted class with occlusion, `gpu2/` for the CNNs on the geometric set, the tuned recipe on the 471-image and geometric sets (seeds 42, 1 and 2) and the INT8 configurations) |
| `gpu/review2_cnn.py`, `gpu/gradcam_review2.py` | GPU scripts of the second round: CNNs on the geometric set, tuned recipe (mask bounding-box input, trainable BatchNorm, learning rate chosen in the inner split), INT8 export with the normalisation outside the graph and first/last layers in FP32; Grad-CAM of the predicted class with the 7 × 7 ceiling and tissue occlusion (`run_review2_cnn.bat`, `run_gradcam2.bat`) |
| `codigo/review3/` | Post hoc analyses after the third internal review (CPU): `review3_geomverify_lo.py` and `review3_thresholds.py` (sensitivity of the count of distinct images to the NCC and inlier thresholds, and an upper bound on false links from label concordance), `review3_cluster_cis.py` (cluster-bootstrap CIs and Obuchowski tests for Table 2 and Figure 3), `review3_cpu.py` (the two duplication patterns, paired comparison on the 337 images common to the 471-image and geometric sets, size control, development sample size by Riley et al. 2020, Random Forest against the tuned CNNs with cluster bootstrap, DeLong and Holm, eroded-mask pipeline on perturbed images, leave-one-hospital-out on the geometric set and its singletons), `review3_size2.py` (size control with random removal, XGBoost and Random Forest), `review3_ensemble.py` (AUC of the run-averaged probabilities, the estimand of the DeLong contrast), `run_all_review3.sh` |
| `resultados/review3/` | Result files of the third round (`review3.json`, `cluster_cis.json`, `thresholds.json`, `size2.json`, `geomverify_lo.npy`, Random Forest out-of-fold predictions per repetition `rf_rep471.npy` and `rf_repgeo.npy`) |
| `gpu/run_review3_cnn.bat` | GPU batch of the third round: tuned recipe on the geometric set (`review2_cnn.py tunedgeo`, three architectures × seeds 42, 1 and 2) and seeds 1 and 2 of MobileNetV3-Large with the tuned recipe; outputs in `resultados/review2/gpu2/` |
| `resultados/review/` | Result files of the review analyses (`review_results.json`, `review_results_b.json`, `review_strict.json`, out-of-fold predictions `oof_review.json` / `oof_strict.json`, `dihedral_links.json`, `strict_pairs.json`). DeLong tests use the fast method of Sun & Xu (2014) with ranks over the positives-first ordering; 33 pairwise comparisons, Holm-adjusted |
| `codigo/group_discordance.csv` | For every multi-row group of the three maps: number of rows, of distinct files and of distinct Hb, age, sex, facility and label values (the counts behind the metadata-discordance results, without the values themselves) |
| `codigo/rebuild_meta.py` | Restores the Hb, age, sex and facility columns of the maps from your own copy of CP-AnemiC v1 (checks every image digest first); the analysis scripts that use those columns need it |
| `codigo/review5/` | Post hoc analyses after the fifth internal review (CPU): `review5_synthetic.py` (recall of every audit stage on synthetic copies of known identity: mirror, rotations, rescaling, JPEG, re-cropping, re-segmentation, combined edits; spurious links to other images), `review5_cpu.py` (tree ensembles on the fold seeds of the CNN runs, hue statistics on a rotated axis, grouped SHAP), `review5_post.py` (stage-4 funnel, label concordance of rejected look-alike stage-4 pairs and the false-link bound, DeLong tests between single runs on identical folds, Nadeau–Bengio corrected resampled t test, leave-one-hospital-out pooled on the logit scale), `review5_kit.py` (builds a blinded two-rater inspection kit of 70 coded pairs: accepted stage-4 links, rejected look-alike candidates and random pairs; it needs the images, so the kit itself is not distributed), `kappa_inspection.py` (Cohen's kappa between the raters and against the detector), `run_all_review5.sh`; results in `resultados/review5/` |
| `codigo/review6/` | Post hoc analyses after the sixth internal review (CPU): `review6_geo.py` (the geometric set as the principal set: metrics at 0.5, Brier skill, calibration slope and intercept for the classical models and every CNN run, decision curves on the geometric and 471-image sets, Figures S4 and S6), `review6_lum.py` (luminance-only ablation with the 25 luminance statistics; an earlier selection by channel prefix also took R/G), `run_all_review6.sh`; results in `resultados/review6/`. The corrected stage-4 funnel, which counts the 9,891 candidates and the 1,000 random control pairs separately, is the key `funnel` of `resultados/review5/post.json` |
| `gpu/run_resnet50_naive.bat` | Optional GPU run of ResNet50, the architecture of the original CP-AnemiC paper, under the naive protocol (seed 42); output `gpu/out/cnn_ResNet50_naive.json` |
| `sha256sums.txt` | SHA-256 digest of every CP-AnemiC v1 file we analysed, so that readers can check whether their copy is the same data |

## Data

CP-AnemiC is not redistributed here. It was published on Mendeley Data: Appiahene P, Asare JW, Donkoh ET. CP-AnemiC. https://doi.org/10.17632/m53vz6b7fx.1; the manifest of our local ingestion (28 May 2026) records the CC BY 4.0 licence shown on the record at the time. Both versions of the record have since been withdrawn by the depositors without a stated reason, so the images can no longer be downloaded there. We analysed version 1 (downloaded on or before 28 May 2026) and publish the SHA-256 digest of every file in `sha256sums.txt`. Because the reason for the withdrawal is unknown, the per-child clinical metadata (Hb, age, sex, facility) are **not** redistributed: the maps keep the image identifier, digest, binary anaemia label and group, and `codigo/group_discordance.csv` gives the discordance counts. If you hold a copy, point `DATA_DIR` to the extracted folder (expects `metadata.csv` and `images/`), compare the digests, and run `python codigo/rebuild_meta.py <folder>` to restore the full columns locally (do not commit them). An earlier Mendeley Data record by the same authors (https://doi.org/10.17632/nt7r8hv2pz.1) describes 710 conjunctival images of children from ten Ghanaian hospitals; we have not examined it.

## Reproduce

```bash
# 1. environments, pinned to the exact versions that produced the results
#    container (Linux x86-64, Python 3.13.18): duplicate audit, colour features, classical models, statistics, figures
pip install -r requirements-lock.txt
#    GPU (Windows 11, Python 3.12.10, RTX 4060 Laptop GPU, CUDA 12.4): CNNs, ONNX export, Grad-CAM
pip install -r gpu/requirements_gpu_lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
#    (requirements.txt and gpu/requirements_gpu.txt list the same packages without pins)

# 2. duplicate audit -> meta_groups.csv (already provided; regenerate with)
python codigo/dedup_audit.py

# 3. colour features + classical models, both protocols (clean = deduplicated, naive = with duplicates)
python codigo/classic.py

# 4. CNNs on GPU (one call per architecture; seed 42 exports ONNX variants and measures latency)
cd gpu && run_all.bat            # or: python train_torch.py EfficientNet-B0 clean
set SEED=1 && python train_torch.py EfficientNet-B0 clean   # extra seeds for between-seed SD

# 5. explainability, regression, WHO-2024 sensitivity, tables and figures
python codigo/xai_classic.py; python gpu/gradcam_torch.py EfficientNet-B0
python codigo/hb_reg.py; python codigo/who2024.py; python codigo/make_table.py; python codigo/figures.py

# 6. post hoc analyses (internal review rounds), in this order; run codigo/rebuild_meta.py first
bash codigo/review/run_all_review.sh                 # protocols, feature-space audit, LOHO, strict set
bash codigo/review2/run_all_review2.sh               # statistics, image analyses, fourth audit stage, geometric set
cd gpu && run_review.bat && run_strict.bat && run_gradcam2.bat && run_review2_cnn.bat && run_review3_cnn.bat   # GPU runs
bash codigo/review3/run_all_review3.sh               # third round: thresholds, cluster CIs, size control, contrasts, LOHO
bash codigo/review5/run_all_review5.sh               # fifth round: detector validation, matched single runs, rotated hue, grouped SHAP
bash codigo/review6/run_all_review6.sh               # sixth round: geometric set as principal set, decision curves, luminance-25 ablation
```

Protocol fixed in `common.py`: `SEED = 42`, stratified 5-fold CV (repeated 5× for classical models), nested
`GridSearchCV` for hyper-parameters, 15 % internal validation split for CNN early stopping, 2 000-sample bootstrap
95 % CIs, metrics at threshold 0.5 (AUC, F1, accuracy, sensitivity, specificity, Brier).
Latency: onnxruntime CPU, 1 intra-op thread, median of 100 single-image runs after 10 warm-ups.

## Key numbers

| Set (audit stage) | Images | Random Forest, colour (AUC per repetition) | XGBoost, colour | Best CNN, original recipe (mean of 3 seeds) | Best CNN, tuned recipe (mean of 3 seeds) |
|---|---|---|---|---|---|
| Pre-specified (stages 1–2) | 471 | 0.857 ± 0.010 | 0.845 ± 0.010 | 0.765 (EfficientNetV2-S) | 0.794 (EfficientNet-B0) |
| Strict (stage 3, post hoc) | 452 | 0.832 ± 0.007 | 0.813 ± 0.011 | 0.710 (EfficientNet-B0) | not run |
| Geometric (stage 4, post hoc) | 337 | 0.741 ± 0.010 | 0.718 ± 0.012 | 0.620 (EfficientNetV2-S) | 0.678 (EfficientNet-B0) |

- CP-AnemiC v1 holds 338 distinct images in 710 rows; copies stay within one anaemia class far more often than chance (permutation p < 0.0001) although their Hb, age, sex and facility differ.
- On the same 708 rows, keeping duplicate groups in one fold lowers the AUC of the colour models by 0.05–0.07 with the pre-specified map and by 0.18–0.22 with the geometric map (cluster bootstrap CIs exclude zero).
- The duplicates follow two class-dependent patterns: hash-detected copies are mostly anaemic (81 % of duplicated rows of the pre-specified map, against 42 % of singletons), whereas the images of the same eye found only by the keypoint search are mostly non-anaemic (131 of 182).
- The count of 338 distinct images ranges from 327 to 355 over a 3 × 3 grid of NCC (0.6–0.8) and RANSAC-inlier (8–15) thresholds; none of the 168 new pairs has discordant labels, which bounds the false-link rate at 3.5 % (one-sided 95 %) if a false link agrees by chance. Rejected look-alike pairs agree on the label at chance level (about 50 %), and on synthetic copies of known identity the detector produced no spurious link in 18 535 comparisons while missing most heavily edited copies, so 338 is more likely an overestimate of the distinct images than an underestimate.
- Size control: cutting the 471-image set to 337 images by removing only images without a near-duplicate keeps the AUC high (Random Forest 0.886, XGBoost 0.871), against 0.741 and 0.718 on the geometric set; removing images at random gives 0.811 and 0.792, so the smaller sample accounts for at most 0.046 and 0.052 of the falls of 0.116 and 0.127. On the 337 common images, models trained on the 471-image folds score 0.07–0.09 higher.
- A tuned CNN recipe (mask bounding-box input, trainable BatchNorm, learning rate chosen per fold) raises EfficientNet-B0 from 0.73 to 0.79, MobileNetV2 from 0.65 to 0.78 and MobileNetV3-Large from 0.51 to 0.75 on the pre-specified set, and EfficientNet-B0 from 0.59 to 0.68 on the geometric set (three seeds each). Single runs stay below Random Forest on both sets (cluster bootstrap, Holm p ≤ 0.024); a three-seed ensemble of the tuned EfficientNet-B0 comes within 0.03 of the averaged Random Forest (DeLong, Holm p = 0.11 and 0.36).
- With the 2-px eroded mask, Random Forest keeps AUC 0.82 on JPEG-75 and 0.86 on rescaled test images (original mask: 0.61 and 0.71); neither mask protects against a 5 % white-balance shift (0.75–0.76). Use the eroded mask with these features.
- The manuscript reports discrimination, calibration and operating points on the geometric set (337 images), the most complete audit, as a declared deviation from the pre-specified 471-image set. There, net benefit exceeds that of testing every child only at threshold probabilities of 0.30 or more; at the screening thresholds 0.10–0.20 the models add nothing over testing everyone.
- Post-training static INT8 loses discrimination for EfficientNet-B0 and MobileNetV2 even with the normalisation outside the graph and the first and last layers in FP32; FP16 is lossless.

## Citation

Archived on Zenodo: **https://doi.org/10.5281/zenodo.23851748** (release v1.0.2; concept DOI for all versions: https://doi.org/10.5281/zenodo.23851747). See `CITATION.cff` for the full citation.

## Licence

Code: MIT. Duplicate maps and derived tables: CC BY 4.0 (derived from CP-AnemiC, distributed under CC BY 4.0 when downloaded; Appiahene et al. 2023).

## Reporting checklist

`resultados/S1_reporting_map_current.csv` maps each CLAIM 2024 / TRIPOD+AI reporting topic to the section of the current
manuscript that addresses it (`S1_reporting_map.csv` is the first-draft version it was derived from). Transfer it to the official forms before submission: CLAIM 2024 (Tejani et al., Radiol Artif Intell
2024;6(4):e240300, https://doi.org/10.1148/ryai.240300) and TRIPOD+AI (Collins et al., BMJ 2024;385:e078378).

## Paths

The container scripts in `codigo/` use absolute paths (`/home/claude/data/cp-anemic/`, `/home/claude/exp/out/`,
`/home/claude/sr/meta_groups.csv`) as they were run; edit `D`, `O` and the `meta_groups.csv` path in `common.py`
(or set `DATA_DIR` for the GPU scripts) to your own locations. `gpu/` scripts read `DATA_DIR` and write to `gpu/out/`.
