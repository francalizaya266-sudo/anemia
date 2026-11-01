[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23851748.svg)](https://doi.org/10.5281/zenodo.23851748)

# Forensic duplicate audit of the CP-AnemiC conjunctival anaemia dataset

Code, duplicate maps and out-of-fold predictions for the article
*"Class-conditional duplicate images inflate conjunctival anaemia detection in children: a forensic audit of the CP-AnemiC dataset"*
(Calizaya Apaza F, Arce Apaza B, Apaza Quispe JH, Condori Coaquira AR; Universidad Peruana Unión).

The audit precedes a prospective study in children aged 6–59 months in Puno, Peru (3 800 m), which is described in the article.
No Peruvian data are included here.

## What is here

| Folder | Content |
|---|---|
| `codigo/` | Container pipeline (CPU): `common.py` (loading, dedup, folds, metrics, bootstrap CI), `classic.py` (63 colour features, nested repeated CV for LR/SVM/RF/XGBoost), `cnn.py` (TensorFlow/Keras CNNs + TFLite), `fusion.py`, `probe.py`, `hb_reg.py`, `xai_classic.py` (SHAP), `gradcam.py`, `make_table.py`, `who2024.py` |
| `gpu/` | PyTorch/CUDA replica used for the reported CNN results: `train_torch.py` (7 architectures, 2-phase fine-tuning, ONNX FP32/FP16/INT8 export, onnxruntime latency), `gradcam_torch.py`, `run_all.bat`, `run_seeds.bat`; post hoc: `train_torch_grouped.py` (grouped protocol, StratifiedGroupKFold on the 708 rows), `gradcam_review.py` (fold-wise Grad-CAM with in-mask ratio, bootstrap CI and Wilcoxon test), `run_review.bat`, `run_strict.bat`; their outputs are in `gpu/out_review/` (grouped CNNs, Grad-CAM) and `gpu/out_strict/` (seven CNNs re-run on the strict 452-image map); `codigo/review/integrate_gpu.py` merges them with the CPU results into `resultados/review/gpu_review.json` (leakage CIs, DeLong, Grad-CAM statistics by predicted class and without ratio-0 maps) ; the figure scripts of the current manuscript are in `codigo/review/figs/` (`make_fig1.py`, `make_fig2.py`, `make_fig6.py`, `make_figS7.py`, `make_figs_a.py`, `make_figs_b.py`) |
| `codigo/meta_groups.csv` | **The duplicate map.** One row per CP-AnemiC image: `image_id, image_path, hb, anemia, severity, hospital, age_months, gender, sha, grp`. Rows sharing `grp` are byte-identical or perceptual near-duplicates (pHash-16 with mirror, Hamming ≤ 20, connected components). 710 rows → 472 unique groups → 471 after removing one label-conflict group. |
| `resultados/` | `classic_results.csv`, `cnn_gpu_results.csv`, `fusion_results.csv`, `tabla_comparativa.csv`, `hb_reg.json`, `who2024.json`, `xai_classic.json`, SHAP and Grad-CAM figures, out-of-fold predictions (`*_oof.json` / `cnn_*_clean.json`) |
| `figuras/` | Figures 1–4 of the first draft (PNG + PDF); `figuras/review/` holds the figures of the current manuscript (Fig. 1–7, S3–S7) |
| `codigo/review/` | Post hoc analyses after the first internal review (CPU): `review_analyses.py` (naive / grouped / deduplicated protocols with paired bootstrap and DeLong tests, per-repetition AUC, nested screening thresholds, calibration, PPV/NPV, sex subgroups, singleton-only and random-representative sensitivity), `review_analyses_b.py` (feature-space duplicate audit, 1-NN/5-NN, shape-only and brightness ablations, leave-one-hospital-out with random effects, DeLong + Holm comparisons against the CNNs, SHAP), `review_strict.py` (strict 452-image set), `figs/` (figure scripts) |
| `codigo/dedup_audit_v2.py` | **Third stage of the duplicate audit (post hoc).** Perceptual hash under the eight dihedral transforms (rotations and mirror) plus feature-vector identity. It found 19 copies rotated by 90°/180°/270° that the mirror-only hash had missed. |
| `codigo/meta_groups_strict.csv` | The strict duplicate map produced by `dedup_audit_v2.py`: 710 rows → 453 components → 452 after removing the label-conflict group (218 anaemic / 234 non-anaemic). The pre-specified analyses use `meta_groups.csv` (471); the strict map is used in the sensitivity analysis. |
| `codigo/review2/` | Post hoc analyses after the second internal review (CPU): `review2_stats.py` (class-conditional duplication with permutation tests, Obuchowski clustered-ROC and cluster-bootstrap tests, paired 471-vs-452 comparison, PPV/NPV at 42/47/70 % prevalence, REML + Hartung–Knapp meta-analysis of leave-one-hospital-out, Brier skill score, Riley et al. sample size), `review2_images.py` (Hamming-distance histogram, Grad-CAM ceiling at 7 × 7, mask-area classifier, feature stability ICC under JPEG / rescaling / white balance / mask erosion, singleton-only LOHO), `review2_orb.py` (**fourth audit stage**: SIFT keypoints + RANSAC on every pair of the 472 representatives of the pre-specified map, mirror included), `review2_geomverify.py` (photometric verification: NCC of the aligned green channels and overlap), `review2_build_geo_map.py` (accept rule and union with the strict groups), `review2_geo.py` (all classical analyses on the geometric set and grouped protocol with the geometric groups), `review2_geo_clustered.py` (cluster-aware tests with the geometric groups as clusters), `review2_montage_pairs.py` (contact sheets for visual checks) |
| `codigo/meta_groups_geo.csv` | **The geometric duplicate map** (fourth stage): 710 rows → 338 distinct photographs → 337 for classification (179 anaemic / 158 non-anaemic). Accept rule: ≥ 10 RANSAC inliers, NCC ≥ 0.70, overlap ≥ 0.30, scale 0.5–2. |
| `resultados/review2/` | Result files of the second round (`review2_stats.json`, `review2_images.json`, `geo_map_summary.json`, `orb_results.json`, candidate pairs `orb_pairs.npy` and verification `geomverify.npy`, `clustered_geo.json`, `geo/` for the geometric-set models, `gpu/` for Grad-CAM of the predicted class with occlusion, `gpu2/` for the CNNs on the geometric set, the tuned recipe and the INT8 configurations) |
| `gpu/review2_cnn.py`, `gpu/gradcam_review2.py` | GPU scripts of the second round: CNNs on the geometric set, tuned recipe (mask bounding-box input, trainable BatchNorm, learning rate chosen in the inner split), INT8 export with the normalisation outside the graph and first/last layers in FP32; Grad-CAM of the predicted class with the 7 × 7 ceiling and tissue occlusion (`run_review2_cnn.bat`, `run_gradcam2.bat`) |
| `resultados/review/` | Result files of the review analyses (`review_results.json`, `review_results_b.json`, `review_strict.json`, out-of-fold predictions `oof_review.json` / `oof_strict.json`, `dihedral_links.json`, `strict_pairs.json`). DeLong tests use the fast method of Sun & Xu (2014) with ranks over the positives-first ordering; 33 pairwise comparisons, Holm-adjusted |
| `sha256sums.txt` | SHA-256 digest of every CP-AnemiC v1 file we analysed, so that readers can check whether their copy is the same data |

## Data

CP-AnemiC is not redistributed here. It was published on Mendeley Data (CC BY 4.0): Appiahene P, Asare JW, Donkoh ET. CP-AnemiC. https://doi.org/10.17632/m53vz6b7fx.1. Both versions of the record have since been withdrawn by the depositors; we analysed version 1 (downloaded before the withdrawal) and publish the SHA-256 digest of every file in `sha256sums.txt`. If you hold a copy, point `DATA_DIR` to the extracted folder (expects `metadata.csv` and `images/`) and compare the digests.

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

# 6. post hoc analyses (first and second internal review), in this order
bash codigo/review/run_all_review.sh                 # protocols, feature-space audit, LOHO, strict set
bash codigo/review2/run_all_review2.sh               # statistics, image analyses, fourth audit stage, geometric set
cd gpu && run_review.bat && run_strict.bat && run_gradcam2.bat && run_review2_cnn.bat   # GPU runs of both rounds
```

Protocol fixed in `common.py`: `SEED = 42`, stratified 5-fold CV (repeated 5× for classical models), nested
`GridSearchCV` for hyper-parameters, 15 % internal validation split for CNN early stopping, 2 000-sample bootstrap
95 % CIs, metrics at threshold 0.5 (AUC, F1, accuracy, sensitivity, specificity, Brier).
Latency: onnxruntime CPU, 1 intra-op thread, median of 100 single-image runs after 10 warm-ups.

## Key numbers

| Set (audit stage) | Images | Random Forest, colour (AUC per repetition) | XGBoost, colour | Best CNN (mean of 3 seeds, original recipe) |
|---|---|---|---|---|
| Pre-specified (stages 1–2) | 471 | 0.857 ± 0.010 | 0.845 ± 0.010 | 0.765 (EfficientNetV2-S) |
| Strict (stage 3, post hoc) | 452 | 0.832 ± 0.007 | 0.813 ± 0.011 | 0.710 (EfficientNet-B0) |
| Geometric (stage 4, post hoc) | 337 | 0.741 ± 0.010 | 0.718 ± 0.012 | 0.620 (EfficientNetV2-S) |

- CP-AnemiC v1 holds 338 distinct photographs in 710 rows; copies stay within one anaemia class far more often than chance (permutation p < 0.0001) although their Hb, age, sex and facility differ.
- On the same 708 rows, keeping duplicate groups in one fold lowers the AUC of the colour models by 0.05–0.07 with the pre-specified map and by 0.18–0.22 with the geometric map (cluster bootstrap CIs exclude zero).
- A tuned CNN recipe (mask bounding-box input, trainable BatchNorm, learning rate chosen per fold) raises EfficientNet-B0 from 0.73 to 0.79 and MobileNetV2 from 0.65 to 0.78 on the pre-specified set (three seeds), still below Random Forest (0.857).
- Post-training static INT8 loses discrimination for EfficientNet-B0 and MobileNetV2 even with the normalisation outside the graph and the first and last layers in FP32; FP16 is lossless.

## Citation

Archived on Zenodo: **https://doi.org/10.5281/zenodo.23851748** (release v1.0.2; concept DOI for all versions: https://doi.org/10.5281/zenodo.23851747). See `CITATION.cff` for the full citation.

## Licence

Code: MIT. `meta_groups.csv` and derived tables: CC BY 4.0 (derived from CP-AnemiC, CC BY 4.0, Appiahene et al. 2023).

## Reporting checklist

`resultados/S1_reporting_map.csv` maps each CLAIM 2024 / TRIPOD+AI reporting topic to the manuscript section that
addresses it. Transfer it to the official forms before submission: CLAIM 2024 (Tejani et al., Radiol Artif Intell
2024;6(4):e240300, https://doi.org/10.1148/ryai.240300) and TRIPOD+AI (Collins et al., BMJ 2024;385:e078378).

## Paths

The container scripts in `codigo/` use absolute paths (`/home/claude/data/cp-anemic/`, `/home/claude/exp/out/`,
`/home/claude/sr/meta_groups.csv`) as they were run; edit `D`, `O` and the `meta_groups.csv` path in `common.py`
(or set `DATA_DIR` for the GPU scripts) to your own locations. `gpu/` scripts read `DATA_DIR` and write to `gpu/out/`.
