[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23851748.svg)](https://doi.org/10.5281/zenodo.23851748)

# Deduplicated benchmark for smartphone conjunctival anaemia detection in children (CP-AnemiC)

Code, duplicate map and out-of-fold predictions for the article
*"Data leakage inflates smartphone conjunctival pallor anaemia detection in children: a deduplicated benchmark of colour features versus lightweight CNNs on CP-AnemiC"*
(Calizaya Apaza F, Arce Apaza B, Apaza Quispe JH, Condori Coaquira AR; Universidad Peruana Unión).

This is the development phase of a system intended for children aged 6–59 months in Puno, Peru (3 800 m).
The prospective Puno protocol is described in the article and in `protocolo_puno/`.

## What is here

| Folder | Content |
|---|---|
| `codigo/` | Container pipeline (CPU): `common.py` (loading, dedup, folds, metrics, bootstrap CI), `classic.py` (63 colour features, nested repeated CV for LR/SVM/RF/XGBoost), `cnn.py` (TensorFlow/Keras CNNs + TFLite), `fusion.py`, `probe.py`, `hb_reg.py`, `xai_classic.py` (SHAP), `gradcam.py`, `make_table.py`, `who2024.py` |
| `gpu/` | PyTorch/CUDA replica used for the reported CNN results: `train_torch.py` (7 architectures, 2-phase fine-tuning, ONNX FP32/FP16/INT8 export, onnxruntime latency), `gradcam_torch.py`, `run_all.bat`, `run_seeds.bat` |
| `codigo/meta_groups.csv` | **The duplicate map.** One row per CP-AnemiC image: `image_id, image_path, hb, anemia, severity, hospital, age_months, gender, sha, grp`. Rows sharing `grp` are byte-identical or perceptual near-duplicates (pHash-16 with mirror, Hamming ≤ 20, connected components). 710 rows → 472 unique groups → 471 after removing one label-conflict group. |
| `resultados/` | `classic_results.csv`, `cnn_gpu_results.csv`, `fusion_results.csv`, `tabla_comparativa.csv`, `hb_reg.json`, `who2024.json`, `xai_classic.json`, SHAP and Grad-CAM figures, out-of-fold predictions (`*_oof.json` / `cnn_*_clean.json`) |
| `figuras/` | Figures 1–4 of the article (PNG + PDF) |

## Data

CP-AnemiC is not redistributed here. Download it from Mendeley Data (CC BY 4.0):
Appiahene P, Asare JW, Donkoh ET. CP-AnemiC. https://doi.org/10.17632/m53vz6b7fx.1
and point `DATA_DIR` to the extracted folder (expects `metadata.csv` and `images/`).

## Reproduce

```bash
# 1. environment (Python 3.11/3.12)
pip install -r requirements.txt          # numpy pandas scikit-learn xgboost opencv-python-headless imagehash shap matplotlib
pip install -r gpu/requirements_gpu.txt  # torch torchvision onnx onnxruntime onnxconverter-common (GPU runs)

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
```

Protocol fixed in `common.py`: `SEED = 42`, stratified 5-fold CV (repeated 5× for classical models), nested
`GridSearchCV` for hyper-parameters, 15 % internal validation split for CNN early stopping, 2 000-sample bootstrap
95 % CIs, metrics at threshold 0.5 (AUC, F1, accuracy, sensitivity, specificity, Brier).
Latency: onnxruntime CPU, 1 intra-op thread, median of 100 single-image runs after 10 warm-ups.

## Key numbers (deduplicated protocol, n = 471)

| Model | AUC (95 % CI) | Size | Latency (1 thread) |
|---|---|---|---|
| Random Forest, colour | 0.871 (0.837–0.903) | 3.8 MB | 30 ms |
| XGBoost, colour | 0.863 (0.830–0.895) | 0.4 MB | 6 ms |
| EfficientNet-B0 FP32 | 0.734 (0.69–0.78) | 15.3 MB | 9.6 ms |
| MobileNetV2 FP32 | 0.664 (0.61–0.71) | 8.5 MB | 6.8 ms |

With duplicates (naive protocol) the same models reach 0.95 / 0.94 / 0.86 / 0.83.

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
