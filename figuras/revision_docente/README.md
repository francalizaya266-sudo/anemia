# Figures added after the course adviser's review

Scripts of the two figures that the revision adds. They read only aggregate result files of this repository
(no images and no per-child data) and write PDF and PNG files to `es/` (Spanish labels, the default) or `en/`
(set `FIGLANG=en`).

| Script | Figure | Inputs |
|---|---|---|
| `fig8_secondary.py` | Figure 8 of the main text: secondary analyses on D-337 (single-run AUC of every model, tissue occlusion, Grad-CAM in-mask ratio) | `resultados/review6/geo_primary.json`, `resultados/review7/review7_shape.json`, `resultados/review7/gradcam_geo_*.json` |
| `figS10_shap_sets.py` | Supplementary Figure S10: SHAP of XGBoost on D-471 and D-337 | `resultados/review7/review7_shap.json` |

`figstyle.py` holds the shared style (one typeface, one palette with fixed meanings, 16 cm page width).

```bash
python figuras/revision_docente/fig8_secondary.py
FIGLANG=en python figuras/revision_docente/figS10_shap_sets.py
```
