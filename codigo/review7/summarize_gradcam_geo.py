"""Summarise the 12 occlusion / Grad-CAM runs on D-337 (teacher's observation 1).

Reads   resultados/review7/gradcam_geo_<model>_<recipe>_s<seed>.json   (written by gpu/gradcam_geo.py)
Writes  resultados/review7/gradcam_geo_runs.csv      one row per run (12 rows)
        resultados/review7/gradcam_geo_summary.csv   one row per model and recipe (mean, SD and range of the three seeds)

Only aggregate numbers are read or written: no per-image predictions, labels or images.
Usage: python codigo/review7/summarize_gradcam_geo.py
"""
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', '..', 'resultados', 'review7')
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'gradcam_geo_*_s*.json')))]
assert len(runs) == 12, len(runs)

rows = []
for r in runs:
    p, a = r['predicted_class'], r['anaemia_logit']
    rows.append(dict(model=r['model'], recipe=r['recipe'], seed=r['seed'], n=r['n'],
                     auc_intact=r['AUC_oof_fp32'], auc_silhouette=r['AUC_silhouette'], auc_meanfill=r['AUC_meanfill'],
                     drop_silhouette=r['AUC_oof_fp32'] - r['AUC_silhouette'], drop_meanfill=r['AUC_oof_fp32'] - r['AUC_meanfill'],
                     auc_mask_area=r['AUC_mask_area'], mask_area_mean=r['mask_area_mean'],
                     ceiling_median=r['ceiling_median'], ceiling_p10=r['ceiling_p10'], ceiling_p90=r['ceiling_p90'],
                     ratio_pred_mean=p['ratio_mean'], ratio_pred_ci_lo=p['ratio_ci'][0], ratio_pred_ci_hi=p['ratio_ci'][1],
                     ratio_pred_frac_gt1=p['frac_ratio_gt1'], ratio_pred_wilcoxon_p=p['wilcoxon_p_greater_than_1'],
                     ratio_pred_over_ceiling=p['ratio_mean'] / r['ceiling_median'],
                     ratio_anaemia_logit_mean=a['ratio_mean'], ratio_anaemia_logit_frac_gt1=a['frac_ratio_gt1']))
R = pd.DataFrame(rows).sort_values(['model', 'recipe', 'seed']).reset_index(drop=True)
R.to_csv(os.path.join(RES, 'gradcam_geo_runs.csv'), index=False, float_format='%.4f')

out = []
for (mo, re_), g in R.groupby(['model', 'recipe'], sort=False):
    d = dict(model=mo, recipe=re_, seeds=len(g))
    for c in ['auc_intact', 'auc_silhouette', 'auc_meanfill', 'drop_silhouette', 'drop_meanfill', 'ratio_pred_mean',
              'ratio_pred_frac_gt1', 'ratio_pred_over_ceiling', 'ratio_anaemia_logit_mean']:
        v = g[c].to_numpy()
        d[c + '_mean'] = v.mean(); d[c + '_sd'] = v.std(ddof=1); d[c + '_min'] = v.min(); d[c + '_max'] = v.max()
    d['mask_area_mean'] = g['mask_area_mean'].iloc[0]; d['ceiling_median'] = g['ceiling_median'].iloc[0]
    d['max_wilcoxon_p'] = g['ratio_pred_wilcoxon_p'].max()
    out.append(d)
S = pd.DataFrame(out)
S.to_csv(os.path.join(RES, 'gradcam_geo_summary.csv'), index=False, float_format='%.4f')

pd.set_option('display.width', 250)
print(S[['model', 'recipe', 'auc_intact_mean', 'auc_intact_sd', 'auc_silhouette_mean', 'auc_silhouette_sd', 'auc_meanfill_mean', 'auc_meanfill_sd']].round(3).to_string(index=False))
print(S[['model', 'recipe', 'drop_silhouette_min', 'drop_silhouette_max', 'drop_meanfill_min', 'drop_meanfill_max', 'ratio_pred_mean_mean', 'ratio_pred_frac_gt1_mean',
         'ratio_pred_over_ceiling_mean', 'ratio_anaemia_logit_mean_mean', 'mask_area_mean', 'ceiling_median', 'max_wilcoxon_p']].round(3).to_string(index=False))
