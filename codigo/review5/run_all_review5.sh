#!/bin/bash
# Fifth internal review (data-integrity audit), in dependency order. Run after codigo/review3/run_all_review3.sh.
# Scripts are run from /home/claude/exp, as in earlier rounds; edit the paths for your setup.
set -e
cd /home/claude/exp; mkdir -p out/review5/kit
PY=/home/claude/venv/bin/python
export OMP_WAIT_POLICY=PASSIVE
$PY review5_synthetic.py > out/review5/synthetic_log.txt 2>&1  # detector recall on 11 synthetic copy types and spurious links
$PY review5_cpu.py       > out/review5/cpu_log.txt 2>&1        # matched single runs (RF, XGBoost), rotated hue (ICC under white balance,
                                                               # 5 repetitions), grouped SHAP
$PY review5_post.py                                            # near-miss label concordance, false-link bound, matched DeLong with Holm,
                                                               # Nadeau-Bengio corrected t, LOHO on the logit scale -> out/review5/post.json
$PY make_figS5_logit.py                                     # Supplementary Fig. S5 on the logit scale
$PY review5_kit.py                                             # blinded two-rater kit (needs the images; not redistributed)
# After two raters fill their sheets: $PY kappa_inspection.py respuestas_evaluador_A.csv respuestas_evaluador_B.csv CLAVE_no_mostrar_a_evaluadores.csv
echo ALL_DONE $(date)
