#!/bin/bash
# Third internal review, in dependency order. Run after codigo/review2/run_all_review2.sh and after the GPU batch
# gpu/run_review3_cnn.bat (tuned recipe on the geometric set; its JSONs are read from out/review2/gpu2/).
# Scripts are run from /home/claude/exp, as in the second round; edit the paths for your setup.
set -e
cd /home/claude/exp; mkdir -p out/review3
PY=/home/claude/venv/bin/python
export OMP_WAIT_POLICY=PASSIVE   # avoids XGBoost slowing down when several jobs share the cores
$PY review3_geomverify_lo.py > out/review3/geomverify_lo_log.txt 2>&1  # photometric verification of the pairs with 8-9 SIFT inliers
$PY review3_thresholds.py                                             # 3 x 3 grid of NCC and inlier thresholds; false-link bound from label concordance
$PY review3_cluster_cis.py   > out/review3/cluster_cis_log.txt 2>&1   # Table 2 / Figure 3: cluster-bootstrap CIs and Obuchowski tests (geometric groups)
$PY review3_cpu.py           > out/review3/cpu_log.txt 2>&1           # duplication patterns, paired 337-image comparison, size control, Riley 2020,
                                                                      # RF vs tuned CNN, eroded mask on perturbed images, LOHO on the geometric set
$PY review3_size2.py         > out/review3/size2_log.txt 2>&1         # size control with random removal within class (XGBoost)
$PY review3_ensemble.py                                              # AUC of the run-averaged probabilities (estimand of the DeLong contrast)
echo ALL_DONE $(date)
