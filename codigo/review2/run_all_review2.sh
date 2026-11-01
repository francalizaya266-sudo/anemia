#!/bin/bash
# Second internal review, in dependency order. Scripts are run from /home/claude/exp, where the container pipeline keeps
# common.py, the feature tables (out/feat_all.csv, out/feat_dedup.csv) and the outputs (out/review2/); edit the paths for your setup.
set -e
cd /home/claude/exp
PY=/home/claude/venv/bin/python
$PY review2_stats.py        > out/review2/stats_log.txt 2>&1      # class-conditional duplication, clustered tests, PPV/NPV, REML-HK, BSS, sample size
$PY review2_images.py       > out/review2/images_log.txt 2>&1     # Hamming histogram, Grad-CAM ceiling, mask area, feature stability, singleton LOHO
$PY review2_orb.py          > out/review2/orb_log.txt 2>&1        # fourth audit stage: SIFT + RANSAC on all pairs (about 1 h on 4 cores)
$PY review2_geomverify.py   > out/review2/geomverify_log.txt 2>&1 # photometric verification of the candidates
$PY review2_build_geo_map.py                                      # accept rule -> codigo/meta_groups_geo.csv, out/review2/geo_map_summary.json
$PY review2_geo.py                                                # classical models on the geometric set + grouped protocol (log in out/review2/geo/)
$PY review2_geo_clustered.py                                      # cluster-aware tests with the geometric groups as clusters
echo ALL_DONE $(date)
