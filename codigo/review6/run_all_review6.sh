#!/bin/bash
# Sixth internal review, run after codigo/review5/run_all_review5.sh.
# Scripts are run from /home/claude/exp, as in earlier rounds; edit the paths for your setup.
set -e
cd /home/claude/exp; mkdir -p out/review6
python review5_post.py      # corrected stage-4 funnel: candidates and the 1,000 random control pairs counted separately
python review6_geo.py       # geometric set as the principal set: metrics, calibration, decision curves, Figures S4 and S6
python review6_lum.py       # luminance-only ablation with the 25 luminance statistics (R/G excluded)
