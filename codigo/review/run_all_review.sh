#!/bin/bash
# Run the three review analysis scripts in parallel (each resumes from its saved results) and wait for all of them.
cd /home/claude/exp
nice -n 5 /home/claude/venv/bin/python review_analyses.py >> out/review/log.txt 2>&1 &
P1=$!
nice -n 5 /home/claude/venv/bin/python review_analyses_b.py >> out/review/log_b.txt 2>&1 &
P2=$!
nice -n 10 /home/claude/venv/bin/python review_strict.py >> out/review_strict_log.txt 2>&1 &
P3=$!
wait $P1; echo "main exit $?"; wait $P2; echo "b exit $?"; wait $P3; echo "strict exit $?"
echo ALL_DONE $(date)
