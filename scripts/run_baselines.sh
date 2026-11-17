#!/usr/bin/env bash
# Reference-solver baseline: Gurobi (instances inside the size limit) and HiGHS (all),
# 4 threads, 3 seeds, one 60 s run each; values at 5/10/30 s are read from the trace.
# Runs are sequential and resumable. Usage:
#   scripts/run_baselines.sh                 # everything (about 3 hours)
#   PYTHON=/path/to/python scripts/run_baselines.sh
set -euo pipefail
PY="${PYTHON:-python}"
RUN=("$PY" -I -m scsp.bench.run --time-limits 60 --seeds 1 2 3 --threads 4)

# the sets the P1 go/no-go criterion uses come first
"${RUN[@]}" --configs gurobi --sets dms scp4 scp5 scp6 scpe sp sppnw
"${RUN[@]}" --configs highs --sets dms scp4 scp5 scp6 scpe scpa scpb scpc scpd
"${RUN[@]}" --configs highs --sets scpnre scpnrf scpnrg scpnrh rail
"${RUN[@]}" --configs highs --sets sp sppnw sppaa sppus sppkl miplib
