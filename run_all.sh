#!/usr/bin/env bash
set -euo pipefail

cd /home/sr2/kjh9503/qa_task

KG_MODE="${KG_MODE:-merge}"

for qnum in $(seq 0 99); do
  echo "Running QNUM=${qnum}"
  KG_MODE="${KG_MODE}" QNUM="${qnum}" crewai run
done
