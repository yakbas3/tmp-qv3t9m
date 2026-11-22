#!/bin/sh
# Record what a pytest session opens, lists and imports before any test runs
# (startup, conftests, plugins, collection), on the pinned tree. Run in the
# entry image:  docker run --rm -v <this dir>:/rec IMG sh /rec/startup_reads.sh
# Writes /rec/out_<name>.json for the executor file alone and the whole suite.
cd /opt/sqlglot || exit 1
for spec in "executor tests/test_executor.py" "suite tests"; do
  set -- $spec
  PYTHONPATH=/rec/startup_reads:/opt/sqlglot STARTUP_OUT=/tmp/out_$1.json SKIP_INTEGRATION=0 \
    python -m pytest --collect-only -q -p no:cacheprovider $2 > /tmp/log_$1.txt 2>&1
  tail -1 /tmp/log_$1.txt
  cat /tmp/out_$1.json > /rec/out_$1.json
done
