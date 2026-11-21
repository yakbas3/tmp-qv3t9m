"""pytest plugin that records which tests raise a warning on the pinned tree.

Used only to build app/warn.json.gz, never at scoring time. Run the whole
suite with every warning shown, so filters that hide some do not hide them:

  WARN_OUT=/tmp/warn.json python -m pytest tests -p warn_plugin -W always \
      -p no:cacheprovider -o addopts=

Writes {"tests": [ids that warned while running], "other": count of
warnings raised outside a test (collection, config)}. Gzip it into app/.
A pytest config that turns warnings into errors can only fail those tests.
"""

import json
import os

TESTS = set()
OTHER = [0]


def pytest_warning_recorded(warning_message, when, nodeid, location):
    if when == "runtest" and nodeid:
        TESTS.add(nodeid)
    else:
        OTHER[0] += 1


def pytest_sessionfinish(session):
    with open(os.environ["WARN_OUT"], "w") as f:
        json.dump({"tests": sorted(TESTS), "other": OTHER[0]}, f)
