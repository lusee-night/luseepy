"""Test setup for lusee.ingest.

``lusee.ingest.collation`` imports the uncrater package at module top
(it is the canonical source of AppID constants and Class A/B/C
classification predicates -- see the AppID-source-of-truth addendum in
the plan file). For local-dev convenience this conftest prepends
``UNCRATER_PATH`` to ``sys.path`` so a sibling checkout works without an
explicit ``pip install -e``. CI is expected to install uncrater into
the environment normally.
"""

from __future__ import annotations

import os
import sys


_uncrater_path = os.environ.get("UNCRATER_PATH")
if _uncrater_path and _uncrater_path not in sys.path:
    sys.path.insert(0, _uncrater_path)

# DCB / encoder telemetry decoding lives in a private package
# (``lusee_telemetry``). When ``LUSEE_TELEMETRY_PATH`` is set, prepend
# that directory so the public proxy in ``lusee.ingest.telemetry`` can
# import it. Tests that exercise telemetry skip when the decoder isn't
# loadable; this is just a convenience for local development.
_telemetry_path = os.environ.get("LUSEE_TELEMETRY_PATH")
if _telemetry_path and _telemetry_path not in sys.path:
    sys.path.insert(0, _telemetry_path)
