from __future__ import annotations

import os

import pytest

TEST_PG_DSN = os.environ.get("TEST_PG_DSN")

requires_pg = pytest.mark.skipif(
    not TEST_PG_DSN, reason="set TEST_PG_DSN to run integration tests against PostgreSQL+pgvector"
)
