"""rag.raw_artifact: registry of original files stored on disk

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "raw_artifact",
        sa.Column("raw_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("meta", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ingested_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("source", "name", "sha256", name="uq_raw_artifact_source_name_sha"),
        schema="rag",
    )
    op.create_index(
        "ix_raw_artifact_pending",
        "raw_artifact",
        ["source", "fetched_at"],
        schema="rag",
        postgresql_where=sa.text("ingested_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_table("raw_artifact", schema="rag")
