"""i18n schema: locale, text, translation (as given in the task)

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "i18n_locale",
        sa.Column("locale_code", sa.String(16), primary_key=True),
        sa.Column("display_name", sa.String(64), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
    )
    op.create_table(
        "i18n_text",
        sa.Column("text_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("text_code", sa.String(256), unique=True),
        sa.Column("owner_kind", sa.String(20)),
    )
    op.create_table(
        "i18n_translation",
        sa.Column("translation_id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "text_id",
            sa.BigInteger,
            sa.ForeignKey("i18n_text.text_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale_code",
            sa.String(16),
            sa.ForeignKey("i18n_locale.locale_code", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("translated_text", sa.Text, nullable=False),
        sa.Column("date_updated", sa.DateTime, nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("text_id", "locale_code", name="uq_i18n_translation_text_locale"),
    )
    # Для інкрементального fetch за date_updated.
    op.create_index("ix_i18n_translation_date_updated", "i18n_translation", ["date_updated"])


def downgrade() -> None:
    op.drop_index("ix_i18n_translation_date_updated", table_name="i18n_translation")
    op.drop_table("i18n_translation")
    op.drop_table("i18n_text")
    op.drop_table("i18n_locale")
