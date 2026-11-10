"""timed mute + per-user notification preferences

Revision ID: b2d7e94c1f60
Revises: e3b9d5a7c142
Create Date: 2026-11-06 12:00:00.000000

`conversation_participants.muted_until` lets a mute lapse on its own ("8 hours",
"1 week"). NULL alongside is_muted is "muted always", which is what every
existing muted row already means, so there is nothing to backfill. A lapsed mute
is never cleared by a job; readers treat `muted_until <= now()` as not muted.

`users.notification_prefs` holds the push preferences the server must honour
because it is the server that sends push (show_preview, messages, groups,
calls). Stored sparse — only keys the user changed — and merged over defaults on
read, so '{}' means "everything on" and a preference added later needs no
backfill either.

Both ALTERs are metadata-only on Postgres 11+: a nullable column with no default,
and a NOT NULL column whose default is a constant. No table rewrite, so the API's
boot-time `alembic upgrade head` does not stall on either table.

No indexes: muted_until is read on participant rows already loaded for the
conversation, and notification_prefs on user rows fetched by primary key.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "b2d7e94c1f60"
down_revision = "e3b9d5a7c142"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add conversation_participants.muted_until and users.notification_prefs."""
    op.add_column(
        "conversation_participants",
        sa.Column("muted_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "notification_prefs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    """Drop both columns. Timed mutes become permanent mutes on the way down, which
    is the only meaning is_muted had before this revision."""
    op.drop_column("users", "notification_prefs")
    op.drop_column("conversation_participants", "muted_until")
