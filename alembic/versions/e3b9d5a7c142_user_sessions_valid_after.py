"""users.sessions_valid_after: the session epoch an admin password reset stamps

Revision ID: e3b9d5a7c142
Revises: a4f81c6b2e07
Create Date: 2026-10-01 12:00:00.000000

Batch 73 review. Both admin resets promise that "old sessions die", and they
revoke every refresh token, but an access token is a signed JWT the server cannot
revoke. Batch 73 refused the target's pre-reset access token only while
must_change_password was set, so the moment the owner chose a new password that
token worked again for the rest of its lifetime — long enough to register a push
subscription, which never expires. The reset now stamps this column, and
app/core/deps.py refuses any access token issued before it on every gated route
and at the websocket handshake.

Nullable with no server default, deliberately. NULL is the true value for every
existing row — no reset has ended their sessions under this rule — so there is
nothing to backfill, and adding a nullable column without a default is a
metadata-only ALTER on Postgres: no table rewrite and only a momentary lock, so
the API's boot-time `alembic upgrade head` does not stall on the users table.

No index: it is read only on the row already fetched by primary key to
authenticate a request, never filtered on.
"""

import sqlalchemy as sa

from alembic import op

revision = "e3b9d5a7c142"
down_revision = "a4f81c6b2e07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the nullable users.sessions_valid_after column, with no default and no index."""
    op.add_column(
        "users",
        sa.Column("sessions_valid_after", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Drop users.sessions_valid_after and every session epoch stored in it."""
    op.drop_column("users", "sessions_valid_after")
