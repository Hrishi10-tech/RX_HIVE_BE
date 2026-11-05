"""conversation deletion tombstone

Revision ID: f9c3b471e5a8
Revises: c5d81e37a204
Create Date: 2026-08-12 00:00:00.000000

Separates "this group was deleted" from "this group was archived".

Both operations on a cross-org group set is_active = False, and is_active was
the only state either of them wrote. POST /{id}/archive toggles that column, so
running it against a deleted group read the tombstone as an archive and flipped
the group back to active — members got a removed_from_conversation event on the
delete and a conversation_created event on the toggle, and the group returned to
every sidebar. A destructive admin action was undoable by an unrelated one.

deleted_at is one-way. Nothing in the app clears it; the admin loader in
api/cross_org.py refuses to return a row that has it set, so archive, update,
member changes and a second delete all 404 rather than acting on a deleted
group. is_active is still set alongside it, because every member-facing read
path (api/conversations.py, services/conversations.py, services/messaging.py)
gates on is_active and none of them need to learn about this column.

No backfill, and none is possible: rows deleted before this migration are
byte-for-byte identical to archived ones. They stay archived — recoverable
through the toggle, which is the pre-existing behaviour and strictly safer than
guessing which of them an admin meant to destroy.

Nullable with no server default. The column is written on exactly one code path,
and a plain ADD COLUMN of a nullable column is metadata-only in Postgres 11+ —
no table rewrite, no lock held for the length of a scan. No index: conversations
is small, and the only queries that filter on it (the superadmin group list) are
already scanning by type.
"""

import sqlalchemy as sa
from alembic import op

revision = "f9c3b471e5a8"
down_revision = "c5d81e37a204"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Downgrading collapses deleted groups back into archived ones — the exact
    # ambiguity this revision removes. They become resurrectable again; that is
    # inherent to dropping the column, not something the downgrade can preserve.
    op.drop_column("conversations", "deleted_at")
