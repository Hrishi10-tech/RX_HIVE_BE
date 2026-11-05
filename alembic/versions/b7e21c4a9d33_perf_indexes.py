"""performance indexes: users roster ordering and the attachment foreign key

Revision ID: b7e21c4a9d33
Revises: f9c3b471e5a8
Create Date: 2026-08-13 00:00:00.000000

Two indexes, both measured rather than guessed.

users (org_id, display_name) — GET /api/users/contacts now takes a LIMIT, and
without a matching index a capped query is still a top-N heapsort over the whole
tenant: 702 buffers touched to return 200 rows against a seeded 25,000-user org.
The column order matches the query exactly (equality on org_id, then the ORDER BY
on display_name), which is what lets Postgres walk the index and stop at the cap
instead of sorting the tenant.

message_attachments (message_id) — an unindexed foreign key on the hottest join
in the product. Every message page joins attachments back to messages, and every
attachment lookup was a sequential scan of the whole table.

BOTH ARE BUILT CONCURRENTLY, and for carinrowena that is not optional. It takes
UPDATEs on three hot paths — login and token refresh both stamp last_seen_at
(app/api/auth.py), and so does every WebSocket liveness tick
(app/realtime/hub.py). A plain CREATE INDEX takes a SHARE lock, which conflicts
with the ROW EXCLUSIVE those writes hold, so building it non-concurrently would
block every login, every refresh and every heartbeat in the fleet for the length
of a full table scan.

CONCURRENTLY cannot run inside a transaction, hence the autocommit block, which
costs this migration its atomicity — so each build is preceded by a DROP INDEX
CONCURRENTLY IF EXISTS. A failed concurrent build leaves an INVALID index behind,
and a plain CREATE ... IF NOT EXISTS would mistake that for a finished one and
report success while the index enforced and accelerated nothing. Same pattern,
and the same reasoning, as c5d81e37a204.

The DROPs are concurrent for the same reason the CREATEs are. A plain DROP INDEX
takes ACCESS EXCLUSIVE on the parent table, which is precisely the lock this
migration exists to avoid taking on users. In the ordinary case the drop is a
no-op with nothing to lock, but the case it is written for — clearing an INVALID
index left by an earlier failed build — is exactly the case where the table is
live and the lock would be felt.
"""

from alembic import op

revision = "b7e21c4a9d33"
down_revision = "f9c3b471e5a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_users_org_display_name")
        op.execute("CREATE INDEX CONCURRENTLY ix_users_org_display_name ON users (org_id, display_name)")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_message_attachments_message_id")
        op.execute(
            "CREATE INDEX CONCURRENTLY ix_message_attachments_message_id ON message_attachments (message_id)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_message_attachments_message_id")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_users_org_display_name")
