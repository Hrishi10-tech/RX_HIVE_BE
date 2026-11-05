"""message client idempotency key

Revision ID: c5d81e37a204
Revises: a1c7f20e8b64
Create Date: 2026-08-06 00:00:00.000000

Gives every client-originated message a durable idempotency key so a retry after
an UNCERTAIN send cannot create a second row.

send_message commits before the sender learns anything, so a lost response, a
request timeout or a 502 from in front of the app all leave the client holding a
message it cannot classify. The UI called that failed, offered a retry, and the
retry minted a fresh temp id — so precisely the case where the message HAD been
stored is the case that stored it twice. `temp_id` was already on the wire for
both transports and was only echoed back; this persists it.

The index is PARTIAL, and it has to be: system messages carry no client id, and
neither does any row written before this migration. A plain unique constraint
over a nullable column would be satisfied by those rows under Postgres NULL
semantics, but a partial index states the intent — uniqueness applies only where
a client id exists — instead of relying on that.

Scoped to (conversation_id, sender_id, client_msg_id) rather than the id alone:
client ids are uuid4 in practice, but they are client-supplied, and a key that
is only unique per sender per conversation cannot be used by one account to
block a send in another.

No backfill. Existing rows keep a NULL key and fall outside the index, which is
correct — they predate the guarantee and there is nothing to reconcile them to.

The index is built CONCURRENTLY, which costs this migration its atomicity; see
upgrade() for why neither half of that trade is optional.
"""

from alembic import op

revision = "c5d81e37a204"
down_revision = "a1c7f20e8b64"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ADD COLUMN IF NOT EXISTS rather than op.add_column: the autocommit block
    # below commits this statement, and alembic_version is only stamped once
    # upgrade() returns. A build that is cancelled or hits a lock timeout
    # therefore leaves the column committed and this revision unstamped, so the
    # retry re-enters upgrade() — where a plain ADD COLUMN would fail on
    # DuplicateColumn and hand back a migration that can never be completed.
    # Safe in a way the index below is not: a column either exists as declared or
    # does not, with no half-built state for IF NOT EXISTS to mistake for done.
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS client_msg_id VARCHAR(64)")

    # CONCURRENTLY, which cannot run inside a transaction, hence the autocommit
    # block. A plain CREATE UNIQUE INDEX takes a ShareLock and blocks writes; worse
    # here, the ADD COLUMN above holds ACCESS EXCLUSIVE until commit, so one
    # transaction spanning both blocks every read *and* write to messages until the
    # index is finished. That the index covers no rows yet does not make it quick —
    # the predicate still has to be evaluated against every row in the table, and
    # messages is the highest-volume table in this schema. Non-concurrently, this
    # migration is a full stall of sends for the length of a full heap scan.
    #
    # DROP first instead of CREATE ... IF NOT EXISTS: a failed concurrent build
    # leaves an INVALID index of this name behind, and IF NOT EXISTS would take it
    # for the finished article. The retry would report success while the index
    # enforces nothing — the exact duplicate row this revision exists to prevent,
    # now with a unique index apparently standing guard over it. Dropping first
    # costs nothing when the index isn't there and re-does the work when it is.
    #
    # The drop is CONCURRENTLY for the same reason the create is. A plain DROP
    # INDEX takes ACCESS EXCLUSIVE on messages — every read and every write
    # queued behind it — which is the one lock this whole block is written to
    # avoid taking on the highest-volume table in the schema. It is easy to read
    # the drop as the harmless half because it is usually a no-op, but the case
    # it exists for is the retry after a failed build, and that is exactly the
    # case where the index IS there, is INVALID, and the table is live. Since
    # production boots with `alembic upgrade head && ... && uvicorn`
    # (infra/docker-compose.prod.yml), blocking here does not merely stall
    # sends — the API never finishes starting.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_messages_client_msg_id")
        op.execute(
            """CREATE UNIQUE INDEX CONCURRENTLY uq_messages_client_msg_id
               ON messages (conversation_id, sender_id, client_msg_id)
               WHERE client_msg_id IS NOT NULL"""
        )


def downgrade() -> None:
    # Not CONCURRENTLY, unlike upgrade(): drop_column below takes ACCESS EXCLUSIVE
    # on messages regardless, so a concurrent drop here would buy nothing and
    # would cost this function the ability to run in one transaction.
    op.execute("DROP INDEX IF EXISTS uq_messages_client_msg_id")
    op.drop_column("messages", "client_msg_id")
