"""conversation pin order

Revision ID: c3f1a7b92d04
Revises: 4471a6d661d3
Create Date: 2026-07-28 13:40:00.000000

Adds a per-user ordinal for pinned conversations.

`conversation_participants.is_pinned` already existed and is already per-user
(the table's PK is (conversation_id, user_id)), so pinning itself needed no
schema change. What a boolean cannot express is ORDER: within the pinned block
the list query fell back to `Conversation.last_message_at DESC`, so a user's
pinned chats reshuffled the moment any one of them received a message. A
user-chosen order was not representable at all, let alone persistable across
sessions.

`pin_order` is nullable on purpose:
  * NULL  = not pinned, or pinned before this column existed. Ordered last
            within the pinned block via NULLS LAST, so pre-existing pins keep
            working and simply fall back to recency until they are re-pinned.
  * 0..n  = the user's explicit position, lowest first.

No backfill. Existing pinned rows keep pin_order NULL; the ORDER BY handles them
with NULLS LAST rather than this migration having to invent an order for data it
cannot know the intent of. That also keeps the migration instant on a large
table — it is a nullable column add with no default, so Postgres does not rewrite
the heap.

The partial index covers exactly the rows the sidebar query sorts on. Pinned rows
are a tiny fraction of participant rows (a user pins a handful of chats out of
hundreds), so a partial index is both much smaller than a full one and the only
part Postgres needs.
"""

import sqlalchemy as sa
from alembic import op

revision = "c3f1a7b92d04"
down_revision = "4471a6d661d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_participants",
        sa.Column("pin_order", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_participants_pin_order",
        "conversation_participants",
        ["user_id", "pin_order"],
        unique=False,
        postgresql_where=sa.text("is_pinned"),
    )


def downgrade() -> None:
    op.drop_index("ix_participants_pin_order", table_name="conversation_participants")
    op.drop_column("conversation_participants", "pin_order")
