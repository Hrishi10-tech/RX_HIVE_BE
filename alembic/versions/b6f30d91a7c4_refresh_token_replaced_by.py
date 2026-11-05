"""refresh token replaced_by link

Revision ID: b6f30d91a7c4
Revises: f4a2c8e1b09d
Create Date: 2026-07-29 15:05:00.000000

Adds the self-reference that makes refresh rotation delivery-safe: the row that
superseded a rotated token.

Rotation revokes the presented token and commits BEFORE the response carrying its
replacement can reach the client. When that response is lost — the mobile client's
30s budget expiring, a proxy 502 after the commit, the app being suspended, a
reset connection — the client is left holding a token the server has already spent
through no fault of its own, and its next refresh was a 401 that ended a session
which was in fact healthy. That is one of the two server-side causes of the
"re-enter your password" reports.

The link is what tells the two kinds of reuse apart. A replay whose successor is
recorded, unspent and unexpired, and whose revocation is inside
RXHIVE_REFRESH_REUSE_GRACE_SECONDS (60s), is an undelivered rotation: the API
rotates from the successor instead and returns 200. Anything else — outside the
window, or a successor someone has already spent — is the signature of a captured
cookie, and now revokes every live session for that user and that client rather
than only failing the token presented. So this is a net tightening, not a
loosening: the previous build let a thief keep using the rest of the chain.

Nullable with no default and no backfill, so this is a metadata-only ALTER with no
table rewrite. NULL means "never rotated, or rotated before this column existed",
and both must read as "no successor to fall back on" — a replay of a pre-migration
token therefore fails exactly as it does today, which is the safe direction.

ondelete SET NULL, not CASCADE. The successor is the LIVE session; a CASCADE from
the ancestor row would let pruning an expired token delete the session the user is
currently working in. Losing the link is harmless — it degrades that one row to
"no successor recorded".

No index on replaced_by_id. It is only ever dereferenced as a primary-key lookup
of a single row on the refresh path, and the family revocation filters on
(user_id, client), already served by ix_refresh_tokens_user. Postgres does not
index a foreign key automatically, and adding one here would be write cost on
every rotation for no read at all.
"""

import sqlalchemy as sa
from alembic import op

revision = "b6f30d91a7c4"
down_revision = "f4a2c8e1b09d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("refresh_tokens", sa.Column("replaced_by_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "refresh_tokens_replaced_by_id_fkey",
        "refresh_tokens",
        "refresh_tokens",
        ["replaced_by_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # Dropping the column would take the constraint with it; named explicitly so a
    # partially applied upgrade leaves nothing behind.
    op.drop_constraint("refresh_tokens_replaced_by_id_fkey", "refresh_tokens", type_="foreignkey")
    op.drop_column("refresh_tokens", "replaced_by_id")
