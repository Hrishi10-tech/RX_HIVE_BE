"""mobile access grant

Revision ID: d7a4e51c9b30
Revises: c3f1a7b92d04
Create Date: 2026-07-28 22:05:00.000000

Adds the per-user grant that gates sign-in from the native mobile app, plus the
column that records which client opened a refresh session.

`users.mobile_access` defaults to FALSE, not TRUE. Shipping a second client to an
existing user base is exactly the moment you do not want an implicit grant: the
web app is reachable from managed browsers, a phone is not, so mobile is opt-in
per account and a superadmin decides account by account. The tradeoff is that
nobody can sign in on mobile until someone grants it — deliberate, and the login
error says so explicitly rather than reading as bad credentials.

`refresh_tokens.client` exists because the grant has to be *revocable*. is_active
is re-checked on refresh, which is what makes deactivation stick; mobile_access
needs the same treatment, but only for mobile sessions — revoking mobile access
must not sign the user out of the web app they are using. user_agent could not
serve this purpose: it is client-supplied text a mobile client could spoof to
look like a browser, and the whole point of the column is to be trusted. The
server sets it from the authenticated login call, so it cannot be forged after
the fact.

Both columns carry a server_default so the ALTERs are non-blocking metadata-only
changes on Postgres 11+ (no heap rewrite), and so rows written by an older API
process mid-deploy still satisfy NOT NULL. Existing sessions predate the column
and become 'web', which is correct: every session that exists today was opened by
the web app.

No index on mobile_access. It is a low-cardinality boolean read one row at a time
on the login path (already a primary-key/unique-email lookup), and the admin
list filters on org/dept/status, not on this — an index would be write cost for
no read benefit.
"""

import sqlalchemy as sa
from alembic import op

revision = "d7a4e51c9b30"
down_revision = "c3f1a7b92d04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "mobile_access",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "refresh_tokens",
        sa.Column(
            "client",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'web'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("refresh_tokens", "client")
    op.drop_column("users", "mobile_access")
