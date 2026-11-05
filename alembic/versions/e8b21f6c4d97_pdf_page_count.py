"""pdf page count

Revision ID: e8b21f6c4d97
Revises: d7a4e51c9b30
Create Date: 2026-07-28 18:05:00.000000

Stores the page count of an uploaded PDF so a document bubble can render
"13 pages" and the viewer knows how many pages to request, without re-parsing
the file on every read.

Two columns because the value has to survive the staging hop: /api/upload writes
an `uploads` row, and the count is copied onto `message_attachments` when a
message claims that upload. Neither table had a spare or JSON column to overload
— message_attachments has width/height/duration_seconds, all typed for their own
purpose, and uploads has none of them.

Nullable with no default and no backfill, so this is a metadata-only ALTER with
no table rewrite. NULL means "not a PDF, or uploaded before previews existed" —
consumers must treat NULL as unknown and fall back to the plain icon bubble,
never coerce it to 0.
"""

import sqlalchemy as sa
from alembic import op

revision = "e8b21f6c4d97"
down_revision = "d7a4e51c9b30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("uploads", sa.Column("page_count", sa.Integer(), nullable=True))
    op.add_column("message_attachments", sa.Column("page_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("message_attachments", "page_count")
    op.drop_column("uploads", "page_count")
