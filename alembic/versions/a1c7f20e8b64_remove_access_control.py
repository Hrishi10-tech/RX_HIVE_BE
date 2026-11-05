"""Remove access control: everyone in an organisation may chat and send anything.

Drops the three tables f4a2c8e1b09d added — chat_access_rules, send_policies and
admin_departments — plus the access_party_type enum.

This is a DATA-DESTROYING migration, deliberately. Every authored rule, every
per-category send policy and every admin-to-department delegation is gone once
it runs, and downgrade() cannot bring them back: it recreates the empty schema
so the chain stays reversible, but the rows are not recoverable from anything
here. Take a snapshot first if the rules might ever be wanted again.

Reachability is now "same organisation", enforced at
conversations.get_or_create_direct, and file type and size are enforced only by
the storage limits in services/storage.py.

Note admin_departments goes too, which changes org-admin behaviour rather than
merely removing an unused table. managed_dept_ids read "no rows" as
ORGANISATION-WIDE, so dropping the table puts every org admin back to seeing and
managing their whole organisation. That is the intended outcome, not a
side-effect. The separate rule that an admin may create members but not other
admins does not depend on this table and is unaffected.

Revision ID: a1c7f20e8b64
Revises: b6f30d91a7c4
"""

import sqlalchemy as sa
from alembic import op

revision = "a1c7f20e8b64"
down_revision = "b6f30d91a7c4"
branch_labels = None
depends_on = None

# create_type=False for the same reason f4a2c8e1b09d needed it: create_table has
# no checkfirst, so letting it create the type implicitly collides with the
# explicit create in downgrade(). The type is managed in one place.
party_enum = sa.dialects.postgresql.ENUM(
    "user", "department", name="access_party_type", create_type=False
)


def upgrade() -> None:
    bind = op.get_bind()

    # Reported rather than silently discarded: an operator running this against
    # production should see in the log how much authored configuration it took
    # with it.
    for table in ("chat_access_rules", "send_policies", "admin_departments"):
        exists = bind.execute(sa.text("SELECT to_regclass(:t)"), {"t": table}).scalar()
        if exists is None:
            print(f"  {table}: absent, nothing to drop")
            continue
        count = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar()  # noqa: S608
        print(f"  {table}: dropping {count} row(s)")

    op.drop_table("admin_departments")
    op.drop_table("send_policies")
    # Dropped explicitly before the table: created explicitly in f4a2c8e1b09d, so
    # drop_table does not know to remove it.
    op.drop_index("ix_chat_access_rules_org", table_name="chat_access_rules")
    op.drop_table("chat_access_rules")
    party_enum.drop(bind, checkfirst=True)


def downgrade() -> None:
    """Recreate the schema, EMPTY. The rules themselves are not recoverable."""
    bind = op.get_bind()
    party_enum.create(bind, checkfirst=True)

    op.create_table(
        "chat_access_rules",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "org_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("a_type", party_enum, nullable=False),
        sa.Column("a_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("b_type", party_enum, nullable=False),
        sa.Column("b_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("allow", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "created_by",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.UniqueConstraint("a_type", "a_id", "b_type", "b_id", name="uq_chat_access_rule_pair"),
    )
    op.create_index("ix_chat_access_rules_org", "chat_access_rules", ["org_id"])

    op.create_table(
        "send_policies",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "org_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope_type", party_enum, nullable=False),
        sa.Column("scope_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("image", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("video", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("audio", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("document", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("doc_extensions", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_by",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.UniqueConstraint("scope_type", "scope_id", name="uq_send_policy_scope"),
    )

    op.create_table(
        "admin_departments",
        sa.Column(
            "user_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "dept_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("departments.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
