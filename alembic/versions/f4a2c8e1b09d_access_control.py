"""Access control: chat reachability rules, send policies, admin departments.

Revision ID: f4a2c8e1b09d
Revises: e8b21f6c4d97

Creates the three tables, then BACKFILLS reachability from observed traffic.

The backfill is the whole reason this migration is more than DDL. Reachability
is deny-by-default, so applying it to a live database with no rules would make
every existing conversation unreachable and the product would come up dead. The
inference writes DEPARTMENT-level allow rules for every pair of departments that
already share a conversation, which produces a small rule set an administrator
can actually read and prune — as opposed to one row per historical pair, which
on a real deployment is thousands of rows nobody will ever audit.

The deliberate trade, stated so nobody is surprised by it: department-level
inference is COARSER than the history it is derived from. If one person in
Radiology ever messaged one person in Pharmacy, the whole of Radiology can reach
the whole of Pharmacy afterwards. That is the granularity the rule set was asked
for; tightening it is a pruning job in the admin UI, not a migration.

Users with dept_id NULL cannot be covered by a department rule, so pairs
involving them get an explicit user-level rule instead. Without that they would
be silently cut off from conversations they are already in.

cross_org conversations are excluded: they have org_id NULL by construction and
rules are org-scoped, so there is nothing to key a rule on. Enforcement exempts
them explicitly rather than leaving them to fail closed.

Idempotent — every insert is ON CONFLICT DO NOTHING against the canonical-pair
unique constraint, so re-running adds nothing.
"""

import sqlalchemy as sa
from alembic import op

revision = "f4a2c8e1b09d"
down_revision = "e8b21f6c4d97"
branch_labels = None
depends_on = None


# create_type=False so the create_table calls below do not emit CREATE TYPE a
# second time. Alembic's create_table has no checkfirst, so the implicit creation
# collided with the explicit checkfirst create in upgrade() and every migration
# run died on DuplicateObjectError. The explicit .create()/.drop() calls still
# issue the DDL, so the type is managed in exactly one place.
party_enum = sa.dialects.postgresql.ENUM(
    "user", "department", name="access_party_type", create_type=False
)


def upgrade() -> None:
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
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "created_by",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
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
        sa.Column("allow_text", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("allow_image", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("allow_video", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("allow_audio", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("allow_document", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("doc_extensions", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_by",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
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
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- Backfill -----------------------------------------------------------
    #
    # The party literals are cast explicitly: SELECT DISTINCT has to resolve a
    # concrete type for every output column, which pins an unadorned literal to
    # text, and Postgres will not implicitly widen text into an enum column.
    #
    # LEAST/GREATEST reproduce access.canonical_pair()'s ordering. That function
    # sorts on (type value, str(uuid)); both sides of each pair below share a
    # type, so it reduces to ordering the uuids — and Postgres uuid ordering
    # agrees with Python's str(uuid) ordering (fixed-position dashes, lowercase
    # hex). If the two ever diverge, duplicate mirrored rules become possible
    # and the unique constraint stops protecting anything.

    dept_pairs = bind.execute(
        sa.text(
            """
            INSERT INTO chat_access_rules (id, org_id, a_type, a_id, b_type, b_id, allow)
            SELECT DISTINCT
                gen_random_uuid(),
                c.org_id,
                'department'::access_party_type, LEAST(ua.dept_id, ub.dept_id),
                'department'::access_party_type, GREATEST(ua.dept_id, ub.dept_id),
                true
            FROM conversation_participants pa
            JOIN conversation_participants pb
              ON pb.conversation_id = pa.conversation_id AND pb.user_id <> pa.user_id
            JOIN users ua ON ua.id = pa.user_id
            JOIN users ub ON ub.id = pb.user_id
            JOIN conversations c ON c.id = pa.conversation_id
            WHERE c.org_id IS NOT NULL
              AND c.type <> 'cross_org'
              AND ua.dept_id IS NOT NULL
              AND ub.dept_id IS NOT NULL
              AND ua.org_id = c.org_id
              AND ub.org_id = c.org_id
            ON CONFLICT ON CONSTRAINT uq_chat_access_rule_pair DO NOTHING
            """
        )
    ).rowcount

    # Anyone without a department cannot be described by the rules above, so
    # their existing relationships are preserved individually.
    user_pairs = bind.execute(
        sa.text(
            """
            INSERT INTO chat_access_rules (id, org_id, a_type, a_id, b_type, b_id, allow)
            SELECT DISTINCT
                gen_random_uuid(),
                c.org_id,
                'user'::access_party_type, LEAST(ua.id, ub.id),
                'user'::access_party_type, GREATEST(ua.id, ub.id),
                true
            FROM conversation_participants pa
            JOIN conversation_participants pb
              ON pb.conversation_id = pa.conversation_id AND pb.user_id <> pa.user_id
            JOIN users ua ON ua.id = pa.user_id
            JOIN users ub ON ub.id = pb.user_id
            JOIN conversations c ON c.id = pa.conversation_id
            WHERE c.org_id IS NOT NULL
              AND c.type <> 'cross_org'
              AND (ua.dept_id IS NULL OR ub.dept_id IS NULL)
              AND ua.org_id = c.org_id
              AND ub.org_id = c.org_id
            ON CONFLICT ON CONSTRAINT uq_chat_access_rule_pair DO NOTHING
            """
        )
    ).rowcount

    # Existing org admins keep organization-wide reach: AdminDepartment treats
    # "no rows" as org-wide precisely so this migration does not have to guess
    # which departments each of them should manage, and does not silently strip
    # anyone. Assigning departments is an explicit act in the admin UI.
    print(
        f"[access-control] backfilled {dept_pairs} department pair rule(s) "
        f"and {user_pairs} user pair rule(s) from existing conversations"
    )


def downgrade() -> None:
    op.drop_table("admin_departments")
    op.drop_table("send_policies")
    op.drop_index("ix_chat_access_rules_org", table_name="chat_access_rules")
    op.drop_table("chat_access_rules")
    party_enum.drop(op.get_bind(), checkfirst=True)
