"""stars pins and conversation permissions

Revision ID: bdbf829f823f
Revises: 09c892c227fc
Create Date: 2026-07-25 16:41:03.123899

Reviewed after --autogenerate:
- The generated drop of messages.search_tsv + ix_messages_search_tsv was REMOVED.
  That column is a raw-SQL generated tsvector created in 09c892c227fc and is
  deliberately absent from the models, so autogenerate always wants to drop it.
- The five conversations.perm_* booleans are added WITH a server_default so
  existing rows backfill, then the default is dropped: the models carry
  Python-side defaults only, matching every other boolean column in the schema.
"""

import sqlalchemy as sa
from alembic import op

revision = 'bdbf829f823f'
down_revision = '09c892c227fc'
branch_labels = None
depends_on = None

# (column, default) — send_messages is NOT here: it maps to the existing
# admin_only_messages column, inverted.
_PERMISSION_COLUMNS = (
    ('perm_edit_info', 'true'),
    ('perm_add_members', 'true'),
    ('perm_send_history', 'true'),
    ('perm_invite_via_link', 'true'),
    ('perm_approve_new_members', 'false'),
)


def upgrade() -> None:
    op.create_table('message_pins',
    sa.Column('message_id', sa.UUID(), nullable=False),
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.Column('pinned_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['pinned_by'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('message_id')
    )
    op.create_index('ix_message_pins_conversation', 'message_pins', ['conversation_id'], unique=False)
    op.create_table('message_stars',
    sa.Column('message_id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('message_id', 'user_id')
    )
    op.create_index('ix_message_stars_user', 'message_stars', ['user_id'], unique=False)
    for name, default in _PERMISSION_COLUMNS:
        op.add_column(
            'conversations',
            sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.text(default)),
        )
        op.alter_column('conversations', name, server_default=None)


def downgrade() -> None:
    for name, _default in reversed(_PERMISSION_COLUMNS):
        op.drop_column('conversations', name)
    op.drop_index('ix_message_stars_user', table_name='message_stars')
    op.drop_table('message_stars')
    op.drop_index('ix_message_pins_conversation', table_name='message_pins')
    op.drop_table('message_pins')
