"""SQLAlchemy 2.0 models — the §6 schema.

Every tenant-scoped table carries org_id so the tenant guard can scope every
query by construction. Embedded Mongo arrays became join tables.
"""

import datetime as dt
import enum
import uuid

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {
        uuid.UUID: UUID(as_uuid=True),
        dt.datetime: DateTime(timezone=True),
        dict: JSONB,
    }


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[dt.datetime]:
    return mapped_column(server_default=func.now())


class UserRole(str, enum.Enum):
    superadmin = "superadmin"
    org_admin = "org_admin"
    member = "member"


class ConversationType(str, enum.Enum):
    direct = "direct"
    group = "group"
    cross_org = "cross_org"


class ParticipantRole(str, enum.Enum):
    creator = "creator"
    admin = "admin"
    member = "member"


class MessageType(str, enum.Enum):
    text = "text"
    image = "image"
    video = "video"
    audio = "audio"
    file = "file"
    system = "system"


class CallType(str, enum.Enum):
    voice = "voice"
    video = "video"


class CallStatus(str, enum.Enum):
    ringing = "ringing"
    connected = "connected"
    answered = "answered"
    missed = "missed"
    declined = "declined"
    cancelled = "cancelled"
    busy = "busy"
    no_answer = "no_answer"


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    logo_url: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[dt.datetime] = _now()

    departments: Mapped[list["Department"]] = relationship(back_populates="organization")


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = (UniqueConstraint("org_id", "name"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None]
    created_at: Mapped[dt.datetime] = _now()

    organization: Mapped[Organization] = relationship(back_populates="departments")


class User(Base):
    """A person who signs in, with their organization, department, role, password hash and account flags.

    must_change_password and sessions_valid_after are what an administrator's password reset sets.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="SET NULL"))
    dept_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("departments.id", ondelete="SET NULL"))
    email: Mapped[str] = mapped_column(CITEXT(), unique=True)
    display_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", values_callable=lambda e: [m.value for m in e])
    )
    avatar_url: Mapped[str | None]
    about: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(default=True)
    # Per-user grant for the native mobile app, issued by a superadmin only.
    # Defaults to False: mobile is opt-in per account, so an existing user base
    # does not silently become reachable from a new client. is_active governs
    # "may this account sign in at all"; this governs "may it sign in on mobile".
    # Superadmins are excluded from mobile entirely regardless of this flag —
    # the admin portal is web-only (see api/auth.py:_assert_mobile_allowed).
    mobile_access: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    must_change_password: Mapped[bool] = mapped_column(default=False)
    # The session epoch (batch 73 review). An access token issued before this moment
    # is refused on every gated route and at the websocket handshake, so the sessions
    # an administrator's reset ends stay ended: before it existed, a pre-reset access
    # token was only refused while must_change_password was set, and came back to
    # life the moment the owner chose a new password. Stamped by both admin resets
    # and nothing else; NULL means no reset has ever ended this account's sessions.
    # Nullable with no default on purpose — every existing row is correctly "never
    # reset", and the ALTER stays a metadata-only change.
    sessions_valid_after: Mapped[dt.datetime | None]
    last_seen_at: Mapped[dt.datetime | None]
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[dt.datetime] = _now()

    organization: Mapped[Organization | None] = relationship()
    department: Mapped[Department | None] = relationship()

    __table_args__ = (Index("ix_users_org_id", "org_id"),)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # NULL org_id only for cross-org conversations (created by super-admins).
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    type: Mapped[ConversationType] = mapped_column(
        Enum(
            ConversationType,
            name="conversation_type",
            values_callable=lambda e: [m.value for m in e],
        )
    )
    name: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None]
    avatar_url: Mapped[str | None]
    purpose_tag: Mapped[str | None] = mapped_column(String(100))  # cross_org groups
    allowed_org_ids: Mapped[dict] = mapped_column(JSONB, default=list)  # [str] for cross_org
    # The wire "send_messages" permission is this column INVERTED — one source of
    # truth for who may post, shared with the send path in services/messaging.
    admin_only_messages: Mapped[bool] = mapped_column(default=False)
    perm_edit_info: Mapped[bool] = mapped_column(default=True)
    perm_add_members: Mapped[bool] = mapped_column(default=True)
    # RESERVED — stored but NOT ENFORCED anywhere, and deliberately not exposed on
    # the wire (see serialize_permissions in services/enrich.py). No read path honours
    # perm_send_history, so a member added later still sees the whole pre-join
    # history through messages, search, starred, pinned, media/links and export.
    # Do not re-expose these until the read paths actually check them.
    perm_send_history: Mapped[bool] = mapped_column(default=True)
    perm_invite_via_link: Mapped[bool] = mapped_column(default=True)
    perm_approve_new_members: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    # Deletion tombstone, deliberately NOT the same column as is_active. Archiving
    # and deleting a cross-org group both drop it out of every member's list, so
    # while is_active was the only signal the two were the same state — and the
    # archive toggle, which just flips is_active, read a deleted group as merely
    # archived and put it back. deleted_at is one-way: nothing clears it, and the
    # admin paths in api/cross_org.py refuse to load a row that has it set.
    deleted_at: Mapped[dt.datetime | None]
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    last_message_at: Mapped[dt.datetime | None]
    created_at: Mapped[dt.datetime] = _now()

    participants: Mapped[list["ConversationParticipant"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_conversations_org_id", "org_id"),)


class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[ParticipantRole] = mapped_column(
        Enum(
            ParticipantRole,
            name="participant_role",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=ParticipantRole.member,
    )
    last_read_at: Mapped[dt.datetime | None]
    is_pinned: Mapped[bool] = mapped_column(default=False)
    # Explicit position within MY pinned block, lowest first. NULL means "pinned
    # but unordered" — either not pinned at all, or pinned before this column
    # existed. The list query sorts NULLS LAST so those fall back to recency.
    # A boolean alone could not express order, so pinned chats reshuffled
    # whenever any of them received a message.
    pin_order: Mapped[int | None] = mapped_column(default=None)
    is_muted: Mapped[bool] = mapped_column(default=False)
    joined_at: Mapped[dt.datetime] = _now()

    conversation: Mapped[Conversation] = relationship(back_populates="participants")
    user: Mapped[User] = relationship()

    __table_args__ = (
        Index("ix_participants_user_id", "user_id"),
        # Matches the sidebar's ORDER BY, and partial because only a handful of a
        # user's participant rows are ever pinned.
        Index(
            "ix_participants_pin_order",
            "user_id",
            "pin_order",
            postgresql_where=text("is_pinned"),
        ),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    sender_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    type: Mapped[MessageType] = mapped_column(
        Enum(MessageType, name="message_type", values_callable=lambda e: [m.value for m in e]),
        default=MessageType.text,
    )
    content: Mapped[str] = mapped_column(Text, default="")
    # The sender's own id for this message, carried as `temp_id` on both send
    # transports and stored so a retry cannot create a second row.
    #
    # A send is persisted and committed BEFORE the sender is told anything, so a
    # lost response, a timeout or a 502 leaves the client unable to tell "never
    # arrived" from "arrived, answer lost". It marked the bubble failed and offered
    # a retry, and the retry minted a fresh id, so the uncertain case duplicated the
    # message. Nullable because system messages and every row written before this
    # column existed have no client id; the unique index below is partial for the
    # same reason.
    client_msg_id: Mapped[str | None] = mapped_column(String(64))
    reply_to_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"))
    is_forwarded: Mapped[bool] = mapped_column(default=False)
    edited_at: Mapped[dt.datetime | None]
    deleted_at: Mapped[dt.datetime | None]  # delete-for-everyone tombstone
    created_at: Mapped[dt.datetime] = _now()

    sender: Mapped[User | None] = relationship()
    reply_to: Mapped["Message | None"] = relationship(remote_side="Message.id")
    attachments: Mapped[list["MessageAttachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="MessageAttachment.created_at"
    )
    reactions: Mapped[list["MessageReaction"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        # One row per client id per sender per conversation — the actual guarantee
        # behind retry safety. The service checks for an existing row first, but
        # that is check-then-act and two concurrent copies of one send can both
        # pass it; this is what makes the loser fail instead of insert.
        #
        # Partial because system messages and every row predating the column carry
        # no client id. Declared here as well as in the migration, unlike the Links
        # scan index, because uniqueness is a property of the schema rather than a
        # shape chosen for one query.
        Index(
            "uq_messages_client_msg_id",
            "conversation_id",
            "sender_id",
            "client_msg_id",
            unique=True,
            postgresql_where=text("client_msg_id IS NOT NULL"),
            sqlite_where=text("client_msg_id IS NOT NULL"),
        ),
    )


class MessageAttachment(Base):
    __tablename__ = "message_attachments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    storage_key: Mapped[str] = mapped_column(String(500))
    thumbnail_key: Mapped[str | None] = mapped_column(String(500))
    filename: Mapped[str] = mapped_column(String(300))
    mime_type: Mapped[str] = mapped_column(String(150))
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    width: Mapped[int | None]
    height: Mapped[int | None]
    duration_seconds: Mapped[float | None]
    # PDF page count, so a document bubble can say "13 pages" and the viewer
    # knows how many pages to request.
    #   NULL = not a PDF, or a PDF never processed for previews yet. The lazy
    #          path in /media/{id}/thumb treats NULL as "try rendering it".
    #   0    = processed and could NOT be rendered (encrypted, corrupt). This is
    #          what stops an un-renderable file being re-parsed on every view.
    #   >0   = real page count.
    page_count: Mapped[int | None]
    created_at: Mapped[dt.datetime] = _now()

    message: Mapped[Message] = relationship(back_populates="attachments")


class MessageReaction(Base):
    __tablename__ = "message_reactions"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    emoji: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = _now()

    message: Mapped[Message] = relationship(back_populates="reactions")
    user: Mapped[User] = relationship()


class MessageStar(Base):
    """Starred messages — per-user and private (never visible to other members)."""

    __tablename__ = "message_stars"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[dt.datetime] = _now()

    __table_args__ = (Index("ix_message_stars_user", "user_id"),)


class MessagePin(Base):
    """Pinned messages — conversation-wide, so one row per message (PK message_id).

    conversation_id is denormalized so the pinned list for a conversation is one
    indexed lookup instead of a join back through messages.
    """

    __tablename__ = "message_pins"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    pinned_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[dt.datetime] = _now()

    __table_args__ = (Index("ix_message_pins_conversation", "conversation_id"),)


class MessageDeletion(Base):
    """Delete-for-me: rows here are filtered out of that user's reads."""

    __tablename__ = "message_deletions"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[dt.datetime] = _now()


class Call(Base):
    __tablename__ = "call_history"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    initiated_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    type: Mapped[CallType] = mapped_column(
        Enum(CallType, name="call_type", values_callable=lambda e: [m.value for m in e])
    )
    status: Mapped[CallStatus] = mapped_column(
        Enum(CallStatus, name="call_status", values_callable=lambda e: [m.value for m in e]),
        default=CallStatus.ringing,
    )
    is_group: Mapped[bool] = mapped_column(default=False)
    room_name: Mapped[str] = mapped_column(String(200), unique=True)
    started_at: Mapped[dt.datetime] = _now()
    answered_at: Mapped[dt.datetime | None]
    ended_at: Mapped[dt.datetime | None]

    participants: Mapped[list["CallParticipant"]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_calls_org_started", "org_id", "started_at"),)


class CallParticipant(Base):
    __tablename__ = "call_participants"

    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("call_history.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    invited_at: Mapped[dt.datetime] = _now()
    joined_at: Mapped[dt.datetime | None]
    left_at: Mapped[dt.datetime | None]
    seen_at: Mapped[dt.datetime | None]  # missed-call badge dismissal

    call: Mapped[Call] = relationship(back_populates="participants")
    user: Mapped[User] = relationship()

    __table_args__ = (Index("ix_call_participants_user", "user_id"),)


class ScheduledCall(Base):
    __tablename__ = "scheduled_calls"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    creator_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None]
    call_type: Mapped[CallType] = mapped_column(
        Enum(CallType, name="call_type", values_callable=lambda e: [m.value for m in e], create_type=False)
    )
    call_link_code: Mapped[str | None] = mapped_column(String(64))
    start_time: Mapped[dt.datetime]
    end_time: Mapped[dt.datetime | None]
    reminder_minutes: Mapped[int] = mapped_column(default=15)
    participant_ids: Mapped[dict] = mapped_column(JSONB, default=list)  # list[str] of user ids
    status: Mapped[str] = mapped_column(String(30), default="scheduled")
    created_at: Mapped[dt.datetime] = _now()


class CallLink(Base):
    __tablename__ = "call_links"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    creator_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    code: Mapped[str] = mapped_column(String(64), unique=True)
    call_type: Mapped[CallType] = mapped_column(
        Enum(CallType, name="call_type", values_callable=lambda e: [m.value for m in e], create_type=False)
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[dt.datetime] = _now()


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="SET NULL"))
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    actor_type: Mapped[str] = mapped_column(String(30), default="superadmin")
    action: Mapped[str] = mapped_column(String(100))
    target: Mapped[str | None] = mapped_column(String(300))
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[dt.datetime] = _now()

    __table_args__ = (Index("ix_audit_org_created", "org_id", "created_at"),)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_read: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = _now()

    __table_args__ = (Index("ix_notifications_user_created", "user_id", "created_at"),)


class RefreshToken(Base):
    """Persisted, revocable refresh sessions (NEW vs the Mongo build)."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    user_agent: Mapped[str | None] = mapped_column(String(300))
    # Which client opened this session: "web" | "mobile". Recorded because the
    # mobile-access grant is re-checked on refresh, and revoking the grant must
    # kill that user's MOBILE sessions without touching their web session.
    # user_agent cannot serve this purpose — it is attacker-controlled text.
    client: Mapped[str] = mapped_column(String(20), default="web", server_default=text("'web'"))
    expires_at: Mapped[dt.datetime]
    revoked_at: Mapped[dt.datetime | None]
    # The row that superseded this one on rotation. Without it, a client replaying
    # the token it holds because the rotation response never arrived is
    # indistinguishable from a thief replaying a stolen one — so both were signed
    # out. SET NULL, not CASCADE: pruning an old session must never delete the
    # live one that replaced it.
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL")
    )
    created_at: Mapped[dt.datetime] = _now()

    user: Mapped[User] = relationship()

    __table_args__ = (Index("ix_refresh_tokens_user", "user_id"),)


class Upload(Base):
    """Staging record for uploaded files before they're claimed by a message.

    The Mongo build wrote files to disk with no DB record (orphans forever).
    Here every upload is tracked; a cleanup job can purge unclaimed uploads.
    """

    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = _uuid_pk()
    uploader_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    storage_key: Mapped[str] = mapped_column(String(500))
    thumbnail_key: Mapped[str | None] = mapped_column(String(500))
    filename: Mapped[str] = mapped_column(String(300))
    mime_type: Mapped[str] = mapped_column(String(150))
    file_type: Mapped[str] = mapped_column(String(20))  # image|video|audio|document
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    # Carried from upload to MessageAttachment when the upload is claimed, so a
    # PDF's page count survives the staging hop. NULL for everything else.
    page_count: Mapped[int | None]
    claimed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = _now()


class PushSubscription(Base):
    """Web Push subscriptions (v1 notifications pipeline)."""

    __tablename__ = "push_subscriptions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    endpoint: Mapped[str] = mapped_column(Text, unique=True)
    keys: Mapped[dict] = mapped_column(JSONB, default=dict)  # {p256dh, auth}
    created_at: Mapped[dt.datetime] = _now()


# Schema the MIGRATIONS own outright, deliberately absent from the metadata above.
#
# `alembic revision --autogenerate` diffs the live database against
# Base.metadata, so every object listed here reads to it as something to DROP.
# That is not hypothetical: bdbf829f823f records deleting exactly such a drop out
# of a generated migration by hand, and back then the set was two entries. Three
# batches have added to it since, and the only safeguard was still whoever
# remembered to re-read the generated file. include_in_autogenerate() below is
# what alembic/env.py hands to alembic's include_object hook so the drops are
# never proposed in the first place; tests/test_migrations.py asserts this set
# stays exactly as wide as reality.
#
# Full-text search over messages is a GENERATED tsvector column plus its GIN
# index, created as raw DDL in 09c892c227fc_initial_schema.py:
#   ALTER TABLE messages ADD COLUMN search_tsv tsvector
#     GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;
#   CREATE INDEX ix_messages_search_tsv ON messages USING gin (search_tsv);
# Losing that column is not a performance regression — it takes the message
# results out of /api/search entirely.
MIGRATION_OWNED_COLUMNS = frozenset({("messages", "search_tsv")})

# Indexes that are a QUERY SHAPE rather than a property of the schema, which is
# why they are declared in their migration and nowhere else:
#   ix_messages_search_tsv          the GIN index over the column above
#   ix_messages_links               the Links tab's own ILIKE predicate (4471a6d661d3)
#   ix_users_org_display_name       the capped roster's ORDER BY (b7e21c4a9d33)
#   ix_message_attachments_message_id  the attachment foreign key (b7e21c4a9d33)
#   ix_message_attachments_created_keyset   \
#   ix_message_attachments_pdf_unpreviewed   > the preview tools' keyset walks
#   ix_uploads_pdf_unpreviewed              /  (a4f81c6b2e07)
#
# The last three exist because app/tools walks those tables keyset-paged, and two
# of them carry the tools' own scan predicates. Declaring them on the models would
# describe a maintenance script's query shape as a property of the schema, which
# it is not — and the partial predicates could not be expressed there anyway
# without repeating them a third time.
MIGRATION_OWNED_INDEXES = frozenset(
    {
        "ix_messages_search_tsv",
        "ix_messages_links",
        "ix_users_org_display_name",
        "ix_message_attachments_message_id",
        "ix_message_attachments_created_keyset",
        "ix_message_attachments_pdf_unpreviewed",
        "ix_uploads_pdf_unpreviewed",
    }
)


def include_in_autogenerate(object_, name, type_, reflected, compare_to) -> bool:
    """alembic's `include_object` hook: keep autogenerate off the two sets above.

    Scoped as narrowly as possible on purpose. An over-wide filter here would be
    worse than no filter at all: it would hide REAL drift — a model column that
    never got a migration — behind a clean autogenerate run.
    """
    if type_ == "index":
        return name not in MIGRATION_OWNED_INDEXES
    if type_ == "column":
        return (object_.table.name, name) not in MIGRATION_OWNED_COLUMNS
    return True
