"""Uploads + authenticated media serving.

Replaces the Mongo build's unauthenticated /api/uploads/{filename} disk serving:
every file lives in object storage under a DB-tracked Upload/MessageAttachment
row, and every read is auth-checked (uploader or conversation membership) before
redirecting to a short-lived presigned URL.
"""

import os
import re
from urllib.parse import urlparse

import anyio
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import and_, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import TenantContext, get_current_user, get_tenant
from app.core.rate_limit import upload_limiter
from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageAttachment,
    MessageDeletion,
    MessageType,
    Upload,
    User,
)
from app.db.session import get_db
from app.services import storage
from app.services.enrich import attachment_thumb_url, media_url_for
from app.utils import iso_z, parse_uuid, sanitize_text

router = APIRouter(prefix="/api", tags=["media"])

_TOO_LARGE_MESSAGE = "File too large (max 2GB)"

_MEDIA_LIST_TYPES = {"image", "video", "audio", "file"}

_LINK_TYPE = "link"
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?)]}>'\""
# Links come out of message text, so the scan is bounded to the newest N
# link-bearing messages rather than the whole (unbounded) history.
_LINK_SCAN_LIMIT = 500
# Inlined as a SQL literal, not a bind parameter, so it matches the predicate of
# the ix_messages_links partial index verbatim (Postgres cannot prove a
# parameterised ILIKE implies the index predicate when it caches a generic plan,
# and silently falls back to scanning the conversation's whole history).
# Safe to inline: a fixed constant, never user input.
_LINK_PATTERN = literal_column("'%http%'")

_INLINE_MIME_PREFIXES = ("image/", "video/", "audio/")


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="File not found")


def _safe_filename(name: str | None) -> str:
    """Strip HTML and any path components from a client-supplied filename."""
    cleaned = sanitize_text(name or "").replace("\\", "/").split("/")[-1].strip()
    return (cleaned or "file")[:300]


def _is_inline(mime_type: str) -> bool:
    return mime_type.startswith(_INLINE_MIME_PREFIXES)


def _extract_links(content: str) -> list[tuple[str, str]]:
    """(url, domain) pairs found in message text, de-duplicated, order preserved.

    The URLs are NEVER fetched: no titles, no favicons, no metadata requests.
    Fetching user-supplied URLs server-side would hand every user an SSRF probe
    into the deployment's private network.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in _URL_RE.findall(content or ""):
        url = raw.rstrip(_URL_TRAILING)
        if not url or url in seen:
            continue
        domain = (urlparse(url).hostname or "").lower()
        if not domain:
            continue
        seen.add(url)
        out.append((url, domain))
    return out


@router.post("/upload")
async def upload_file(
    file: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(upload_limiter),
):
    if file is None or not (file.filename or "").strip():
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename or "")[1].lower()
    # Every extension is accepted. classify only decides which bubble renders it.
    file_type = storage.classify(ext)

    # Size from the spooled file rather than by counting bytes as we read them.
    # Starlette has already streamed the whole request body to a
    # SpooledTemporaryFile — on disk past 1 MB — so it is complete by the time
    # this handler runs and seek/tell is exact. The old loop accumulated it into
    # a bytearray as well, which meant a second full copy in memory; at the
    # 2 GB ceiling that is the difference between working and killing the
    # process for every other user on the box.
    size = file.size
    if size is None:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
    if size > storage.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=_TOO_LARGE_MESSAGE)
    if size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Content type from the extension map — the client's Content-Type is never
    # trusted, and unknown extensions stay application/octet-stream on purpose.
    #
    # Load-bearing now that any format is accepted. Media is served from a
    # SAME-ORIGIN path (see storage.rewrite_to_public) with
    # Content-Disposition: inline, so a stored type the browser renders is a
    # stored-XSS primitive on the app's own origin — an uploaded .html or .svg
    # would execute as us. octet-stream makes the browser download it instead.
    # Do not "improve" this into mimetypes.guess_type.
    content_type = storage.MIME_BY_EXT.get(ext, "application/octet-stream")

    key = storage.new_storage_key(user.org_id, ext)
    await storage.put_stream(key, file.file, size, content_type)

    # Previews need the bytes in memory, so they are the one thing still bounded
    # by size. Past the bound the upload succeeds without a preview rather than
    # the request failing — see storage.THUMBNAIL_SOURCE_LIMIT.
    thumbnail_key = None
    page_count = None
    thumb = None
    if size <= storage.THUMBNAIL_SOURCE_LIMIT and (file_type == "image" or content_type == "application/pdf"):
        file.file.seek(0)
        payload = await anyio.to_thread.run_sync(file.file.read)
        if file_type == "image":
            thumb = await storage.make_thumbnail(payload)
        else:
            thumb, page_count = await storage.make_pdf_preview(payload)
        del payload
    if thumb is not None:
        thumbnail_key = f"{key}_thumb.jpg"
        await storage.put_object(thumbnail_key, thumb, "image/jpeg")

    upload = Upload(
        uploader_id=user.id,
        org_id=user.org_id,
        storage_key=key,
        thumbnail_key=thumbnail_key,
        filename=_safe_filename(file.filename),
        mime_type=content_type,
        file_type=file_type,
        file_size=size,
        page_count=page_count,
    )
    db.add(upload)
    await db.commit()

    file_url = f"/api/media/up/{upload.id}"
    response = {
        "file_id": str(upload.id),
        "filename": upload.filename,
        "file_url": file_url,
        "file_type": upload.file_type,
        "file_size": upload.file_size,
        "mime_type": upload.mime_type,
    }
    if file_type == "image":
        # Contract: thumbnail_url falls back to the file itself if thumbnailing failed.
        response["thumbnail_url"] = f"{file_url}/thumb" if thumbnail_key else file_url
    elif content_type == "application/pdf":
        # No fallback to the file itself here: the PDF bytes are not an image, so
        # pointing thumbnail_url at them would render a broken tile. Absent means
        # "no preview", which the bubble handles by keeping the icon.
        if thumbnail_key:
            response["thumbnail_url"] = f"{file_url}/thumb"
        response["page_count"] = page_count
    return response


async def _own_upload(db: AsyncSession, upload_id: str, user: User) -> Upload:
    """Load an upload iff the caller is its uploader (404 otherwise)."""
    uid = parse_uuid(upload_id)
    if uid is None:
        raise _not_found()
    upload = await db.get(Upload, uid)
    if upload is None or upload.uploader_id != user.id:
        raise _not_found()
    return upload


@router.get("/media/up/{upload_id}")
async def serve_upload(
    upload_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    upload = await _own_upload(db, upload_id, user)
    url = await storage.presign_get(upload.storage_key, filename=upload.filename, inline=True)
    return RedirectResponse(url, status_code=307)


@router.get("/media/up/{upload_id}/thumb")
async def serve_upload_thumb(
    upload_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    upload = await _own_upload(db, upload_id, user)
    if not upload.thumbnail_key:
        raise _not_found()
    url = await storage.presign_get(upload.thumbnail_key, inline=True)
    return RedirectResponse(url, status_code=307)


async def _member_attachment(db: AsyncSession, attachment_id: str, user: User) -> MessageAttachment:
    """Load an attachment iff the caller participates in its conversation and
    the message is not tombstoned (404 otherwise — existence is not revealed)."""
    att_id = parse_uuid(attachment_id)
    if att_id is None:
        raise _not_found()
    # ONE round trip, not three. This ran as three sequential db.get calls —
    # attachment, then its message, then the caller's participant row — and every
    # per-asset route goes through it: the bytes, the thumbnail, /meta and every PDF
    # page. Each lookup was an index hit but they were serialised, so a chat pane
    # showing thirty images spent ninety round trips purely on authorisation, each
    # holding one of the ten pooled connections. Measured over 40 requests with a
    # fresh session apiece: 120 queries and 1.23ms each, against 40 and 0.67ms here,
    # and the gap widens with network distance because it is round trips rather than
    # work.
    #
    # The joins reproduce every 404 condition at once — no attachment, no message or a
    # tombstoned one, no membership, a revoked conversation, the wrong org — and all of
    # them stay indistinguishable to the caller. Still returns a session-attached row,
    # which _ensure_pdf_preview depends on: it mutates thumbnail_key/page_count and
    # commits.
    #
    # is_active and the org rule are the SAME gate _require_org_access applies to the
    # messages themselves (app/api/messages.py), and they are here because media was
    # missed when that gate was added. Deactivation is the only lever a superadmin has
    # to cut a tenant off from a cross-org group — it leaves the participant rows in
    # place — and it revoked the message list and the media drawer while the bytes kept
    # serving: /messages 404, /media 404, but the attachment 307 and /meta 200. Anyone
    # holding an attachment id went on reading files out of a group they had been
    # removed from, which is exactly what that comment says the flag exists to prevent.
    #
    # The org half mirrors _require_org_access rather than being stricter than it: a
    # cross_org conversation passes on membership. Adding an allowed_org_ids test here
    # would make media 404 for a caller whose messages still render, and membership
    # already implies the org is in that list (groups.py refuses a cross-org member on
    # create and on add; cross_org.py only ever appends to allowed_org_ids).
    attachment = (
        await db.execute(
            select(MessageAttachment)
            .join(Message, Message.id == MessageAttachment.message_id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .join(
                ConversationParticipant,
                and_(
                    ConversationParticipant.conversation_id == Message.conversation_id,
                    ConversationParticipant.user_id == user.id,
                ),
            )
            .where(
                MessageAttachment.id == att_id,
                Message.deleted_at.is_(None),
                Conversation.is_active.is_(True),
                or_(
                    Conversation.type == ConversationType.cross_org,
                    Conversation.org_id == user.org_id,
                ),
            )
        )
    ).scalar_one_or_none()
    if attachment is None:
        raise _not_found()
    return attachment


@router.get("/media/{attachment_id}")
async def serve_attachment(
    attachment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    attachment = await _member_attachment(db, attachment_id, user)
    url = await storage.presign_get(
        attachment.storage_key,
        filename=attachment.filename,
        inline=_is_inline(attachment.mime_type),
    )
    return RedirectResponse(url, status_code=307)


def _too_large_to_render(attachment: MessageAttachment) -> bool:
    """Would rendering this hold more in memory than the UPLOAD path would accept?

    The upload path declines to buffer a preview source above
    THUMBNAIL_SOURCE_LIMIT — "Previews need the bytes in memory, so they are the
    one thing still bounded by size". The paths that read the same object back
    applied no bound at all, and the self-healing preview below is aimed exactly at
    the files upload declined: its guard runs when a PDF has no thumbnail_key AND
    no page_count, which is precisely the state a >64 MB upload leaves. So the file
    upload refused to buffer was the file a member's first chat open pulled into the
    worker whole, up to max_upload_bytes — 2 GB by default.

    storage._PDF_LIMITER is CapacityLimiter(1), which makes that worse rather than
    better: it serialises the render, so concurrent requests queue and each one is
    already sitting on its own full copy of the file.

    Decided from file_size on the row, so nothing is fetched to find out.
    """
    return storage.too_large_to_render(attachment.file_size)


async def _ensure_pdf_preview(db: AsyncSession, attachment: MessageAttachment) -> None:
    """Render and persist page 1 of a PDF that has never been processed.

    A PDF uploaded before previews existed — or one that arrived by any path that
    skipped thumbnailing — has no thumbnail_key. Rendering on first view and
    persisting makes the feature self-healing rather than dependent on a backfill
    having been run.

    page_count doubles as the "already processed" marker: NULL means never tried,
    0 means tried and the file could not be rendered. Without that, an
    un-renderable PDF would be re-parsed on every single view.

    Called from all three PDF routes rather than just /thumb. /page/{n} used to
    clamp against `page_count or 0`, so for an unprocessed PDF every page 404'd
    and the reader could only ever work if something had happened to fetch the
    thumbnail first.
    """
    if (
        attachment.thumbnail_key
        or attachment.mime_type != "application/pdf"
        or attachment.page_count is not None
    ):
        return
    if _too_large_to_render(attachment):
        # Marked as tried rather than fetched. page_count = 0 is this module's
        # "could not render" marker, so the file keeps its icon and no later view
        # re-enters this path — without it the refusal would be re-made, and the
        # object re-considered, on every single request forever.
        attachment.page_count = 0
        await db.commit()
        return
    data = await storage.get_object(attachment.storage_key)
    thumb, pages = await storage.make_pdf_preview(data)
    if thumb is not None:
        thumb_key = f"{attachment.storage_key}_thumb.jpg"
        await storage.put_object(thumb_key, thumb, "image/jpeg")
        attachment.thumbnail_key = thumb_key
        attachment.page_count = pages
    else:
        attachment.page_count = 0
    await db.commit()


@router.get("/media/{attachment_id}/meta")
async def attachment_meta(
    attachment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Current server-side facts about one attachment.

    Exists because the lazy render above mutates state the client already holds.
    A gallery or starred row is fetched with page_count NULL, rendering its
    thumbnail heals page_count in the database, and the client's copy is now
    stale — so opening the reader from that stale row showed "No preview
    available" for a document whose page 1 was visibly on screen a moment ago.
    The reader asks here when it was handed no page count.

    Membership is enforced by _member_attachment, exactly as for the bytes.
    """
    attachment = await _member_attachment(db, attachment_id, user)
    await _ensure_pdf_preview(db, attachment)
    return {
        "id": str(attachment.id),
        "filename": attachment.filename,
        "mime_type": attachment.mime_type,
        "file_size": attachment.file_size,
        "page_count": attachment.page_count,
        "thumbnail_url": attachment_thumb_url(attachment),
    }


@router.get("/media/{attachment_id}/thumb")
async def serve_attachment_thumb(
    attachment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    attachment = await _member_attachment(db, attachment_id, user)
    await _ensure_pdf_preview(db, attachment)

    if not attachment.thumbnail_key:
        raise _not_found()
    url = await storage.presign_get(attachment.thumbnail_key, inline=True)
    return RedirectResponse(url, status_code=307)


@router.get("/media/{attachment_id}/page/{page_no}")
async def serve_pdf_page(
    attachment_id: str,
    page_no: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """One rendered page of a PDF attachment, as a JPEG.

    Rendered on demand and cached back into object storage, so the first reader
    of a page pays for it and everyone after gets a presigned redirect. Pages are
    produced in windows (storage.PDF_WINDOW) because opening the document
    dominates the cost.

    Membership is enforced by _member_attachment exactly as for the file itself,
    so this cannot become a way to read a PDF you were never sent.
    """
    attachment = await _member_attachment(db, attachment_id, user)
    if attachment.mime_type != "application/pdf":
        raise _not_found()
    # Heal first: an unprocessed PDF has page_count NULL, which the clamp below
    # turns into a total of 0, which 404s every page. Before this, the reader
    # worked only if something else had already fetched the thumbnail.
    await _ensure_pdf_preview(db, attachment)
    # page_count is parser output over untrusted input, so clamp it here as well
    # as at render time — the count itself must never become the attack.
    total = min(attachment.page_count or 0, storage.PDF_MAX_PAGES)
    if page_no < 1 or page_no > total:
        raise _not_found()

    key = f"{attachment.storage_key}_p{page_no}.jpg"
    if not await storage.object_exists(key):
        # Checked here as well as in _ensure_pdf_preview, and not redundantly: a
        # page_count already set does not prove the file is small. The backfill and
        # regenerate tools in app/tools set it with no size bound of their own, and
        # rows predate this limit — either way a window miss would fetch the whole
        # object. The cached-page branch above stays reachable, so a file rendered
        # before this still serves the pages it already has.
        if _too_large_to_render(attachment):
            raise _not_found()
        data = await storage.get_object(attachment.storage_key)
        pages = await storage.render_pdf_window(data, page_no)
        if page_no not in pages:
            raise _not_found()
        for n, jpeg in pages.items():
            await storage.put_object(f"{attachment.storage_key}_p{n}.jpg", jpeg, "image/jpeg")

    url = await storage.presign_get(key, inline=True)
    return RedirectResponse(url, status_code=307)


def _deletion_join(user_id):
    return and_(MessageDeletion.message_id == Message.id, MessageDeletion.user_id == user_id)


async def _sender_names(db: AsyncSession, messages) -> dict:
    sender_ids = {m.sender_id for m in messages if m.sender_id}
    if not sender_ids:
        return {}
    rows = (await db.execute(select(User.id, User.display_name).where(User.id.in_(sender_ids)))).all()
    return {row.id: row.display_name for row in rows}


async def _conversation_links(db: AsyncSession, conv_id, user_id, page: int, limit: int) -> dict:
    """Links shared in a conversation, extracted from message text server-side."""
    deletion_join = _deletion_join(user_id)
    messages = (
        (
            await db.execute(
                select(Message)
                .outerjoin(MessageDeletion, deletion_join)
                .where(
                    Message.conversation_id == conv_id,
                    Message.type != MessageType.system,
                    Message.deleted_at.is_(None),
                    MessageDeletion.message_id.is_(None),  # exclude the caller's delete-for-me
                    # A leading-wildcard match can use no ordinary index, so the
                    # limit below bounded the rows RETURNED while the scan still
                    # read every message in the conversation. The partial index
                    # ix_messages_links carries exactly this predicate, moving
                    # that work to write time; keep the two in lockstep.
                    Message.content.ilike(_LINK_PATTERN),
                )
                .order_by(Message.created_at.desc())
                .limit(_LINK_SCAN_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    sender_names = await _sender_names(db, messages)

    items: list[dict] = []
    for msg in messages:
        content = (msg.content or "").strip()
        for url, domain in _extract_links(content):
            items.append(
                {
                    # iOS's MediaItem declares a non-optional `id` keyed to "_id"
                    # with a synthesized decoder, so a payload without it throws
                    # keyNotFound and takes the whole page down — the gallery has
                    # never been able to decode this endpoint. One message can
                    # yield several links, so the row index keeps it unique;
                    # message_id below stays the thing to jump to.
                    "_id": f"{msg.id}:{len(items)}",
                    "message_id": str(msg.id),
                    "url": url,
                    # No fetching means no remote <title>: the surrounding message
                    # text is the only local context, with the domain as fallback.
                    "title": content[:200] or domain,
                    "domain": domain,
                    "sender_name": sender_names.get(msg.sender_id, "Unknown"),
                    "created_at": iso_z(msg.created_at),
                }
            )

    total = len(items)
    offset = (page - 1) * limit
    return {"data": items[offset : offset + limit], "total": total, "has_more": offset + limit < total}


@router.get("/conversations/{conv_id}/media")
async def conversation_media(
    conv_id: str,
    media_type: str = Query("image", alias="type"),
    page: int = Query(1, ge=1),
    limit: int = Query(30, ge=1, le=100),
    tenant: TenantContext = Depends(get_tenant),
):
    conv_uuid = parse_uuid(conv_id)
    if conv_uuid is None:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")
    if media_type not in _MEDIA_LIST_TYPES and media_type != _LINK_TYPE:
        # Deliberate fix vs the Mongo build, which accepted any type string
        # (type=text returned text messages reshaped as media items).
        raise HTTPException(status_code=400, detail="Invalid media type")
    conv = await tenant.require_membership(conv_uuid)
    # Same revocation gate as the message read paths in api/messages.py: a
    # deactivated conversation keeps its participant rows, so without this the
    # media/docs/links tab still served the whole archive of a group the caller
    # had been cut off from.
    if not conv.is_active:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db = tenant.db

    if media_type == _LINK_TYPE:
        return await _conversation_links(db, conv.id, tenant.user.id, page, limit)

    deletion_join = _deletion_join(tenant.user.id)
    filters = [
        Message.conversation_id == conv.id,
        Message.type == MessageType(media_type),
        Message.deleted_at.is_(None),
        MessageDeletion.message_id.is_(None),  # exclude the caller's delete-for-me
    ]

    total = (
        await db.execute(
            select(func.count(Message.id)).outerjoin(MessageDeletion, deletion_join).where(*filters)
        )
    ).scalar_one()

    messages = (
        (
            await db.execute(
                select(Message)
                .options(selectinload(Message.attachments))
                .outerjoin(MessageDeletion, deletion_join)
                .where(*filters)
                .order_by(Message.created_at.desc())
                .offset((page - 1) * limit)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    sender_names = await _sender_names(db, messages)

    data = []
    for msg in messages:
        first = msg.attachments[0] if msg.attachments else None
        data.append(
            {
                # See the links branch: iOS keys MediaItem.id to "_id" and cannot
                # decode a row without it. One row per message here, so the
                # message id is already unique.
                "_id": str(msg.id),
                "message_id": str(msg.id),
                "media_url": media_url_for(first.id) if first else None,
                # Same helper the message serializer uses, deliberately: gating on
                # thumbnail_key alone withheld the URL from PDFs whose page 1 has
                # not been rendered yet, and that URL is precisely what triggers
                # the render. The Docs tab showed a generic icon forever and the
                # reader opened empty.
                "thumbnail_url": attachment_thumb_url(first) if first else None,
                "filename": first.filename if first else "",
                "file_size": first.file_size if first else 0,
                # Both added so the web gallery can open a document rather than
                # only jump to its message: mime_type is how it decides a row is
                # a PDF (the filename extension is user-controlled), and
                # page_count is what PdfViewer pages over — without it the
                # reader opens on "No preview available". Purely additive; the
                # iOS client decodes this payload too and ignores unknown keys.
                "mime_type": first.mime_type if first else None,
                "page_count": first.page_count if first else None,
                "sender_name": sender_names.get(msg.sender_id, "Unknown"),
                "created_at": iso_z(msg.created_at),
            }
        )

    return {"data": data, "total": total, "has_more": page * limit < total}
