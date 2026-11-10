"""Link preview routes: GET /api/link-preview and its preview images.

Both require a signed-in user (get_current_user, which also enforces the
forced-password-change gate). See services/link_preview.py for the fetch rules.
"""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.core.rate_limit import link_preview_limiter
from app.db.models import User
from app.realtime.redis_bus import get_redis
from app.services import link_preview
from app.services.link_preview_parse import MAX_URL_LENGTH, cache_key, normalize_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/link-preview", tags=["link-preview"])

_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


@router.get("")
async def get_link_preview(
    request: Request,
    url: str = Query(..., max_length=MAX_URL_LENGTH),
    _: User = Depends(get_current_user),
):
    """`{"preview": {...}}`, or `{"preview": null}` when there is nothing to show.

    Never an error for an unreachable or unsuitable page — "no preview" is a
    normal answer, and the client simply shows the plain link.
    """
    if not get_settings().link_previews_enabled:
        return {"preview": None}
    normalized = normalize_url(url)
    if normalized is None:
        return {"preview": None}
    # Rate-limit only real fetches: a cached answer costs nothing, and every
    # member of a busy group asking about the same link must not burn the budget.
    try:
        cached = await get_redis().exists(f"lp:meta:{cache_key(normalized)}")
    except Exception:
        cached = False
    if not cached:
        await link_preview_limiter(request)
    try:
        return {"preview": await link_preview.get_preview(normalized)}
    except Exception:
        # Redis down, parser surprise: degrade to "no preview", never a 500 in chat.
        logger.exception("link preview failed for %s", normalized)
        return {"preview": None}


@router.get("/image/{key}")
async def get_link_preview_image(key: str, _: User = Depends(get_current_user)):
    if not _KEY_RE.match(key):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        data = await link_preview.get_cached_image(key)
    except Exception:
        data = None
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )
