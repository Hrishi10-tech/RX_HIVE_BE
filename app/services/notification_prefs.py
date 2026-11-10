"""Who wants to be notified about what: timed mutes and per-user push preferences.

One module so the two questions every notification path has to ask — "is this
chat muted right now?" and "does this user want this kind of push?" — have one
answer each. Mute was already the cautionary tale: is_muted was stored and served
for a long time while nothing read it, and a timed mute adds a second way to get
it wrong (reading is_muted and ignoring that the mute has lapsed).
"""

import datetime as dt
from typing import Any

from app.db.models import ConversationParticipant
from app.utils import now_utc

# Every preference and its default. Stored rows hold only what the user changed,
# so a key added here is live for everyone immediately with no backfill.
#   show_preview  push body carries message text (False: "New message")
#   messages      push for direct-chat messages
#   groups        push for group / cross-org group messages
#   calls         push for incoming and missed calls
NOTIFICATION_PREF_DEFAULTS: dict[str, bool] = {
    "show_preview": True,
    "messages": True,
    "groups": True,
    "calls": True,
}

# What PUT /mute accepts. "always" and "off" are not durations, but they share the
# one field so a client has a single control to drive.
MUTE_DURATIONS: dict[str, dt.timedelta | None] = {
    "8h": dt.timedelta(hours=8),
    "1w": dt.timedelta(weeks=1),
    "always": None,
}


def effective_prefs(stored: Any) -> dict[str, bool]:
    """The stored sparse dict merged over the defaults, unknown keys dropped.

    Defensive about the stored shape: only real booleans override a default, so a
    malformed row degrades to "notify" rather than raising on the send path.
    """
    prefs = dict(NOTIFICATION_PREF_DEFAULTS)
    if isinstance(stored, dict):
        for key in prefs:
            value = stored.get(key)
            if isinstance(value, bool):
                prefs[key] = value
    return prefs


def is_muted_now(participant: ConversationParticipant | None, now: dt.datetime | None = None) -> bool:
    """Muted, and either "always" or not yet lapsed. The only correct reading of mute."""
    if participant is None or not participant.is_muted:
        return False
    until = participant.muted_until
    if until is None:
        return True
    if until.tzinfo is None:
        until = until.replace(tzinfo=dt.UTC)
    return until > (now or now_utc())


def active_muted_until(participant: ConversationParticipant | None, now: dt.datetime | None = None):
    """muted_until as served on the wire: only while the mute is in force.

    A lapsed timestamp is not sent back — the client would otherwise have to
    repeat the expiry check to know the chat is no longer muted.
    """
    if participant is None or not is_muted_now(participant, now):
        return None
    return participant.muted_until
