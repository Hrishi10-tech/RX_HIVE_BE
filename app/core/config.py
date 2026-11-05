from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secrets shipped in .env.example / code defaults. Booting production with any
# of these is a hard error — they are public.
_KNOWN_PLACEHOLDER_SECRETS = {
    "dev-only-secret-change-in-production",
    "change-me-openssl-rand-hex-32",
    "devsecret-at-least-32-characters-long",
    "rxhive-dev",
    "change-me-strong-password",
    "change-me-32-chars-minimum-secret",
    "",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RXHIVE_", env_file=".env", extra="ignore")

    app_name: str = "RX HIVE"
    environment: str = "development"  # development | production | test

    database_url: str = "postgresql+asyncpg://rxhive:rxhive@localhost:5432/rxhive"
    redis_url: str = "redis://localhost:6379/0"

    secret_key: str = "dev-only-secret-change-in-production"
    access_token_minutes: int = 15
    refresh_token_days: int = 30

    # How long after a refresh token is rotated it may still be replayed once, by
    # a client that never received the rotation response (request timeout, proxy
    # error, app suspended mid-flight). Wide enough to cover a lost round trip,
    # short enough that a stolen cookie is out of time; a replay outside it — or
    # of a successor already spent — is treated as theft and burns the family.
    refresh_reuse_grace_seconds: int = 60

    # None → derive from environment (Secure cookies in production). Set
    # RXHIVE_COOKIE_SECURE explicitly to force a value (e.g. false for local
    # HTTPS-less testing). Read via the `cookies_secure` property.
    cookie_secure: bool | None = None
    cookie_domain: str | None = None

    cors_origins: str = ""  # comma-separated exact origins; empty = same-origin only

    # True when running behind a trusted reverse proxy (the bundled Caddy): the
    # real client IP is taken from the LAST X-Forwarded-For hop for rate limiting
    # — the one the proxy appended from the TCP peer, and the only entry in that
    # header a caller cannot forge. See core/rate_limit._client_ip. Only enable
    # when the proxy is the sole ingress, which is what makes "last" the right
    # hop to take.
    trust_proxy: bool = False

    # Per-IP requests allowed per 60s window. 0 disables that limiter.
    # Raise for E2E runs where all traffic shares one source IP.
    rate_limit_login: int = 10
    rate_limit_refresh: int = 30
    rate_limit_password: int = 5
    rate_limit_upload: int = 30
    rate_limit_push_subscribe: int = 10

    s3_endpoint: str = "http://localhost:9000"
    # Endpoint the *browser* uses for presigned URLs. May be a bare path like "/s3"
    # (the bundled Caddy route) or a full origin like "https://cdn.example.com".
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "rxhive"
    s3_secret_key: str = "rxhive-dev"
    s3_bucket: str = "rxhive-attachments"
    s3_region: str = "us-east-1"
    presign_expiry_seconds: int = 300

    # URL handed to browsers for the SFU websocket. May be a bare path like
    # "/livekit" (the bundled Caddy route) — a browser resolves that against the
    # site origin, but the API server cannot, so /api/health probes
    # `livekit_health_url` when the public URL is path-style.
    livekit_url: str = "ws://localhost:7880"
    # Server-side address of the SFU, used only by the /api/health probe.
    # Empty → derive from livekit_url. Set this when livekit_url is path-style
    # (docker compose sets it to the service address: http://livekit:7880).
    livekit_health_url: str = ""
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecret-at-least-32-characters-long"

    # One upload ceiling for every kind of file, read by services/storage.py.
    #
    # 2 GB is not arbitrary. Starlette spools a multipart upload to a temp file
    # once it passes 1 MB, and media.py hands that file object straight to S3, so
    # the API never holds the payload in memory — but it does hold it on the
    # instance's DISK for the length of the request. Two or three concurrent
    # uploads at this size are what the box can absorb, not twenty. Lower it on a
    # smaller instance.
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    password_min_length: int = 10

    seed_superadmin_email: str = "admin@rhythmrx.ai"
    seed_superadmin_password: str = ""

    # Web Push (VAPID). Generate a keypair with `python -m app.tools.vapid`.
    vapid_public_key: str = ""
    vapid_private_key: str = ""
    vapid_subject: str = "mailto:admin@rhythmrx.ai"

    @model_validator(mode="after")
    def _reject_placeholder_secrets_in_prod(self):
        """In production, refuse to boot with a public/placeholder or too-short
        secret. This closes the 'default JWT key ships to prod' auth bypass."""
        if self.environment != "production":
            return self
        checks = {
            "RXHIVE_SECRET_KEY": self.secret_key,
            "RXHIVE_LIVEKIT_API_SECRET": self.livekit_api_secret,
        }
        # S3 keys are only checked when static credentials are actually in use.
        # Leaving BOTH empty is the deployed AWS path: storage.py then pulls
        # short-lived credentials from the instance role, so there is no secret
        # to validate — and demanding one here would refuse to boot on EC2.
        if self.s3_access_key or self.s3_secret_key:
            checks["RXHIVE_S3_SECRET_KEY"] = self.s3_secret_key
        for name, value in checks.items():
            if value in _KNOWN_PLACEHOLDER_SECRETS:
                raise ValueError(f"{name} is a known placeholder — set a real secret for production")
            if len(value) < 32:
                raise ValueError(f"{name} must be at least 32 characters in production")
        if not self.seed_superadmin_password or self.seed_superadmin_password in _KNOWN_PLACEHOLDER_SECRETS:
            # allowed to be empty only if a superadmin already exists; seed skips it
            pass
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def livekit_probe_url(self) -> str | None:
        """HTTP(S) address the API can GET to see whether the SFU is alive.

        LiveKit serves a plain 200 on `/` over the same port as the signalling
        websocket, so the ws→http translation of the client URL is a valid
        liveness probe. Returns None when nothing probeable is configured
        (unset, or a path-style URL only a browser can resolve).
        """
        raw = (self.livekit_health_url or self.livekit_url or "").strip()
        if not raw:
            return None
        if raw.startswith("ws://"):
            raw = "http://" + raw[len("ws://") :]
        elif raw.startswith("wss://"):
            raw = "https://" + raw[len("wss://") :]
        if not raw.startswith(("http://", "https://")):
            # e.g. "/livekit" — resolvable by the browser, not by this process.
            return None
        return raw

    @property
    def cookies_secure(self) -> bool:
        """Secure cookie flag: explicit override, else on in production."""
        if self.cookie_secure is not None:
            return self.cookie_secure
        return self.is_production


@lru_cache
def get_settings() -> Settings:
    return Settings()
