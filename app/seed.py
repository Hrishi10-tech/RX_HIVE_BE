"""Idempotent seed: ensures the super-admin exists. Run with `python -m app.seed`.

Credentials come exclusively from RXHIVE_SEED_SUPERADMIN_EMAIL / _PASSWORD —
no hardcoded defaults ship in code. Safe to run on every boot.
"""

import asyncio
import logging

from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.models import User, UserRole
from app.db.session import SessionLocal, engine

logger = logging.getLogger("rxhive.seed")


async def seed() -> None:
    settings = get_settings()
    if not settings.seed_superadmin_email or not settings.seed_superadmin_password:
        logger.info("[seed] superadmin credentials not set — skipping")
        return
    async with SessionLocal() as db:
        existing = (
            await db.execute(select(User).where(User.email == settings.seed_superadmin_email))
        ).scalar_one_or_none()
        if existing:
            logger.info("[seed] superadmin already present")
            return
        db.add(
            User(
                email=settings.seed_superadmin_email,
                display_name="Super Admin",
                password_hash=await hash_password(settings.seed_superadmin_password),
                role=UserRole.superadmin,
                is_active=True,
            )
        )
        await db.commit()
        logger.info("[seed] superadmin created: %s", settings.seed_superadmin_email)


async def main() -> None:
    await seed()
    await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
