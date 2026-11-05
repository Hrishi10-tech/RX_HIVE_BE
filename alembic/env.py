import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings
from app.db.models import Base, include_in_autogenerate

config = context.config
# `%%`, not `%`: alembic's Config is a configparser with BasicInterpolation, so
# set_main_option raises ValueError("invalid interpolation syntax") on any bare
# percent in the value — before a single migration runs.
#
# A percent is not exotic here, it is the norm the bootstrap creates. The DSN is
# assembled in infra/terraform/user_data.sh.tftpl, which urlencodes the database
# password precisely because a reserved character in it would otherwise corrupt
# the URL — so a password containing any of + ! # $ * , ; = & ' ( ) : arrives as
# %XX. Nothing but terraform's override_special = "-_.~" keeps that from being
# true today, and it stops being true the moment anyone rotates the RDS master
# password by hand or lets RDS generate one.
#
# The blast radius is the whole API, not just migrations: production boots with
# `alembic upgrade head && python -m app.seed && uvicorn ...`
# (infra/docker-compose.prod.yml), so this module raising means uvicorn is never
# reached. The app itself would have been fine — SQLAlchemy unquotes %XX — which
# is what makes the failure so hard to read from the traceback.
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # None (the default) leaves the version table unqualified, so on a
        # non-default search_path Alembic reads whichever alembic_version it
        # finds first and can conclude an empty schema is already at head.
        version_table_schema=config.attributes.get("version_table_schema"),
        # Consulted only by `alembic revision --autogenerate`, and only here:
        # autogenerate runs online, so the offline path has nothing to filter.
        # Without it autogenerate proposes dropping messages.search_tsv, its GIN
        # index and three measured indexes — see app/db/models.py for the list
        # and why each one lives in a migration rather than in the metadata.
        include_object=include_in_autogenerate,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    # A caller can hand us a live connection via config.attributes (the test
    # harness does, so migrations land on that connection's search_path instead
    # of a fresh engine's). Without this hook there is no way to migrate
    # anywhere but the URL's default schema.
    connection = config.attributes.get("connection")
    if connection is not None:
        do_run_migrations(connection)
    else:
        asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
