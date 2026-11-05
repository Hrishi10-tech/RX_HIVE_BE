FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app

# Drop root. This is the internet-facing process — it parses untrusted uploads
# with Pillow and pypdfium2, both of which decode attacker-supplied bytes in C —
# and it was running as uid 0 with the full default capability set, so a memory
# bug in either was a root shell in the container rather than a crash.
#
# Added after the COPYs so the application files stay root-owned and read-only
# to the runtime user. Nothing here writes to /srv: uploads stream to S3, and
# Starlette spools large request bodies to /tmp, which stays world-writable.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin rxhive
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
