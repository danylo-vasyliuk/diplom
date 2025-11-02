FROM python:3.12-slim-bookworm AS base

ARG USERNAME=app
ARG HOMEDIR=/home/$USERNAME
ARG WORKDIR=/usr/src/app
ARG VIRTUAL_ENV=/usr/src/env
ARG UID=1313
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -d "$HOMEDIR" -N -G users -u "$UID" "$USERNAME" \
    && mkdir "$WORKDIR" && chown "$USERNAME:users" "$WORKDIR"
WORKDIR $WORKDIR

FROM base AS venv

ENV PIP_CACHE_DIR=/var/cache/pip
ENV POETRY_CACHE_DIR=/var/cache/poetry
ENV POETRY_VIRTUALENVS_CREATE=0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev

RUN --mount=type=cache,id=pip,target=$PIP_CACHE_DIR,sharing=locked \
    pip install poetry==2.1.2

RUN python -m venv "$VIRTUAL_ENV"

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./pyproject.toml ./poetry.lock ./
RUN --mount=type=cache,id=poetry,target=$POETRY_CACHE_DIR,sharing=locked \
    poetry install --no-root

FROM base AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV UVICORN_ACCESS_LOG=0

EXPOSE 8000

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y libpq5

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=venv $VIRTUAL_ENV $VIRTUAL_ENV

COPY . ./
RUN pip install --no-deps  -e .

USER app

ENTRYPOINT ["docker/start.sh"]
