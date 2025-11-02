from fastapi import FastAPI

from uav_service.application import make_fastapi_app
from uav_service.settings import settings


def build_app() -> FastAPI:
    return make_fastapi_app(
        title=settings.misc.title,
        base_api_path=settings.misc.base_api_path,
    )
