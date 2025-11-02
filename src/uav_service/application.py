from fastapi import FastAPI

from uav_service.views.routers import router as uav_router


def make_fastapi_app(
    title: str,
    base_api_path: str,
) -> FastAPI:
    app = FastAPI(
        title=title,
        openapi_url=f"{base_api_path}/docs/json/",
        docs_url=f"{base_api_path}/docs/swagger/",
        redoc_url=f"{base_api_path}/docs/redoc/",
    )

    app.include_router(uav_router, prefix=base_api_path)

    return app
