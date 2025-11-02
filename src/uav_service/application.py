from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(uav_router, prefix=base_api_path)

    return app
