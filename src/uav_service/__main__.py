import uvicorn


def run() -> None:
    uvicorn_params: dict[str, object] = {
        "factory": True,
        "port": 8000,
        "host": "0.0.0.0",
        "reload": True,
    }
    uvicorn.run(
        "uav_service.asgi:build_app",
        **uvicorn_params,  # type: ignore
    )


if __name__ == "__main__":
    run()
