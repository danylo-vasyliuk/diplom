from fastapi import APIRouter

from uav_service.views.models import UavComputeRequest, UavComputeResponse

router = APIRouter(prefix="/uav")


@router.post("/compute/", status_code=200)
async def start(
    *,
    request_data: UavComputeRequest,
) -> UavComputeResponse:
    return UavComputeResponse(user_id=request_data.user_id)
