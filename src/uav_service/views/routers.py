from fastapi import APIRouter

from uav_service.logic.compute import compute_drone_positions
from uav_service.logic.models import Coordinates3D, Drone
from uav_service.views.models import UavComputeRequest, UavComputeResponse

router = APIRouter(prefix="/uav")


@router.post("/compute/", status_code=200)
async def start(
    *,
    request_data: UavComputeRequest,
) -> UavComputeResponse:
    base = Coordinates3D(x=0, y=0, z=0)
    initial_drone_positions = [
        Drone(label="UAV_1", coordinates=Coordinates3D(x=10, y=5, z=10)),
        Drone(label="UAV_2", coordinates=Coordinates3D(x=20, y=10, z=12)),
        Drone(label="UAV_3", coordinates=Coordinates3D(x=30, y=15, z=15)),
        Drone(label="UAV_4", coordinates=Coordinates3D(x=40, y=20, z=17)),
        Drone(label="UAV_5", coordinates=Coordinates3D(x=50, y=25, z=18)),
    ]
    base_coordinates = request_data.base or base
    drones = request_data.initial_drone_positions or initial_drone_positions

    drone_positions = compute_drone_positions(
        user_coordinates=request_data.user,
        base_coordinates=base,
        drones=drones,
        num_to_use=3,
    )
    return UavComputeResponse(
        base_coordinates=base_coordinates,
        user_coordinates=request_data.user,
        drone_positions=drone_positions,
    )
