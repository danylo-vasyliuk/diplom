from pydantic import BaseModel

from uav_service.logic.models import Coordinates, Coordinates3D, Drone


class UavComputeRequest(BaseModel):
    user: Coordinates


class UavComputeResponse(BaseModel):
    drones: list[Drone]
