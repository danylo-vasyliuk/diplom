from pydantic import BaseModel


class Coordinates(BaseModel):
    x: float
    y: float


class Coordinates3D(Coordinates):
    z: float


class Drone(BaseModel):
    label: str
    coordinates: Coordinates3D
