from pydantic import BaseModel


class UavComputeRequest(BaseModel):
    user_id: int


class UavComputeResponse(BaseModel):
    user_id: int
