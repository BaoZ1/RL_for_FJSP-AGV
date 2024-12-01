from pydantic import BaseModel, Field


class OperationState(BaseModel):
    id: int
    status: int
    machine_type: int
    process_time: float
    processing_machine: int | None
    finish_timestamp: float
    predecessors: list[int]
    arrived_preds: list[int]
    successors: list[int]
    sent_succs: list[int]


class ProductState(BaseModel):
    from_: int = Field(alias="from")
    to: int
    
class Position(BaseModel):
    x: float
    y: float

class MachineState(BaseModel):
    id: int
    type: int
    pos: Position
    status: int
    working_operation: int | None
    waiting_operation: int | None
    materials: list[ProductState]
    products: list[ProductState]


class AGVState(BaseModel):
    id: int
    status: int
    speed: float
    position: int
    target_machine: int
    loaded_item: ProductState | None
    target_item: ProductState | None
    finish_timestamp: float


class EnvState(BaseModel):
    inited: bool
    timestamp: float
    operations: list[OperationState]
    machines: list[MachineState]
    AGVs: list[AGVState]
    paths: dict[int, dict[int, list[int]]]
    # distances: dict[int, dict[int, float]]
    next_operation_id: int
    next_machine_id: int
    next_AGV_id: int


class GenerationParams(BaseModel):
    operation_count: int
    machine_count: int
    AGV_count: int
    machine_type_count: int
    min_transport_time: float
    max_transport_time: float
    min_max_speed_ratio: float
    min_process_time: float
    max_process_time: float