from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    field_serializer,
    SerializationInfo,
    PlainValidator,
    PlainSerializer,
    ValidationInfo,
)
from FJSP_env import *
from typing import Annotated

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
    
    @model_validator(mode="before")
    @classmethod
    def to_dict(cls, data):
        if isinstance(data, Product):
            return {
                "from": data.operation_from,
                "to": data.operation_to,
            }
        return data


class Position(BaseModel):
    x: float
    y: float

class MachineState(BaseModel):
    id: int
    type: int
    pos: Position
    status: int
    working_operation: int | None
    waiting_operations: list[int]
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
    direct_paths: set[tuple[int, int]]
    next_operation_id: int
    next_machine_id: int
    next_AGV_id: int
    
    @model_validator(mode="before")
    @classmethod
    def to_dict(cls, data):
        if isinstance(data, Graph):
            return data.get_state()
        return data


class GenerationParamModel(BaseModel):
    operation_count: int
    machine_count: int
    AGV_count: int
    machine_type_count: int
    min_transport_time: float
    max_transport_time: float
    min_max_speed_ratio: float
    min_process_time: float
    max_process_time: float

class ActionState(BaseModel):
    AGV_id: int | None = None
    action_type: ActionType
    target_machine: int | None = None
    target_product: ProductState | None = None
    
    @model_validator(mode="before")
    @classmethod
    def to_dict(cls, data):
        if isinstance(data, Action):
            return {
                "AGV_id": data.AGV_id,
                "action_type": data.action_type,
                "target_machine": data.target_machine,
                "target_product": data.target_product,
            }
        else:
            return data

    @field_validator("action_type", mode="plain")
    def check_type(v, info: ValidationInfo):
        match info.mode:
            case "json":
                assert isinstance(v, int)
                return ActionType(v)
            case "python":
                assert isinstance(v, ActionType)
                return v


    @field_serializer("action_type", mode="plain")
    def serialize(v, info: SerializationInfo):
        match info.mode:
            case "json":
                return int(v)
            case "python":
                return v

class PredictProgress(BaseModel):
    round_count: int
    finished_step: int
    total_step: int
    graph_state: EnvState
    action: ActionState

if __name__ == "__main__":
    with open(r"private\test_1.json", "r") as f:
        state = EnvState.model_validate_json(f.read())
    import json
    json.dumps(state.model_dump_json())  
