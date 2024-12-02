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

if __name__ == "__main__":
    data = r'{"inited":false,"timestamp":0,"operations":[{"id":0,"status":0,"machine_type":0,"process_time":0,"processing_machine":null,"finish_timestamp":0,"predecessors":[],"arrived_preds":[],"successors":[1,2,3,6,10],"sent_succs":[]},{"id":1,"status":0,"machine_type":2,"process_time":10.9037446975708,"processing_machine":null,"finish_timestamp":0,"predecessors":[0],"arrived_preds":[],"successors":[9999],"sent_succs":[]},{"id":2,"status":0,"machine_type":2,"process_time":11.970909118652344,"processing_machine":null,"finish_timestamp":0,"predecessors":[0],"arrived_preds":[],"successors":[7],"sent_succs":[]},{"id":3,"status":0,"machine_type":3,"process_time":13.022899627685547,"processing_machine":null,"finish_timestamp":0,"predecessors":[0],"arrived_preds":[],"successors":[4,8],"sent_succs":[]},{"id":4,"status":0,"machine_type":3,"process_time":14.04312515258789,"processing_machine":null,"finish_timestamp":0,"predecessors":[3],"arrived_preds":[],"successors":[5,7],"sent_succs":[]},{"id":5,"status":0,"machine_type":1,"process_time":13.21705436706543,"processing_machine":null,"finish_timestamp":0,"predecessors":[4],"arrived_preds":[],"successors":[7],"sent_succs":[]},{"id":6,"status":0,"machine_type":2,"process_time":9.869637489318848,"processing_machine":null,"finish_timestamp":0,"predecessors":[0],"arrived_preds":[],"successors":[9999],"sent_succs":[]},{"id":7,"status":0,"machine_type":1,"process_time":13.654438018798828,"processing_machine":null,"finish_timestamp":0,"predecessors":[2,4,5],"arrived_preds":[],"successors":[9],"sent_succs":[]},{"id":8,"status":0,"machine_type":2,"process_time":11.274375915527344,"processing_machine":null,"finish_timestamp":0,"predecessors":[3],"arrived_preds":[],"successors":[9999],"sent_succs":[]},{"id":9,"status":0,"machine_type":1,"process_time":14.632046699523926,"processing_machine":null,"finish_timestamp":0,"predecessors":[7],"arrived_preds":[],"successors":[9999],"sent_succs":[]},{"id":10,"status":0,"machine_type":2,"process_time":9.365913391113281,"processing_machine":null,"finish_timestamp":0,"predecessors":[0],"arrived_preds":[],"successors":[9999],"sent_succs":[]},{"id":9999,"status":0,"machine_type":0,"process_time":0,"processing_machine":null,"finish_timestamp":0,"predecessors":[1,6,8,9,10],"arrived_preds":[],"successors":[],"sent_succs":[]}],"machines":[{"id":0,"type":0,"pos":{"x":0,"y":0},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]},{"id":1,"type":1,"pos":{"x":29.255443572998047,"y":15.540838241577148},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]},{"id":2,"type":2,"pos":{"x":16.240585327148438,"y":34.199440002441406},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]},{"id":3,"type":3,"pos":{"x":9.248323440551758,"y":44.0601692199707},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]},{"id":4,"type":3,"pos":{"x":19.550159454345703,"y":27.746294021606445},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]},{"id":5,"type":3,"pos":{"x":40.19685363769531,"y":25.10924530029297},"status":0,"working_operation":null,"waiting_operation":null,"materials":[],"products":[]}],"AGVs":[{"id":0,"status":0,"speed":8.77242660522461,"position":0,"target_machine":0,"loaded_item":null,"target_item":null,"finish_timestamp":0},{"id":1,"status":0,"speed":9.91007137298584,"position":0,"target_machine":0,"loaded_item":null,"target_item":null,"finish_timestamp":0},{"id":2,"status":0,"speed":9.787772178649902,"position":0,"target_machine":0,"loaded_item":null,"target_item":null,"finish_timestamp":0},{"id":3,"status":0,"speed":8.71249771118164,"position":0,"target_machine":0,"loaded_item":null,"target_item":null,"finish_timestamp":0},{"id":4,"status":0,"speed":9.370518684387207,"position":0,"target_machine":0,"loaded_item":null,"target_item":null,"finish_timestamp":0}],"paths":{"0":{"0":[0],"1":[1],"2":[4,2],"3":[3],"4":[4],"5":[3,5]},"1":{"0":[0],"1":[1],"2":[4,2],"3":[0,3],"4":[4],"5":[0,3,5]},"2":{"0":[4,0],"1":[4,1],"2":[2],"3":[4,0,3],"4":[4],"5":[4,0,3,5]},"3":{"0":[0],"1":[0,1],"2":[0,4,2],"3":[3],"4":[0,4],"5":[5]},"4":{"0":[0],"1":[1],"2":[2],"3":[0,3],"4":[4],"5":[0,3,5]},"5":{"0":[3,0],"1":[3,0,1],"2":[3,0,4,2],"3":[3],"4":[3,0,4],"5":[5]}},"next_operation_id":11,"next_machine_id":6,"next_AGV_id":5}'
    print(EnvState.model_validate_json(data).model_dump())