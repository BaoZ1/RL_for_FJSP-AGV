export type OperationState = {
    id: number
    status: number
    machine_type: number
    processing_machine: number | null
    finish_timestamp: number
    predecessors: number[]
    arrived_preds: number[]
    successors: number[]
    sent_succs: number[]
}

export type MachineState = {
    id: number
    type: number
    pos: {x: number, y: number}
    status: number
    working_operation: number | null
    waiting_operation: number | null
    materials: { from: number, to: number }[]
    products: { from: number, to: number }[]
}

export type AGVState = {
    id: number
    status: number
    speed: number
    position: number
    target_machine: number
    loaded_item: { from: number, to: number } | null
    target_item: { from: number, to: number } | null
    finish_timestamp: number
}

export type EnvState = {
    inited: boolean
    timestamp: number
    operations: OperationState[]
    machines: MachineState[]
    AGVs: AGVState[]
    paths: {[from: number]: {[to: number]: number[]}}
    // distances: Record<number, Record<number, number>>
    next_operation_id: number
    next_machine_id: number
    next_AGV_id: number
}

export type GenerationParams = {
    operation_count: number
    machine_count: number
    AGV_count: number
    machine_type_count: number
    min_transport_time: number
    max_transport_time: number
    min_max_speed_ratio: number
    min_process_time: number
    max_process_time: number
}