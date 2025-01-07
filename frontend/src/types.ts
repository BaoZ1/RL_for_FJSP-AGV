export const operationStatusMapper = {
  0: "blocked",
  1: "unscheduled",
  2: "waiting",
  3: "processing",
  4: "finished"
} as const

export type OperationStatusIdx = keyof typeof operationStatusMapper

export type OperationStatus = (typeof operationStatusMapper)[OperationStatusIdx]

export type OperationState = {
  id: number
  status: OperationStatusIdx
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
  pos: { x: number, y: number }
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
  direct_paths: [number, number][]
  next_operation_id: number
  next_machine_id: number
  next_AGV_id: number
}

export const actionStatusMapper = {
  0: "move",
  1: "pick",
  2: "transport",
  3: "wait"
} as const

export type ActionStatusIdx = keyof typeof actionStatusMapper

export type ActionStatus = (typeof actionStatusMapper)[ActionStatusIdx]

export type Action = {
  action_type: number
  AGV_id: number | null
  target_machine: number | null
  target_product: { from: number, to: number } | null
}

export type AddOperationParams = {
  machine_type: number
  process_time: number
  pred: number | null
  succ: number | null
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

export type PredictProgress = {
  round_count: number
  finished_step: number
  total_step: number
  graph_state: EnvState
  action: Action
}