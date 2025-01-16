import { FC } from "react"

export type BaseFC<ExtraProps extends Record<string, any> = {}> = FC<{className?: string} & ExtraProps>

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
  process_time: number
  processing_machine: number | null
  finish_timestamp: number
  predecessors: number[]
  arrived_preds: number[]
  successors: number[]
  sent_succs: number[]
}

export const MachineStatusMapper = {
  0: "idle",
  1: "waiting_material",
  2: "working"
} as const

export type MachineStatusIdx = keyof typeof MachineStatusMapper

export type MachineStatus = (typeof MachineStatusMapper)[MachineStatusIdx]

export type MachineState = {
  id: number
  type: number
  pos: { x: number, y: number }
  status: MachineStatusIdx
  working_operation: number | null
  waiting_operation: number | null
  materials: { from: number, to: number }[]
  products: { from: number, to: number }[]
}

export const AGVStatusMapper = {
  0: "idle",
  1: "moving",
  2: "picking",
  3: "transporting"
} as const

export type AGVStatusIdx = keyof typeof AGVStatusMapper

export type AGVStatus = (typeof AGVStatusMapper)[AGVStatusIdx]

export type AGVState = {
  id: number
  status: AGVStatusIdx
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

export type Paths = {
  [from: number]: {
    [to: number]: [number[], number]
  }
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
  action_type: ActionStatusIdx
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

export type AddMachineParams = {
  type: number
  x: number
  y: number
  pathTo: number[]
}

export type AddAGVParams = {
  speed: number
  init_pos: number
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
  simple_mode: boolean
}

export type PredictProgress = {
  round_count: number
  finished_step: number
  total_step: number
  graph_state: EnvState
  action: Action
}