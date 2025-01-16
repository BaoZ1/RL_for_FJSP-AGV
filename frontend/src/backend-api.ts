import { AddAGVParams, AddMachineParams, AddOperationParams, AGVState, EnvState, GenerationParams, Paths, PredictProgress } from "./types"
import { fetch } from "@tauri-apps/plugin-http"
import WebSocket from '@tauri-apps/plugin-websocket';

const PORT = 8000

const BASE_PATH = `http://localhost:${PORT}`

const handleError = async (response: Response) => {
  if (!response.ok) {
    if(response.status === 503) {
      throw "服务器异常"
    }
    const info = await response.text()
    console.log(response, info)
    throw info
  }
}

export const newEnv = async () => {
  const url = new URL("env", BASE_PATH)
  const response = await fetch(url, { method: "GET"})
  await handleError(response)
  return await response.json() as EnvState
}

export const loadEnv = async (path: string) => {
  const url = new URL("env/load", BASE_PATH)
  url.searchParams.append("path", path)
  const response = await fetch(url, { method: "GET" })
  await handleError(response)
  return await response.json() as EnvState
}

export const saveEnv = async (path: string, state: EnvState) => {
  const url = new URL("env/save", BASE_PATH)
  url.searchParams.append("path", path)
  const response = await fetch(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
}

export const randEnv = async (params: GenerationParams) => {
  const url = new URL("env/rand", BASE_PATH)
  Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v.toString()))
  const response = await fetch(url, { method: "GET" })
  await handleError(response)
  return await response.json() as EnvState
}

export const initEnv = async (state: EnvState) => {
  const url = new URL("env/init", BASE_PATH)
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const getPaths = async (state: EnvState) => {
  const url = new URL("env/paths", BASE_PATH)
  const response = await fetch(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as Paths
}

export const addOperation = async (state: EnvState, params: AddOperationParams) => {
  console.log(params);

  const url = new URL("operation/add", BASE_PATH)
  url.searchParams.append("type", params.machine_type.toString())
  url.searchParams.append("time", params.process_time.toString())
  params.pred && url.searchParams.append("pred", params.pred.toString())
  params.succ && url.searchParams.append("succ", params.succ.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const removeOperation = async (state: EnvState, target: number) => {
  const url = new URL("operation/remove", BASE_PATH)
  url.searchParams.append("id", target.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  const ret = await response.json()
  console.log(ret);
  return ret as EnvState
}

export const addMachine = async (state: EnvState, params: AddMachineParams) => {
  const url = new URL("machine/add", BASE_PATH)
  url.searchParams.append("type", params.type.toString())
  url.searchParams.append("x", params.x.toString())
  url.searchParams.append("y", params.y.toString())
  for (const id of params.pathTo) {
    url.searchParams.append("path_to", id.toString())
  }
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const removeMachine = async (state: EnvState, id: number) => {
  const url = new URL("machine/remove", BASE_PATH)
  url.searchParams.append("id", id.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const addPath = async (state: EnvState, a: number, b: number) => {
  const url = new URL("path/add", BASE_PATH)
  url.searchParams.append("a", a.toString())
  url.searchParams.append("b", b.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const removePath = async (state: EnvState, a: number, b: number) => {
  const url = new URL("path/remove", BASE_PATH)
  url.searchParams.append("a", a.toString())
  url.searchParams.append("b", b.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const addAGV = async (state: EnvState, params: AddAGVParams) => {
  const url = new URL("agv/add", BASE_PATH)
  url.searchParams.append("speed", params.speed.toString())
  url.searchParams.append("init_pos", params.init_pos.toString())
  const response = await fetch(
    url,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    }
  )
  await handleError(response)
  return await response.json() as EnvState
}

export const modelList = async () => {
  const url = new URL("model/list", BASE_PATH)
  const response = await fetch(url)
  await handleError(response)
  return await response.json() as string[]
}

export const loadModel = async (model_path: string) => {
  const url = new URL("model/load", BASE_PATH)
  url.searchParams.append("model_path", model_path)
  const response = await fetch(url)
  await handleError(response)
}

export const removeModel = async (model_path: string) => {
  const url = new URL("model/remove", BASE_PATH)
  url.searchParams.append("model_path", model_path)
  const response = await fetch(url)
  await handleError(response)
}

export const predict = async (
  state: EnvState,
  model_path: string,
  sample_count: number,
  sim_count: number,
  cb: (progress: PredictProgress) => void,
  signal: AbortSignal
) => {
  const search_params = new URLSearchParams()
  search_params.append("model_path", model_path)
  search_params.append("sample_count", sample_count.toString())
  search_params.append("sim_count", sim_count.toString())

  const ws = await WebSocket.connect(`ws://localhost:${PORT}/predict?${search_params.toString()}`)

  await ws.send(JSON.stringify(state))

  await new Promise<void>((resolve, reject) => {
    signal.addEventListener("abort", () => {
      ws.disconnect().then(reject)
    })

    ws.addListener((msg) => {
      if (typeof msg.data === "string") {
        const data = JSON.parse(msg.data!.toString()) as PredictProgress
        cb(data)
        if (data.finished_step == data.total_step) {
          ws.disconnect().then(resolve)
        }
      }
    })
  })
}