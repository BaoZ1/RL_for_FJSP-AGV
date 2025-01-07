import { AddOperationParams, EnvState, GenerationParams, PredictProgress } from "./types"
import { fetch } from "@tauri-apps/plugin-http"
import WebSocket from '@tauri-apps/plugin-websocket';

const PORT = 8000

const BASE_PATH = `http://localhost:${PORT}`

export const newEnv = async () => {
  const url = new URL("env", BASE_PATH)
  const response = await fetch(url, { method: "GET" })
  if (!response.ok) {
    console.log(await response.json())
  }
  return await response.json() as EnvState
}

export const loadEnv = async (path: string) => {
  const url = new URL("env/local", BASE_PATH)
  url.searchParams.append("path", path)
  const response = await fetch(url, { method: "GET" })
  if (!response.ok) {
    console.log(await response.json())
  }
  return await response.json() as EnvState
}

export const randEnv = async (params: GenerationParams) => {
  const url = new URL("env/rand", BASE_PATH)
  Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v.toString()))
  const response = await fetch(url, { method: "GET" })
  if (!response.ok) {
    console.log(await response.json())
  }
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
  if (!response.ok) {
    console.log(await response.json())
  }
  return await response.json() as EnvState
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
  if (!response.ok) {
    console.log(await response.json())
  }
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
  const ret = await response.json()
  console.log(ret);
  return ret as EnvState
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
  if (!response.ok) {
    console.log(await response.json())
  }
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
  if (!response.ok) {
    console.log(await response.json())
  }
  return await response.json() as EnvState
}

export const modelList = async () => {
  const url = new URL("model/list", BASE_PATH)
  const response = await fetch(url)
  if (!response.ok) {
    console.log(await response.json())
  }
  return await response.json() as string[]
}

export const loadModel = async (model_path: string) => {
  const url = new URL("model/load", BASE_PATH)
  url.searchParams.append("model_path", model_path)
  const response = await fetch(url)
  if (!response.ok) {
    console.log(await response.json())
  }
}

export const removeModel = async (model_path: string) => {
  const url = new URL("model/remove", BASE_PATH)
  url.searchParams.append("model_path", model_path)
  const response = await fetch(url)
  if (!response.ok) {
    console.log(await response.json())
  }
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

  const ws = await WebSocket.connect(`ws://localhost:${PORT}/test/predict?${search_params.toString()}`)

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