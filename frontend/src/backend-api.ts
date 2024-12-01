import { EnvState, GenerationParams } from "./types"
import { fetch } from "@tauri-apps/plugin-http"

const BASE_PATH = "http://localhost:8000"

export const newEnv = async () => {
  const url = new URL("env", BASE_PATH)
  const response = await fetch(url, { method: "GET" })
  return await response.json() as EnvState
}

export const loadEnv = async (path: string) => {
    const url = new URL("env/local", BASE_PATH)
    url.searchParams.append("path", path)
    const response = await fetch(url, { method: "GET" })
    return await response.json() as EnvState
}

export const randEnv = async (params: GenerationParams) => {
    const url = new URL("env/rand", BASE_PATH)
    Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v.toString()))
    const response = await fetch(url, { method: "GET" })
    return await response.json() as EnvState
}

export const addOperation = async (
  state: EnvState, machine_type: number, 
  process_time: number, pred?: number, succ?: number
) => {
    const url = new URL("operation/add", BASE_PATH)
    url.searchParams.append("type", machine_type.toString())
    url.searchParams.append("time", process_time.toString())
    pred && url.searchParams.append("pred", pred.toString())
    succ && url.searchParams.append("succ", succ.toString())
    const response = await fetch(url, {method: "PUT", body: JSON.stringify(state)})
    return await response.json() as EnvState
}

export const removeOperation = async (state: EnvState, target: number) => {
  const url = new URL("operation/remove", BASE_PATH)
  url.searchParams.append("id", target.toString())
  const response = await fetch(url, { method: "PUT", body: JSON.stringify(state)})
  return await response.json() as EnvState
}

export const addPath = async (state: EnvState, a: number, b: number) => {
  const url = new URL("path/add", BASE_PATH)
  url.searchParams.append("a", a.toString())
  url.searchParams.append("b", b.toString())
  const response = await fetch(url, { method: "PUT", body: JSON.stringify(state)})
  return await response.json() as EnvState
}

export const removePath = async (state: EnvState, a: number, b: number) => {
  const url = new URL("path/remove", BASE_PATH)
  url.searchParams.append("a", a.toString())
  url.searchParams.append("b", b.toString())
  const response = await fetch(url, { method: "PUT", body: JSON.stringify(state) })
  return await response.json() as EnvState
}