import { AddOperationParams, EnvState, GenerationParams } from "./types"
import { fetch } from "@tauri-apps/plugin-http"

const BASE_PATH = "http://localhost:8000"

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