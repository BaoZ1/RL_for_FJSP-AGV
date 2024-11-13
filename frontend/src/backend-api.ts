import { EnvState, GenerationParams } from "./types"
import { fetch } from "@tauri-apps/plugin-http"

const BASE_PATH = "http://localhost:8000"

export const loadEnv = async (path: string) => {
    const url = new URL("local_save", BASE_PATH)
    url.searchParams.append("path", path)
    const response = await fetch(url, { method: "GET" })
    return await response.json() as EnvState
}

export const newEnv = async () => {
    const url = new URL("new_env", BASE_PATH)
    const response = await fetch(url, { method: "GET" })
    return await response.json() as EnvState
}

export const randEnv = async (params: GenerationParams) => {
    const url = new URL("rand_env", BASE_PATH)
    Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v.toString()))
    const response = await fetch(url, { method: "GET" })
    return await response.json() as EnvState
}

export const addOperation = async (state: EnvState, pred?: number, succ?: number) => {
    const url = new URL("add_operation", BASE_PATH)
    const response = await fetch(url, {method: "UPDATE", body: JSON.stringify({state, pred, succ})})
    return await response.json() as EnvState
}

export const removeOperation = async (state: EnvState, target: number) => {
    const url = new URL("remove_operation", BASE_PATH)
    const response = await fetch(url, {method: "UPDATE", body: JSON.stringify({state, target})})
    return await response.json() as EnvState
}