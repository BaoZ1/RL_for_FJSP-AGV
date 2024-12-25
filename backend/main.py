from fastapi import FastAPI, WebSocket, Query, Body, Depends, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Annotated
import uvicorn
from pathlib import Path


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit

torch.jit.script_method = script_method
torch.jit.script = script

import torch
from FJSP_env.FJSP_env import Graph
from models import *
from FJSP_env import *
from FJSP_model.modules import Agent


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

models: dict[str, Agent] = {}


def use_graph(state: Annotated[EnvState, Body()]) -> Graph:
    return Graph.from_state(state.model_dump())


@app.get("/env")
async def get_new_env() -> EnvState:
    return Graph()


@app.get("/env/local")
async def get_local_save(
    path: Annotated[Path, Query()],
) -> EnvState:
    return EnvState.model_validate_json(path.read_text())


@app.post("/env/local")
async def get_local_save(
    path: Annotated[Path, Query()],
    state: Annotated[EnvState, Body()],
) -> None:
    path.write_text(state.model_dump_json())


@app.get("/env/rand")
async def get_rand_env(
    params: Annotated[GenerationParamModel, Query()],
) -> EnvState:
    return Graph.rand_generate(GenerateParam(**params.model_dump())).get_state()


@app.put("/operation/add")
async def add_operation(
    graph: Annotated[Graph, Depends(use_graph)],
    type: Annotated[int, Query()],
    time: Annotated[float, Query()],
    pred: Annotated[int | None, Query()] = None,
    succ: Annotated[int | None, Query()] = None,
) -> EnvState:
    graph.insert_operation(type, time, pred, succ)
    return graph.get_state()


@app.put("/operation/remove")
async def remove_operation(
    graph: Annotated[Graph, Depends(use_graph)],
    id: Annotated[int, Query()],
) -> EnvState:
    graph.remove_operation(id)
    return graph.get_state()


@app.put("/path/add")
async def add_path(
    graph: Annotated[Graph, Depends(use_graph)],
    a: Annotated[int, Query()],
    b: Annotated[int, Query()],
) -> EnvState:
    graph.add_path(a, b)
    return graph.get_state()


@app.put("/path/remove")
async def remove_path(
    graph: Annotated[Graph, Depends(use_graph)],
    a: Annotated[int, Query()],
    b: Annotated[int, Query()],
) -> EnvState:
    graph.remove_path(a, b)
    return graph.get_state()


@app.get("/model/list")
async def model_list():
    return list(models.keys())


@app.get("/model/load")
async def load_model(model_path: Annotated[str, Query()]):
    model = Agent.load_from_checkpoint(model_path, envs=None)
    model.to(device)
    model.eval()
    models[model_path] = model


@app.get("/model/remove")
async def remove_model(model_path: Annotated[str, Query()]):
    models.pop(model_path)


@app.websocket("/test/predict")
async def predict(
    websocket: WebSocket,
    model_path: Annotated[str, Query()],
    sample_count: Annotated[int, Query()],
    sim_count: Annotated[int, Query()],
    predict_num: Annotated[int, Query()],
):
    await websocket.accept()
    try:
        async for finished, info in models[model_path].predict(
            use_graph(EnvState.model_validate(await websocket.receive_json())),
            sample_count,
            sim_count,
            predict_num,
        ):
            if not finished:
                await websocket.send_json(info)
            else:
                await websocket.send_text(PredictResult.model_validate(info).model_dump_json())

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, reload=False)
