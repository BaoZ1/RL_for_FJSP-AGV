from fastapi import FastAPI, Query, Body, Depends
from typing import Annotated
import uvicorn
from pathlib import Path
from FJSP_env.FJSP_env import Graph
from models import *
import json
from FJSP_env import *


app = FastAPI()


def use_graph(state: Annotated[EnvState, Body()]) -> Graph:
    return Graph.from_state(state.model_dump())


@app.get("/env")
def get_new_env() -> EnvState:
    return Graph().get_state()


@app.get("env/local")
def get_local_save(path: Path) -> EnvState:
    return json.load(path.open())


@app.post("env/local")
def get_local_save(path: Path, state: Annotated[EnvState, Body()]) -> None:
    json.dump(state.model_dump(), path.open("w"))


@app.get("env/rand")
def get_rand_env(params: Annotated[GenerationParams, Query()]) -> EnvState:
    return Graph.rand_generate(**params.model_dump()).get_state()


@app.put("/operation/add")
def add_operation(
    graph: Annotated[Graph, Depends(use_graph)],
    type: int,
    time: int,
    pred: int | None,
    succ: int | None,
) -> EnvState:
    graph.insert_operation(type, time, pred, succ)
    return graph.get_state()


@app.put("/operation/remove")
def remove_operation(graph: Annotated[Graph, Depends(use_graph)], id: int) -> EnvState:
    graph.remove_operation(id)
    return graph.get_state()


@app.put("/path/add")
def add_path(graph: Annotated[Graph, Depends(use_graph)], a: int, b: int) -> EnvState:
    graph.add_path(a, b)
    return graph.get_state()


@app.put("/path/remove")
def remove_path(graph: Annotated[Graph, Depends(use_graph)], a: int, b: int) -> EnvState:
    graph.remove_path(a, b)
    return graph.get_state()


if __name__ == "__main__":
    uvicorn.run(app, reload=False)
