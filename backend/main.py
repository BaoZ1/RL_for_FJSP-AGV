from fastapi import FastAPI, WebSocket, Query, Body, Depends, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from typing import Annotated
import uvicorn
from pathlib import Path
from models import *
from FJSP_env import *


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit

torch.jit.script_method = script_method
torch.jit.script = script

from FJSP_model.modules import Agent


import torch.cuda

device = "cuda" if torch.cuda.is_available() else "cpu"

models: dict[str, Agent] = {}

app = FastAPI()


def use_graph(state: Annotated[EnvState, Body()]) -> Graph:
    return Graph.from_state(state.model_dump(by_alias=True))


@app.get("/env")
async def get_new_env() -> EnvState:
    return Graph()


@app.get("/env/load")
async def get_local_env(
    path: Annotated[Path, Query()],
) -> EnvState:
    return EnvState.model_validate_json(path.read_text())


@app.post("/env/save")
async def save_local_env(
    path: Annotated[Path, Query()], graph: Annotated[Graph, Depends(use_graph)]
):
    path.write_text(json.dumps(graph.get_state()))


@app.post("/env/local")
async def set_local_save(
    path: Annotated[Path, Query()],
    state: Annotated[EnvState, Body()],
) -> None:
    path.write_text(state.model_dump_json())


@app.get("/env/rand")
async def get_rand_env(
    params: Annotated[GenerationParamModel, Query()],
) -> EnvState:
    return Graph.rand_generate(GenerateParam(**params.model_dump()))


@app.put("/env/init")
async def init_env(graph: Annotated[Graph, Depends(use_graph)]) -> EnvState:
    return graph.init()


@app.put("/env/reset")
async def reset_env(graph: Annotated[Graph, Depends(use_graph)]) -> EnvState:
    return graph.reset()


@app.post("/env/paths")
async def get_paths(
    graph: Annotated[Graph, Depends(use_graph)]
) -> dict[int, dict[int, tuple[list[int], float]]]:
    graph.calc_distance()
    return graph.get_paths()


@app.put("/operation/add")
async def add_operation(
    graph: Annotated[Graph, Depends(use_graph)],
    type: Annotated[int, Query()],
    time: Annotated[float, Query()],
    pred: Annotated[int | None, Query()] = None,
    succ: Annotated[int | None, Query()] = None,
) -> EnvState:
    graph.insert_operation(type, time, pred, succ)
    return graph


@app.put("/operation/remove")
async def remove_operation(
    graph: Annotated[Graph, Depends(use_graph)],
    id: Annotated[int, Query()],
) -> EnvState:
    graph.remove_operation(id)
    return graph


@app.put("/machine/add")
async def add_machine(
    graph: Annotated[Graph, Depends(use_graph)],
    type: Annotated[int, Query()],
    x: Annotated[float, Query()],
    y: Annotated[float, Query()],
    path_to: Annotated[list[int], Query()],
) -> EnvState:
    new_id = graph.add_machine(type, Position(x, y))
    for target_id in path_to:
        graph.add_path(new_id, target_id)
        graph.add_path(target_id, new_id)
    return graph


@app.put("/machine/remove")
async def remove_machine(
    graph: Annotated[Graph, Depends(use_graph)],
    id: Annotated[int, Query()],
) -> EnvState:
    if id == Graph.dummy_machine_id:
        return PlainTextResponse("不可删除设备0", 400)
    graph.remove_machine(id)
    return graph


@app.put("/path/add")
async def add_path(
    graph: Annotated[Graph, Depends(use_graph)],
    a: Annotated[int, Query()],
    b: Annotated[int, Query()],
) -> EnvState:
    graph.add_path(a, b)
    return graph


@app.put("/path/remove")
async def remove_path(
    graph: Annotated[Graph, Depends(use_graph)],
    a: Annotated[int, Query()],
    b: Annotated[int, Query()],
) -> EnvState:
    graph.remove_path(a, b)
    return graph


@app.put("/agv/add")
async def add_agv(
    graph: Annotated[Graph, Depends(use_graph)],
    speed: Annotated[int, Query()],
    init_pos: Annotated[int, Query()],
) -> EnvState:
    graph.add_AGV(speed, init_pos)
    return graph


@app.get("/model/list")
async def model_list():
    return list(models.keys()) + ["useful_first", "useful_only"]


@app.get("/model/load")
async def load_model(model_path: Annotated[str, Query()]):
    model = Agent.load_from_checkpoint(model_path, envs=None, finished_batch_count=0)
    model.to(device)
    model.eval()
    models[model_path] = model


@app.get("/model/remove")
async def remove_model(model_path: Annotated[str, Query()]):
    models.pop(model_path)


@app.websocket("/predict")
async def predict(
    websocket: WebSocket,
    model_path: Annotated[str, Query()],
    sample_count: Annotated[int, Query()],
    sim_count: Annotated[int, Query()],
):
    await websocket.accept()
    graph = use_graph(EnvState.model_validate(await websocket.receive_json()))
    try:
        match model_path:
            case "useful_first":
                predictor = simple_predict(
                    graph,
                    single_step_useful_first_predict,
                )
            case "useful_only":
                predictor = simple_predict(
                    graph,
                    single_step_useful_only_predict,
                )
            case _:
                predictor = models[model_path].predict(
                    graph,
                    sample_count,
                    sim_count,
                )

        async for data in predictor:
            await websocket.send_text(
                PredictProgress.model_validate(data).model_dump_json(by_alias=True)
            )

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, port=8000, reload=False)
