from fastapi import FastAPI, Query
import uvicorn
from pathlib import Path
from models import *
import json
from FJSP_env import *


app = FastAPI()

@app.get("/local_save")
def get_local_save(path: Path) -> EnvState:
    return json.load(path.open())

@app.get("/new_env")
def get_new_env() -> EnvState:
    return Graph().get_state()


@app.get("/rand_env")
def get_rand_env(params: GenerationParams = Query()) -> EnvState:
    return Graph.rand_generate(**params.model_dump()).get_state()


if __name__ == "__main__":
    uvicorn.run(app, reload=False)
