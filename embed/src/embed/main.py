from embed.agent import Agent
from pathlib import Path
from fjsp_env import Graph, GenerateParam
import asyncio

model = Agent(Path("../fjsp-model/onnx_models"))

print(model.state_model.get_providers())

g = Graph.rand_generate(GenerateParam(8, 5, 4, 2, 5, 7, 0.8, 4, 7, False))
g = g.init()

async def test():
    async for p in model.predict(g, 4, 32):
        print(p)

asyncio.run(test())
