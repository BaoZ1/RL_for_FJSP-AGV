from .FJSP_env import GenerateParam, Graph, Action, GraphFeature, IdIdxMapper
from dataclasses import dataclass
from typing import Self


@dataclass
class Observation:
    feature: GraphFeature
    mapper: IdIdxMapper
    action_list: list[Action]
    
    @classmethod
    def from_env(cls, env: Graph):
        feature, mapper = env.features()
        return Observation(
            feature,
            mapper,
            env.get_available_actions(),
        )


class Environment:
    def __init__(self, count: int, param: GenerateParam, auto_refresh: bool):
        self.count = count
        self.generate_param = param
        self.auto_refresh = auto_refresh

        self.envs: list[Graph] = []
        self.prev_lbs: list[float] = []

        self.reset()

    def reset(self):
        self.envs.clear()
        self.prev_lbs.clear()
        for _ in range(self.count):
            new_env = Graph.rand_generate(self.generate_param)
            new_env.init()
            self.envs.append(new_env)
            self.prev_lbs.append(new_env.finish_time_lower_bound())

    def observe(self) -> list[Observation]:
        if self.auto_refresh:
            for i, env in enumerate(self.envs):
                if env.finished():
                    new_env = Graph.rand_generate(self.generate_param)
                    new_env.init()
                    self.envs[i] = new_env

        return [Observation.from_env(env) for env in self.envs]

    def step(
        self, actions: list[Action], batch: bool = False
    ) -> tuple[list[float], list[bool], list[Observation]]:
        d_lb = []
        done = []
        for i, (env, prev_lb, action) in enumerate(
            zip(self.envs, self.prev_lbs, actions)
        ):
            new_env = env.act(action)
            new_lb = new_env.finish_time_lower_bound()
            d_lb.append(new_lb - prev_lb)
            done.append(new_env.finished())
            self.envs[i] = new_env
            self.prev_lbs[i] = new_lb

        return d_lb, done, [Observation.from_env(env) for env in self.envs]
