from .FJSP_env import (
    GenerateParam,
    Graph,
    Action,
    ActionType,
    GraphFeature,
    IdIdxMapper,
)
from dataclasses import dataclass
from typing import Callable
from itertools import count
import numpy as np


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


def single_step_simple_predict(obs: Observation, rand_prob: float = 0):
    useful_action_idxs = (
        [
            i
            for i, action in enumerate(obs.action_list)
            if action.action_type in (ActionType.pick, ActionType.transport)
        ]
        if np.random.rand() < rand_prob
        else []
    )
    if len(useful_action_idxs) == 0:
        i = np.random.choice(len(obs.action_list))
        act_idx = i
    else:
        i = np.random.choice(len(useful_action_idxs))
        act_idx = useful_action_idxs[i]
    act = obs.action_list[act_idx]
    return act, act_idx


class Environment:
    def __init__(
        self,
        count: int,
        params: list[GenerateParam],
        auto_refresh: bool,
    ):
        self.count = count
        self.generate_params = params
        self.auto_refresh = auto_refresh

        self.envs: list[Graph] = []
        self.prev_lbs: list[float] = []

        self.reset()

    @staticmethod
    def from_graphs(graphs: list[Graph]):
        env = Environment(0, [], False)
        env.reset(graphs)
        return env

    def reset(self, envs: list[Graph] | None = None):
        self.envs.clear()
        self.prev_lbs.clear()
        if envs is not None:
            self.count = len(envs)
        for i in range(self.count):
            if envs:
                new_env = envs[i].reset().init()
            else:
                new_env = self.generate_new().init()
            self.envs.append(new_env)
            self.prev_lbs.append(new_env.finish_time_lower_bound())

    def generate_new(self) -> Graph:
        for try_num in count():
            new_env = Graph.rand_generate(np.random.choice(self.generate_params))
            if self.test(new_env):
                return new_env
            if try_num > 20:
                raise Exception("bad parameters")

    def test(self, graph: Graph):
        env = graph.init()
        while True:
            ob = Observation.from_env(env)
            action, _ = single_step_simple_predict(ob)
            env = env.act(action)
            if env.finished():
                return True

    def observe(self) -> list[Observation]:
        if self.auto_refresh:
            for i, env in enumerate(self.envs):
                if env.finished():
                    new_env = self.generate_new().init()
                    self.envs[i] = new_env
                    self.prev_lbs[i] = new_env.finish_time_lower_bound()

        return [
            Observation.from_env(env)
            for env in self.envs
            if not (env.finished())
        ]

    def step(
        self, actions: list[Action]
    ) -> tuple[list[float], list[bool], list[Observation]]:
        rewards = []
        dones = []
        action_idx = 0
        ret_envs = []
        for i, (env, prev_lb) in enumerate(zip(self.envs, self.prev_lbs)):
            if env.finished():
                continue
            reward = 0
            action = actions[action_idx]
            if action.action_type in (ActionType.pick, ActionType.transport):
                reward += 2
            elif action.action_type == ActionType.move:
                reward += -0.5
            new_env = env.act(action)
            if (
                len(new_actions := new_env.get_available_actions()) == 1
                and new_actions[0].action_type == ActionType.wait
            ):
                new_env = new_env.act(new_actions[0])
            action_idx += 1
            new_lb = new_env.finish_time_lower_bound()
            d_lb = new_lb - prev_lb
            # reward += 4 / (d_lb + 2) - 2
            reward += -d_lb
            if new_env.finished():
                done = True
                reward = 100
            else:
                done = False
            rewards.append(reward)
            dones.append(done)
            self.envs[i] = new_env
            self.prev_lbs[i] = new_lb
            ret_envs.append(new_env)

        return rewards, dones, [Observation.from_env(env) for env in ret_envs]
