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
import asyncio


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


def single_step_useful_first_predict(obs: Observation, rand_prob: float = 0):
    useful_action_idxs = (
        [
            i
            for i, action in enumerate(obs.action_list)
            if action.action_type in (ActionType.pick, ActionType.transport)
        ]
        if np.random.rand() > rand_prob
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


def single_step_useful_only_predict(obs: Observation, rand_prob: float = 0):
    useful_action_idxs = (
        [
            i
            for i, action in enumerate(obs.action_list)
            if action.action_type in (ActionType.pick, ActionType.transport)
        ]
        if np.random.rand() > rand_prob
        else [*range(len(obs.action_list))]
    )

    if len(useful_action_idxs) == 0:
        i = obs.action_list.index(Action(ActionType.wait))
        act_idx = i
    else:
        i = np.random.choice(len(useful_action_idxs))
        act_idx = useful_action_idxs[i]
    act = obs.action_list[act_idx]
    return act, act_idx


async def simple_predict(
    graph: Graph, rule: Callable[[Observation], tuple[Action, int]]
):
    env = Environment.from_graphs([graph])

    for round_count in count(1):
        obs = env.observe()[0]

        action, _ = rule(obs)
        env.step([action], False)

        finished_step, total_step = env.envs[0].progress()

        yield {
            "round_count": round_count,
            "finished_step": finished_step,
            "total_step": total_step,
            "graph_state": env.envs[0],
            "action": action,
        }

        if env.envs[0].finished():
            break

        await asyncio.sleep(0)


class Environment:
    def __init__(
        self,
        count: int,
        params: list[GenerateParam],
        auto_refresh: bool,
    ):
        self.count = count
        self.generate_params = params
        self.params_total_task_count = np.ones(len(params))
        self.auto_refresh = auto_refresh

        self.envs: list[Graph] = []
        self.prev_lbs: list[float] = []

        self.reset()

    @staticmethod
    def from_graphs(graphs: list[Graph], reset: bool = False):
        env = Environment(0, [], False)
        env.set_graphs(graphs)
        if reset:
            env.reset(True)
        return env

    def set_graphs(self, graphs: list[Graph]):
        self.envs.clear()
        self.prev_lbs.clear()
        self.count = len(graphs)
        for i in range(self.count):
            new_env = graphs[i].copy()
            self.envs.append(new_env)
            self.prev_lbs.append(new_env.finish_time_lower_bound())

    def reset(self, keep_graphs: bool = False):
        if not keep_graphs:
            self.envs.clear()
        self.prev_lbs.clear()
        for i in range(self.count):
            if keep_graphs:
                self.envs[i] = self.envs[i].reset().init()
            else:
                self.envs.append(self.generate_new().init())
            self.prev_lbs.append(self.envs[i].finish_time_lower_bound())

    def generate_new(self) -> Graph:
        for try_num in count():
            param_idx = np.random.choice(
                len(self.generate_params),
                p=(1 / self.params_total_task_count)
                / np.sum(1 / self.params_total_task_count),
            )
            new_env = Graph.rand_generate(self.generate_params[param_idx])
            if self.test(new_env):
                self.params_total_task_count[param_idx] += new_env.progress()[1]
                self.params_total_task_count -= np.min(self.params_total_task_count) - 1
                return new_env
            if try_num > 20:
                raise Exception("bad parameters")

    def test(self, graph: Graph):
        env = graph.init()
        for n in count(1):
            ob = Observation.from_env(env)
            action, _ = single_step_useful_first_predict(ob)
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

        return [Observation.from_env(env) for env in self.envs if not (env.finished())]

    def step(
        self, actions: list[Action], auto_wait: bool = False
    ) -> tuple[list[float], list[bool], list[Observation]]:
        rewards = []
        dones = []
        action_idx = 0
        ret_envs = []
        assert sum([not g.finished() for g in self.envs]) == len(actions)
        for i, (env, prev_lb) in enumerate(zip(self.envs, self.prev_lbs)):
            if env.finished():
                continue
            reward = 0
            action = actions[action_idx]
            if action.action_type in (ActionType.pick, ActionType.transport):
                reward += 5
            # elif action.action_type == ActionType.move:
            #     reward += -0.05
            try:
                new_env = env.act(action)
            except Exception as e:
                print(e)
                print(env)
                print(action)
            if (
                auto_wait
                and len(new_actions := new_env.get_available_actions()) == 1
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
                # reward = 1
            else:
                done = False
            rewards.append(reward)
            dones.append(done)
            self.envs[i] = new_env
            self.prev_lbs[i] = new_lb
            ret_envs.append(new_env)

        return rewards, dones, [Observation.from_env(env) for env in ret_envs]
