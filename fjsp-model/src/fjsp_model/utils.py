import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import HeteroData, Batch, Data
from fjsp_env import GenerateParam, Graph, GraphFeature, OperationStatus, Action, ActionType, IdIdxMapper
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Callable
from itertools import count
import asyncio


def contiguous_transpose(x: torch.Tensor):
    return x.T.contiguous()


class Metadata:
    node_types = ["operation", "machine", "AGV"]
    edge_types = [
        # operation relation
        ("operation", "predecessor", "operation"),
        ("operation", "successor", "operation"),
        ("machine", "distance", "machine"),
        # hetero relation
        ("machine", "processable", "operation"),
        ("machine", "processing", "operation"),
        ("machine", "waiting", "operation"),
        ("AGV", "position", "machine"),
        ("AGV", "target", "machine"),
        ("AGV", "load_from", "operation"),
        ("AGV", "load_to", "operation"),
        # reverse hetero relation
        ("operation", "processable_rev", "machine"),
        ("operation", "processing_rev", "machine"),
        ("operation", "waiting_rev", "machine"),
        ("machine", "position_rev", "AGV"),
        ("machine", "target_rev", "AGV"),
        ("operation", "load_from_rev", "AGV"),
        ("operation", "load_to_rev", "AGV"),
    ]
    edge_attrs = {
        ("machine", "distance", "machine"): 1,
        # hetero
        ("machine", "processing", "operation"): 1,
        ("machine", "waiting", "operation"): 2,
        ("AGV", "target", "machine"): 1,
        # reverse
        ("operation", "processing_rev", "machine"): 1,
        ("operation", "waiting_rev", "machine"): 2,
        ("machine", "target_rev", "AGV"): 1,
    }


class HeteroGraph(HeteroData):
    def __cat_dim__(self, key, *args, **kwargs):
        if "index" in key:
            return -1

        if "attr" in key:
            if key == "global_attr":
                return None
            return -2

        return 0


def build_graph(feature: GraphFeature):
    graph = HeteroGraph()
    graph.global_attr = torch.tensor(feature.global_feature)
    graph["operation"].x = torch.tensor(feature.operation_features)
    graph["machine"].x = torch.tensor(feature.machine_features)
    graph["AGV"].x = torch.tensor(feature.AGV_features)

    graph["operation", "predecessor", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.predecessor_idx, dtype=torch.int64).view(-1, 2)
    )

    graph["operation", "successor", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.successor_idx, dtype=torch.int64).view(-1, 2)
    )

    graph["machine", "processable", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.processable_idx, dtype=torch.int64).view(-1, 2)
    )

    graph["machine", "processing", "operation"].edge_index = contiguous_transpose(
        torch.tensor(
            [[x[0], x[1]] for x in feature.processing], dtype=torch.int64
        ).view(-1, 2)
    )
    graph["machine", "processing", "operation"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.processing], dtype=torch.float
    ).view(-1, 1)

    graph["machine", "waiting", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.waiting], dtype=torch.int64).view(
            -1, 2
        )
    )
    graph["machine", "waiting", "operation"].edge_attr = torch.tensor(
        [[x[2], x[3]] for x in feature.waiting], dtype=torch.float
    ).view(-1, 2)

    graph["machine", "distance", "machine"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.distance], dtype=torch.int64).view(
            -1, 2
        )
    )

    graph["machine", "distance", "machine"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.distance], dtype=torch.float
    ).view(-1, 1)

    graph["AGV", "position", "machine"].edge_index = contiguous_transpose(
        torch.tensor(feature.AGV_position, dtype=torch.int64).view(-1, 2)
    )

    graph["AGV", "target", "machine"].edge_index = contiguous_transpose(
        torch.tensor(
            [[x[0], x[1]] for x in feature.AGV_target], dtype=torch.int64
        ).view(-1, 2)
    )
    graph["AGV", "target", "machine"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.AGV_target], dtype=torch.float
    ).view(-1, 1)

    graph["AGV", "load_from", "operation"].edge_index = contiguous_transpose(
        torch.tensor(
            [[x[0], x[1]] for x in feature.AGV_loaded], dtype=torch.int64
        ).view(-1, 2)
    )
    graph["AGV", "load_to", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[2]] for x in feature.AGV_loaded], dtype=torch.int).view(
            -1, 2
        )
    )

    for edge in graph.edge_types:
        src, name, dst = edge
        if name in ["predecessor", "successor", "distance"]:
            continue
        graph[dst, f"{name}_rev", src].edge_index = (
            graph[edge].edge_index.flip(0).contiguous()
        )
        if edge in Metadata.edge_attrs:
            graph[dst, f"{name}_rev", src].edge_attr = graph[edge].edge_attr

    return graph


def get_offsets(batch: Batch) -> dict[str, dict[int, int]]:
    ret: dict[str, dict[int, int]] = {}
    for key in ["operation", "machine", "AGV"]:
        ret[key] = {}
        unique_values = torch.unique(batch[key].batch)
        indices = {
            value.item(): torch.where(batch[key].batch == value)[0][0].item()
            for value in unique_values
        }
        for idx, indice in indices.items():
            ret[key][idx] = indice
    return ret


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
                reward += 3
            elif action.action_type == ActionType.move:
                reward += -0.05
            new_env = env.act(action)
            if action.action_type == ActionType.wait:
                for oid in env.get_operations_id():
                    op = env.get_operation(oid)
                    new_op = new_env.get_operation(oid)
                    if new_op.status != op.status:
                        reward += 0.5
                        if new_op.status == OperationStatus.processing:
                            reward += 4
                        # if new_op.status == OperationStatus.finished:
                        #     ma = new_env.get_machine(new_op.processing_machine)
                        #     if ma.status == MachineStatus.working:
                        #         reward += 2
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


@dataclass
class SequenceReplayItem:
    graphs: list[Graph]
    action_idxs: list[int]
    rewards: list[float]
    dones: list[bool]
    next_graphs: list[Graph]


class ReplayBuffer:

    def __init__(self, seq_len: int, max_len: int = 1000):
        self.buffer: list[SequenceReplayItem] = []
        self.max_len = max_len

        self.seq_len = seq_len
        self.temp_buffer: list[
            list[tuple[Graph, int, float, bool, Graph]]
        ] = []

    def append(
        self,
        graphs: list[Graph],
        action_idxs: list[int],
        rewards: list[float],
        dones: list[bool],
        next_graphs: list[Graph],
    ):
        if len(self.temp_buffer) != 0:
            assert len(graphs) == len(self.temp_buffer)
        else:
            for _ in range(len(graphs)):
                self.temp_buffer.append([])
        for temp_buffer, data in zip(
            self.temp_buffer,
            zip(
                graphs,
                action_idxs,
                rewards,
                dones,
                next_graphs,
            ),
        ):
            temp_buffer.append(data)
            if len(temp_buffer) == self.seq_len:
                self.buffer.append(
                    SequenceReplayItem(
                        *[list(item) for item in zip(*temp_buffer, strict=True)]
                    )
                )
                temp_buffer.pop(0)
                if len(self.buffer) > self.max_len:
                    self.buffer.pop(0)
            assert type(data[3]) == bool
            if data[3]:
                temp_buffer.clear()

    def sample(self, num: int) -> list[SequenceReplayItem]:
        assert len(self.buffer) >= num

        return np.random.choice(self.buffer, num, False)


class ReplayDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int):
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __iter__(self):
        for item in self.buffer.sample(self.sample_size):
            yield item


class NormClipper:
    def __init__(self, module: torch.nn.Module, mean_rate: float = 0.8, abnormal_rate: float = 1.2):
        self.module = module
        self.max_norm = None
        self.mean_rate = mean_rate
        self.abnormal_rate = abnormal_rate
    
    def clip(self):
        raw_norm = clip_grad_norm_(self.module.parameters(), 1e8)
        if self.max_norm is None:
            self.max_norm = raw_norm      
        else:
            self.max_norm *= self.mean_rate + min(raw_norm / self.max_norm, self.abnormal_rate) * (1 - self.mean_rate)
        return clip_grad_norm_(self.module.parameters(), self.max_norm)
