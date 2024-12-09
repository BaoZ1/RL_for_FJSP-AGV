import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from .FJSP_env import GenerateParam, Graph, Action, GraphFeature


def build_graph(feature: GraphFeature):
    graph = HeteroData()
    graph["operation"].x = torch.tensor(feature.operation_features)
    graph["machine"].x = torch.tensor(feature.machine_features)
    graph["AGV"].x = torch.tensor(feature.AGV_features)

    graph["operation", "predecessor", "operation"].edge_index = torch.tensor(
        feature.predecessor_idx
    ).T.contiguous()
    graph["operation", "successor", "operation"].edge_index = torch.tensor(
        feature.successor_idx
    ).T.contiguous()
    graph["machine", "processable", "operation"].edge_idx = torch.tensor(
        feature.processable_idx
    ).T.contiguous()
    graph["machine", "processing", "operation"].edge_idx = torch.tensor(
        [[x[0], x[1]] for x in feature.processing]
    ).T.contiguous()
    graph["machine", "processing", "operation"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.processing]
    ).T.contiguous()
    graph["machine", "waiting", "operation"].edge_idx = torch.tensor(
        [[x[0], x[1]] for x in feature.waiting]
    ).T.contiguous()
    graph["machine", "waiting", "operation"].edge_attr = torch.tensor(
        [[x[2], x[3]] for x in feature.waiting]
    ).T.contiguous()
    graph["AGV", "position", "machine"].edge_idx = torch.tensor(
        feature.AGV_position
    ).T.contiguous()
    graph["AGV", "target", "machine"].edge_idx = torch.tensor(
        [[x[0], x[1]] for x in feature.AGV_target]
    ).T.contiguous()
    graph["AGV", "target", "machine"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.AGV_target]
    ).T.contiguous()
    graph["AGV", "load_from", "operation"].edge_idx = torch.tensor(
        [[x[0], x[1]] for x in feature.AGV_loaded]
    ).T.contiguous()
    graph["AGV", "load_to", "operation"].edge_idx = torch.tensor(
        [[x[0], x[2]] for x in feature.AGV_loaded]
    ).T.contiguous()

    return graph


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
    
    def observe(self):
        if self.auto_refresh:
            for i, env in enumerate(self.envs):
                if env.finished():
                    new_env = Graph.rand_generate(self.generate_param)
                    new_env.init()
                    self.envs[i] = new_env
        return [env.features() for env in self.envs]

    def step(self, actions: list[Action]):
        d_lb = []
        finished = []
        next_obs = []
        for i, (env, prev_lb, action) in enumerate(
            zip(self.envs, self.prev_lbs, actions)
        ):
            new_env = env.act(action)
            new_lb = new_env.finish_time_lower_bound()
            d_lb.append( new_lb - prev_lb)
            finished.append(new_env.finished())
            next_obs.append(new_env.features())
            self.envs[i] = new_env
            self.prev_lbs[i] = new_lb
        
        return d_lb, finished, next_obs
