import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from .FJSP_env import GenerateParam, Graph, Action, GraphFeature, IdIdxMapper


def contiguous_transpose(x: torch.Tensor):
    if x.numel() == 0:
        return x
    return x.T.contiguous()


class Metadata:
    node_types = ["operation", "machine", "AGV"]
    edge_types = [
        # operation relation
        ("operation", "predecessor", "operation"),
        ("operation", "successor", "operation"),
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
        ("machine", "processing", "operation"): 1,
        ("machine", "waiting", "operation"): 2,
        ("AGV", "target", "machine"): 1,
        # reverse
        ("operation", "processing_rev", "machine"): 1,
        ("operation", "waiting_rev", "machine"): 2,
        ("machine", "target_rev", "AGV"): 1,
    }


def build_graph(feature: GraphFeature):
    graph = HeteroData()
    graph["operation"].x = torch.tensor(feature.operation_features)
    graph["machine"].x = torch.tensor(feature.machine_features)
    graph["AGV"].x = torch.tensor(feature.AGV_features)

    if len(feature.predecessor_idx):
        graph["operation", "predecessor", "operation"].edge_index = (
            contiguous_transpose(torch.tensor(feature.predecessor_idx, dtype=torch.int))
        )
    if len(feature.successor_idx):
        graph["operation", "successor", "operation"].edge_index = contiguous_transpose(
            torch.tensor(feature.successor_idx, dtype=torch.int)
        )
    if len(feature.processable_idx):
        graph["machine", "processable", "operation"].edge_index = contiguous_transpose(
            torch.tensor(feature.processable_idx, dtype=torch.int)
        )
    if len(feature.processing):
        graph["machine", "processing", "operation"].edge_index = contiguous_transpose(
            torch.tensor([[x[0], x[1]] for x in feature.processing], dtype=torch.int)
        )
        graph["machine", "processing", "operation"].edge_attr = contiguous_transpose(
            torch.tensor([[x[2]] for x in feature.processing])
        )
    if len(feature.waiting):
        graph["machine", "waiting", "operation"].edge_index = contiguous_transpose(
            torch.tensor([[x[0], x[1]] for x in feature.waiting], dtype=torch.int)
        )
        graph["machine", "waiting", "operation"].edge_attr = contiguous_transpose(
            torch.tensor([[x[2], x[3]] for x in feature.waiting])
        )
    if len(feature.AGV_position):
        graph["AGV", "position", "machine"].edge_index = contiguous_transpose(
            torch.tensor(feature.AGV_position, dtype=torch.int)
        )
    if len(feature.AGV_target):
        graph["AGV", "target", "machine"].edge_index = contiguous_transpose(
            torch.tensor([[x[0], x[1]] for x in feature.AGV_target], dtype=torch.int)
        )
        graph["AGV", "target", "machine"].edge_attr = contiguous_transpose(
            torch.tensor([[x[2]] for x in feature.AGV_target])
        )
    if len(feature.AGV_loaded):
        graph["AGV", "load_from", "operation"].edge_index = contiguous_transpose(
            torch.tensor([[x[0], x[1]] for x in feature.AGV_loaded], dtype=torch.int)
        )
        graph["AGV", "load_to", "operation"].edge_index = contiguous_transpose(
            torch.tensor([[x[0], x[2]] for x in feature.AGV_loaded], dtype=torch.int)
        )

    for edge in graph.edge_types:
        src, name, dst = edge
        if name in ["predecessor", "successor"]:
            continue
        graph[dst, f"{name}_rev", src].edge_index = (
            graph[edge].edge_index.flip(0).contiguous()
        )
        if edge in Metadata.edge_attrs:
            graph[dst, f"{name}_rev", src].edge_attr = graph[edge].edge_attr

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

    def observe(self) -> tuple[list[GraphFeature], list[dict[str, int]], list[IdIdxMapper], list[list[Action]]]:
        features = []
        offsets = []
        offsets_record = [0, 0, 0]
        mappers = []
        actions = []
        if self.auto_refresh:
            for i, env in enumerate(self.envs):
                if env.finished():
                    new_env = Graph.rand_generate(self.generate_param)
                    new_env.init()
                    self.envs[i] = new_env

        for env in self.envs:
            feature, mapper = env.features()
            features.append(feature)
            offsets.append(
                {
                    "operation": offsets_record[0],
                    "machine": offsets_record[1],
                    "AGV": offsets_record[2],
                }
            )
            offsets_record[0] += len(feature.operation_features)
            offsets_record[1] += len(feature.machine_features)
            offsets_record[2] += len(feature.AGV_features)
            mappers.append(mapper)
            actions.append(env.get_available_actions())

        return (features, offsets, mappers, actions)

    def step(
        self, actions: list[Action], batch: bool = False
    ) -> tuple[list[float], list[bool], list[GraphFeature]]:
        d_lb = []
        done = []
        next_obs = []
        if batch:
            new_envs, new_lbs = Graph.batch_step(self.envs, actions)
        for i, (env, prev_lb, action) in enumerate(
            zip(self.envs, self.prev_lbs, actions)
        ):
            if batch:
                new_env = new_envs[i]
                new_lb = new_lbs[i]
            else:
                new_env = env.act(action)
                new_lb = new_env.finish_time_lower_bound()
            d_lb.append(new_lb - prev_lb)
            done.append(new_env.finished())
            next_obs.append(new_env.features())
            self.envs[i] = new_env
            self.prev_lbs[i] = new_lb

        return d_lb, done, next_obs
