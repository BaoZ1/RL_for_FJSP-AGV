import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch_geometric.data import HeteroData, Batch, Data
from FJSP_env import GraphFeature, Action
from dataclasses import dataclass


def contiguous_transpose(x: torch.Tensor):
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


class HeteroGraph(HeteroData):
    def __cat_dim__(self, key, *args, **kwargs):
        if "index" in key:
            return -1

        if "attr" in key:
            return -2

        return 0


def build_graph(feature: GraphFeature):
    graph = HeteroGraph()
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


def build_graph(feature: GraphFeature):
    graph = HeteroGraph()
    graph["operation"].x = torch.tensor(feature.operation_features)
    graph["machine"].x = torch.tensor(feature.machine_features)
    graph["AGV"].x = torch.tensor(feature.AGV_features)

    graph["operation", "predecessor", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.predecessor_idx, dtype=torch.int).view(-1, 2)
    )

    graph["operation", "successor", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.successor_idx, dtype=torch.int).view(-1, 2)
    )

    graph["machine", "processable", "operation"].edge_index = contiguous_transpose(
        torch.tensor(feature.processable_idx, dtype=torch.int).view(-1, 2)
    )

    graph["machine", "processing", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.processing], dtype=torch.int).view(
            -1, 2
        )
    )
    graph["machine", "processing", "operation"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.processing], dtype=torch.float
    ).view(-1, 1)

    graph["machine", "waiting", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.waiting], dtype=torch.int).view(
            -1, 2
        )
    )
    graph["machine", "waiting", "operation"].edge_attr = torch.tensor(
        [[x[2], x[3]] for x in feature.waiting], dtype=torch.float
    ).view(-1, 2)

    graph["AGV", "position", "machine"].edge_index = contiguous_transpose(
        torch.tensor(feature.AGV_position, dtype=torch.int).view(-1, 2)
    )

    graph["AGV", "target", "machine"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.AGV_target], dtype=torch.int).view(
            -1, 2
        )
    )
    graph["AGV", "target", "machine"].edge_attr = torch.tensor(
        [[x[2]] for x in feature.AGV_target], dtype=torch.float
    ).view(-1, 1)

    graph["AGV", "load_from", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[1]] for x in feature.AGV_loaded], dtype=torch.int).view(
            -1, 2
        )
    )
    graph["AGV", "load_to", "operation"].edge_index = contiguous_transpose(
        torch.tensor([[x[0], x[2]] for x in feature.AGV_loaded], dtype=torch.int).view(
            -1, 2
        )
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
class ReplayItem[T]:
    state: T
    action: Action
    reward: float
    done: bool
    next_state: T


class ReplayBuffer[T]:

    def __init__(self, max_len: int = 1000):
        self.buffer: list[ReplayItem[T]] = []
        self.max_len = max_len

    def append(
        self,
        state: T,
        action: Action,
        reward: float,
        done: bool,
        next_state: T,
    ):
        self.buffer.append(
            ReplayItem(
                state,
                action,
                reward,
                done,
                next_state,
            )
        )
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def sample(self, num: int) -> list[ReplayItem[T]]:
        assert len(self.buffer) > num

        return np.random.choice(self.buffer, num, False)


class ReplayDataset[T](IterableDataset):
    def __init__(self, buffer: ReplayBuffer[T], sample_size: int):
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __iter__(self):
        for item in self.buffer.sample(self.sample_size):
            yield item
