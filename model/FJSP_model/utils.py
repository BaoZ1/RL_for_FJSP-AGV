import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch_geometric.data import HeteroData, Batch, Data
from FJSP_env import GraphFeature, Action, Observation
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
class SequenceReplayItem:
    states: list[Observation]
    action_idxs: list[int]
    rewards: list[float]
    dones: list[bool]
    next_states: list[Observation]


class ReplayBuffer[T]:

    def __init__(self, seq_len: int, max_len: int = 1000):
        self.buffer: list[SequenceReplayItem] = []
        self.max_len = max_len

        self.seq_len = seq_len
        self.temp_buffer: list[
            list[tuple[Observation, int, float, bool, Observation]]
        ] = []

    def append(
        self,
        states: list[Observation],
        action_idxs: list[int],
        rewards: list[float],
        dones: list[bool],
        next_states: list[Observation],
    ):
        if len(self.temp_buffer) != 0:
            assert len(states) == len(self.temp_buffer)
        else:
            for _ in range(len(states)):
                self.temp_buffer.append([])
        for temp_buffer, data in zip(
            self.temp_buffer,
            zip(
                states,
                action_idxs,
                rewards,
                dones,
                next_states,
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
