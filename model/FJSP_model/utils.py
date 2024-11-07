import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from FJSP_env import GraphFeature


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


def same_value_mask(seq1: Tensor, seq2: Tensor):
    s1 = list(seq1.size())
    s2 = list(seq2.size())
    s1.append(seq2.size(-1))
    s2.insert(-1, seq1.size(-1))
    cmp = seq1.unsqueeze(-1).expand(s1) == seq2.unsqueeze(-2).expand(s2)
    return cmp.to(torch.float)
