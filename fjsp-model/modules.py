import torch
from torch import nn, Tensor, tensor
from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch_geometric import nn as gnn
from torch_geometric.data import Batch
from torch_geometric.utils.hetero import check_add_self_loops
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeType, EdgeType
from fjsp_env import Graph, Action, ActionType, IdIdxMapper
from utils import *
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.trainer.states import RunningStage
from collections.abc import Callable
from tqdm import tqdm
from typing import Self
import math
from itertools import chain, batched, count
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import asyncio
from enum import IntEnum, auto
from pathlib import Path
from tqdm import tqdm


class ResidualLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels),
        )
        self.project = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels)
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        return self.norm(self.project(x) + self.model(x))


class HeteroConv(nn.Module):
    def __init__(
        self,
        convs: dict[EdgeType, gnn.MessagePassing],
        aggr: str | None = "sum",
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        self.convs = ModuleDict(convs)
        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_attr_dict: dict[EdgeType, Tensor] | None = None,
    ) -> dict[NodeType, Tensor]:
        out_dict: dict[str, list[Tensor]] = {}

        for edge_type, conv in self.convs.items():
            src, rel, dst = edge_type

            has_edge_level_arg = False

            dicts: list[dict] = [x_dict, edge_index_dict]
            if edge_attr_dict:
                dicts.append(edge_attr_dict)
            args = []
            for value_dict in dicts:
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (
                            value_dict.get(src, None),
                            value_dict.get(dst, None),
                        )
                    )
                else:
                    args.append(None)

            if not has_edge_level_arg:
                continue

            out = conv(*args)

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = gnn.conv.hetero_conv.group(value, self.aggr)

        return out_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_relations={len(self.convs)})"


class ExtractLayer(nn.Module):
    def __init__(
        self,
        node_channels: tuple[int, int, int],
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()

        self.residual = residual

        self.operation_relation_extract = HeteroConv(
            {
                ("operation", name, "operation"): gnn.Sequential(
                    "x, edge_index, edge_attr",
                    [
                        (
                            gnn.GATv2Conv(
                                node_channels[0],
                                node_channels[0],
                                dropout=dropout,
                                add_self_loops=False,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                        nn.LayerNorm(node_channels[0]),
                        nn.Tanh(),
                    ],
                )
                for name in ("predecessor", "successor")
            }
        )

        type_channel_map = {
            "operation": node_channels[0],
            "machine": node_channels[1],
            "AGV": node_channels[2],
        }
        self.hetero_relation_extract = HeteroConv(
            {
                edge: gnn.Sequential(
                    "x, edge_index, edge_attr",
                    [
                        (
                            gnn.GATv2Conv(
                                (type_channel_map[edge[0]], type_channel_map[edge[2]]),
                                type_channel_map[edge[2]],
                                edge_dim=Metadata.edge_attrs.get(edge, None),
                                dropout=dropout,
                                add_self_loops=False,
                            ),
                            "x, edge_index, edge_attr -> x",
                        ),
                        nn.LayerNorm(type_channel_map[edge[2]]),
                        nn.Tanh(),
                    ],
                )
                for edge in Metadata.edge_types
                if edge[1] not in ("predecessor", "successor")
            }
        )

        # self.norm = nn.ModuleDict(
        #     {k: nn.BatchNorm1d(v) for k, v in type_channel_map.items()}
        # )

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_attr_dict: dict[EdgeType, Tensor] | None = None,
    ):
        res: dict[NodeType, Tensor] = x_dict.copy() if self.residual else {}

        f1: dict[NodeType, Tensor] = self.operation_relation_extract(
            x_dict, edge_index_dict, edge_attr_dict
        )

        f2: dict[NodeType, Tensor] = self.hetero_relation_extract(
            x_dict, edge_index_dict, edge_attr_dict
        )

        for k, v in chain(f1.items(), f2.items()):
            if k in res:
                res[k] = res[k] + v
            else:
                res[k] = v

        # for k, v in res.items():
        #     res[k] = self.norm[k](v)

        return res


class StateMixer(nn.Module):
    def __init__(
        self,
        node_channels: tuple[int, int, int],
        global_channels: tuple[int, int, int],
        graph_global_channels: int,
    ):
        super().__init__()

        self.global_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(global_channels[idx]))
                for idx, name in enumerate(["operation", "machine", "AGV"])
            }
        )

        self.convs = HeteroConv(
            {
                (name, "global", f"{name}_global"): gnn.Sequential(
                    "x, edge_index",
                    [
                        (
                            gnn.GATv2Conv(
                                (node_channels[idx], global_channels[idx]),
                                global_channels[idx],
                                add_self_loops=False,
                            ),
                            "x, edge_index -> x",
                        ),
                        nn.LayerNorm(global_channels[idx]),
                        nn.Tanh(),
                    ],
                )
                for idx, name in enumerate(["operation", "machine", "AGV"])
            }
        )
        # self.type_norm = nn.ModuleDict(
        #     {
        #         f"{name}_global": nn.BatchNorm1d(global_channels[idx])
        #         for idx, name in enumerate(["operation", "machine", "AGV"])
        #     }
        # )

        self.graph_mix = nn.Sequential(
            ResidualLinear(
                Graph.global_feature_size + sum(global_channels),
                graph_global_channels * 2,
            ),
            # nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            ResidualLinear(graph_global_channels * 2, graph_global_channels * 2),
            # nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            nn.Linear(graph_global_channels * 2, graph_global_channels),
            # nn.BatchNorm1d(graph_global_channels),
        )

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        global_attr: Tensor,
    ):
        node_dict = x_dict.copy()
        edge_index_dict: dict[EdgeType, Tensor] = {}
        for n in Metadata.node_types:
            node_dict[f"{n}_global"] = self.global_tokens[n].unsqueeze(0)
            edge_index_dict[(n, "global", f"{n}_global")] = torch.stack(
                [
                    torch.arange(x_dict[n].shape[0], dtype=torch.int64),
                    torch.zeros(x_dict[n].shape[0], dtype=torch.int64),
                ]
            ).to(global_attr.device)

        type_dict: dict[NodeType, Tensor] = self.convs(node_dict, edge_index_dict)
        type_features = [
            type_dict[f"{name}_global"].squeeze(0) for name in ["operation", "machine", "AGV"]
        ]

        graph_feature = self.graph_mix(
            torch.cat(
                [global_attr] + type_features,
                0,
            )
        )

        return *type_features, graph_feature

    def train_forward(
        self,
        x_dict: dict[NodeType, Tensor],
        global_attr: Tensor,
        batch_dict: dict[NodeType, Tensor],
    ) -> tuple[dict[NodeType, Tensor], Tensor]:
        edge_index_dict: dict[EdgeType, Tensor] = {}
        node_dict = x_dict.copy()

        for k, batch in batch_dict.items():
            node_dict[f"{k}_global"] = self.global_tokens[k].repeat(
                batch.max().item() + 1, 1
            )
            edge_index_dict[(k, "global", f"{k}_global")] = (
                tensor([(i, b) for i, b in enumerate(batch)])
                .T.contiguous()
                .to(batch.device)
            )

        global_dict: dict[NodeType, Tensor] = self.convs(node_dict, edge_index_dict)
        # global_dict = {k: self.type_norm[k](v) for k, v in global_dict.items()}

        graph_feature = self.graph_mix(
            torch.cat(
                [global_attr]
                + [
                    global_dict[f"{name}_global"]
                    for name in ["operation", "machine", "AGV"]
                ],
                1,
            )
        )

        return global_dict, graph_feature


StateEmbedding = tuple[dict[NodeType, Tensor], dict[NodeType, Tensor], Tensor]


class StateExtract(nn.Module):

    def __init__(
        self,
        hidden_channels: tuple[int, int, int],
        global_channels: tuple[int, int, int],
        graph_global_channels: int,
        extract_num_layers: int,
    ):
        super().__init__()

        self.init_project = nn.ModuleDict(
            {
                "operation": ResidualLinear(
                    Graph.operation_feature_size, hidden_channels[0]
                ),
                "machine": ResidualLinear(
                    Graph.machine_feature_size, hidden_channels[1]
                ),
                "AGV": ResidualLinear(Graph.AGV_feature_size, hidden_channels[2]),
            }
        )

        self.backbone = nn.ModuleList(
            ExtractLayer(hidden_channels) for _ in range(extract_num_layers)
        )

        self.mix = StateMixer(
            hidden_channels,
            global_channels,
            graph_global_channels,
        )

    def forward(
        self,
        global_attr: Tensor,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_attr_dict: dict[EdgeType, Tensor],
    ):
        for k, v in x_dict.items():
            x_dict[k] = self.init_project[k](v)
  
        for layer in self.backbone:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        global_features = self.mix(x_dict, global_attr)

        return [*x_dict.values(), *global_features]

    def train_forward(self, data: Batch) -> StateEmbedding:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict: dict[EdgeType, Tensor] = data.collect(
            "edge_attr", allow_empty=True
        )
        batch_dict = {k: data[k].batch for k in x_dict}

        for k, v in x_dict.items():
            x_dict[k] = self.init_project[k](v)

        for layer in self.backbone:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        (
            global_dict,
            graph_feature,
        ) = self.mix.train_forward(x_dict, data.global_attr, batch_dict)

        return x_dict, global_dict, graph_feature


class ActionEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        node_channels: tuple[int, int, int],
        stack_num: tuple[int, int, int],
        hidden_channels: tuple[int, int, int],
    ):
        super().__init__()
        self.out_channels = out_channels
        self.node_channels = node_channels
        self.wait_emb = nn.Parameter(torch.zeros(out_channels))
        self.pick_encoder = nn.Sequential(
            ResidualLinear(
                (
                    node_channels[2]  # AGV
                    + node_channels[0]  # operation_from
                    + node_channels[0]  # operation_to
                    + node_channels[1]  # machine_target
                ),
                hidden_channels[0],
            ),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels[0], hidden_channels[0]),
                    nn.Tanh(),
                )
                for _ in range(stack_num[0])
            ],
            nn.Linear(hidden_channels[0], out_channels),
        )
        self.transport_encoder = nn.Sequential(
            ResidualLinear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                hidden_channels[1],
            ),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels[1], hidden_channels[1]),
                    nn.Tanh(),
                )
                for _ in range(stack_num[1])
            ],
            nn.Linear(hidden_channels[1], out_channels),
        )
        self.move_encoder = nn.Sequential(
            ResidualLinear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                hidden_channels[2],
            ),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels[2], hidden_channels[2]),
                    nn.Tanh(),
                )
                for _ in range(stack_num[2])
            ],
            nn.Linear(hidden_channels[2], out_channels),
        )

    def action_tensor_cat(
        self,
        batch_actions: list[list[Action]],
        embeddings: dict[str, Tensor],
        offsets: dict[str, dict[int, int]],
        idx_mappers: list[IdIdxMapper],
    ) -> list[list[Tensor]]:
        batch_tensors = []
        for i, (actions, idx_mapper) in enumerate(zip(batch_actions, idx_mappers)):
            tensors = []
            for action in actions:
                match action.action_type:
                    case ActionType.wait:
                        tensors.append(self.wait_emb)
                    case ActionType.pick:
                        AGV_emb = embeddings["AGV"][
                            offsets["AGV"][i] + idx_mapper.AGV[action.AGV_id]
                        ]
                        operation_from_emb = embeddings["operation"][
                            offsets["operation"][i]
                            + idx_mapper.operation[action.target_product.operation_from]
                        ]
                        operation_to_emb = embeddings["operation"][
                            offsets["operation"][i]
                            + idx_mapper.operation[action.target_product.operation_to]
                        ]
                        machine_target_emb = embeddings["machine"][
                            offsets["machine"][i]
                            + idx_mapper.machine[action.target_machine]
                        ]
                        tensors.append(
                            torch.cat(
                                [
                                    AGV_emb,
                                    operation_from_emb,
                                    operation_to_emb,
                                    machine_target_emb,
                                ]
                            )
                        )

                    case ActionType.transport:
                        AGV_emb = embeddings["AGV"][
                            offsets["AGV"][i] + idx_mapper.AGV[action.AGV_id]
                        ]
                        machine_target_emb = embeddings["machine"][
                            offsets["machine"][i]
                            + idx_mapper.machine[action.target_machine]
                        ]
                        tensors.append(
                            torch.cat(
                                [
                                    AGV_emb,
                                    machine_target_emb,
                                ]
                            )
                        )
                    case ActionType.move:
                        AGV_emb = embeddings["AGV"][
                            offsets["AGV"][i] + idx_mapper.AGV[action.AGV_id]
                        ]
                        machine_target_emb = embeddings["machine"][
                            offsets["machine"][i]
                            + idx_mapper.machine[action.target_machine]
                        ]
                        tensors.append(
                            torch.cat(
                                [
                                    AGV_emb,
                                    machine_target_emb,
                                ]
                            )
                        )
            batch_tensors.append(tensors)
        return batch_tensors

    def encode(
        self, types: list[list[ActionType]], batched_tensors: list[list[Tensor]]
    ):
        type_grouped_tensors = {
            ActionType.wait: [],
            ActionType.pick: [],
            ActionType.transport: [],
            ActionType.move: [],
        }
        for action_types, action_tensors in zip(types, batched_tensors):
            for action_type, action_tensor in zip(action_types, action_tensors):
                type_grouped_tensors[action_type].append(action_tensor)

        if len(type_grouped_tensors[ActionType.pick]):
            type_grouped_tensors[ActionType.pick][:] = torch.unbind(
                self.pick_encoder(torch.stack(type_grouped_tensors[ActionType.pick]))
            )
        if len(type_grouped_tensors[ActionType.transport]):
            type_grouped_tensors[ActionType.transport][:] = torch.unbind(
                self.transport_encoder(
                    torch.stack(type_grouped_tensors[ActionType.transport])
                )
            )
        if len(type_grouped_tensors[ActionType.move]):
            type_grouped_tensors[ActionType.move][:] = torch.unbind(
                self.move_encoder(torch.stack(type_grouped_tensors[ActionType.move]))
            )

        for batch_i, action_types in enumerate(types):
            for action_i, action_type in enumerate(action_types):
                batched_tensors[batch_i][action_i] = type_grouped_tensors[
                    action_type
                ].pop(0)
        return [torch.stack(tensors) for tensors in batched_tensors]

    def forward(self, embeds: dict[NodeType, Tensor], actions: dict[str, Tensor]):
        for action_type, action_idxs in actions.items():
            match action_type:
                case ActionType.wait.name:
                    wait_embs = self.wait_emb.unsqueeze(0).repeat(
                        action_idxs.shape[0], 1
                    )
                case ActionType.pick.name:
                    (
                        AGV_idx,
                        from_idx,
                        to_idx,
                        target_idx,
                    ) = torch.unbind(action_idxs, 1)
                    AGV_embeds = embeds["AGV"][AGV_idx]
                    from_embeds = embeds["operation"][from_idx]
                    to_embeds = embeds["operation"][to_idx]
                    target_embeds = embeds["machine"][target_idx]
                    raw = torch.cat(
                        [AGV_embeds, from_embeds, to_embeds, target_embeds], dim=1
                    )
                    pick_embs = self.pick_encoder(raw)
                case ActionType.transport.name:
                    AGV_idx, target_idx = torch.unbind(action_idxs, 1)
                    AGV_embeds = embeds["AGV"][AGV_idx]
                    target_embeds = embeds["machine"][target_idx]
                    raw = torch.cat([AGV_embeds, target_embeds], dim=1)
                    transport_embs = self.transport_encoder(raw)
                case ActionType.move.name:
                    AGV_idx, target_idx = torch.unbind(action_idxs, 1)
                    AGV_embeds = embeds["AGV"][AGV_idx]
                    target_embeds = embeds["machine"][target_idx]
                    raw = torch.cat([AGV_embeds, target_embeds], dim=1)
                    move_embs = self.move_encoder(raw)
        return torch.cat([wait_embs, pick_embs, transport_embs, move_embs])

    def train_forward(
        self,
        batch_actions: list[list[Action]],
        embeddings: dict[str, Tensor],
        offsets: dict[str, dict[int, int]],
        idx_mappers: list[IdIdxMapper],
    ):
        types = [
            [action.action_type for action in actions] for actions in batch_actions
        ]
        batched_tensors = self.action_tensor_cat(
            batch_actions,
            embeddings,
            offsets,
            idx_mappers,
        )
        return self.encode(types, batched_tensors)

    def single_action_encode(
        self,
        actions: list[Action],
        embeddings: dict[str, Tensor],
        offsets: dict[str, dict[int, int]],
        idx_mappers: list[IdIdxMapper],
    ):
        return torch.cat(
            self.forward(
                [[action] for action in actions],
                embeddings,
                offsets,
                idx_mappers,
            )
        )


class ActionDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        node_channels: tuple[int, int, int],
    ):
        super().__init__()

        self.cls_net = nn.Sequential(
            ResidualLinear(in_channels, in_channels),
            nn.Tanh(),
            ResidualLinear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, 4),
        )

        self.pick_decoder = nn.Sequential(
            ResidualLinear(in_channels, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(
                in_channels * 5,
                (
                    node_channels[2]
                    + node_channels[0]
                    + node_channels[0]
                    + node_channels[1]
                ),
            ),
        )

        self.transport_decoder = nn.Sequential(
            ResidualLinear(in_channels, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(
                in_channels * 5,
                (node_channels[1] + node_channels[2]),
            ),
        )

        self.move_decoder = nn.Sequential(
            ResidualLinear(in_channels, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            ResidualLinear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(
                in_channels * 5,
                (node_channels[1] + node_channels[2]),
            ),
        )

    def cls(self, actions: Tensor) -> Tensor:
        return self.cls_net(actions)

    def decode(
        self, pick_actions: Tensor, transport_actions: Tensor, move_actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.pick_decoder(pick_actions),
            self.transport_decoder(transport_actions),
            self.move_decoder(move_actions),
        )


class ValueNet(nn.Module):
    def __init__(self, state_channels: int, hidden_channels: int, stack_num: int):
        super().__init__()

        self.state_channels = state_channels

        self.model = nn.Sequential(
            ResidualLinear(state_channels, hidden_channels),
            # nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels, hidden_channels),
                    # nn.BatchNorm1d(hidden_channels),
                    nn.LeakyReLU(),
                )
                for _ in range(stack_num)
            ],
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, state: Tensor):
        return self.model(state).squeeze(-1)

    __call__: Callable[[Tensor], Tensor]


class PolicyNet(nn.Module):
    def __init__(
        self,
        state_channels: int,
        action_channels: int,
        hidden_channels: int,
        stack_num: int,
    ):
        super().__init__()

        self.state_channels = state_channels
        self.action_channels = action_channels

        base_channels = state_channels + action_channels
        self.model = nn.Sequential(
            ResidualLinear(base_channels, hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels, hidden_channels),
                    nn.Tanh(),
                )
                for _ in range(stack_num)
            ],
            nn.Linear(hidden_channels, 1),
            nn.Flatten(-2),
        )

    def forward(self, state: Tensor, actions: Tensor):
        if state.dim() < actions.dim():
            state = state.unsqueeze(-2).repeat_interleave(actions.shape[-2], -2)
        return self.model(torch.cat([state, actions], -1))

    __call__: Callable[[Tensor, Tensor], Tensor]


class PredictNet(nn.Module):
    def __init__(
        self,
        state_channels: int,
        action_channels: int,
        hidden_channels: int,
        stack_num: int,
    ):
        super().__init__()
        base_channels = state_channels + action_channels
        self.model = nn.Sequential(
            ResidualLinear(base_channels, hidden_channels),
            # nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels, hidden_channels),
                    # nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                )
                for _ in range(stack_num)
            ],
            nn.Linear(hidden_channels, state_channels),
        )

    def forward(self, state: Tensor, action: Tensor):
        if state.dim() < action.dim():
            s = list(state.shape)
            s.insert(-1, action.size(-2))
            state = state.unsqueeze(-2).expand(s)
        return self.model(torch.cat([state, action], -1))

    __call__: Callable[[Tensor, Tensor], Tensor]


class Node:
    def __init__(self, parent: Self | None, index: int, logit: float):
        self.parent = parent
        self.index = index
        self.logit = logit
        self.children: list[Node] = []
        self.value_list: list[float] = []
        self.visit_count = 0
        self.depth = 0 if parent is None else parent.depth + 1

    def expand(
        self,
        graph: Graph,
        actions: list[Action],
        reward: float,
        logits: Tensor,
    ):
        self.graph = graph.copy()
        self.actions = actions
        self.reward = reward
        for i in range(len(actions)):
            self.children.append(Node(self, i, logits[i].item()))

        self.finished = self.graph.finished()
        p = self
        while (p := p.parent) is not None:
            children_to_check = (
                [p.children[i] for i in p.remain_actions_idx]
                if isinstance(p, RootNode)
                else p.children
            )
            if all([c.expanded() and c.finished for c in children_to_check]):
                p.finished = True
            else:
                break

    def logits(self):
        return [child.logit for child in self.children]

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return np.mean(self.value_list)

    def sigma_Qs(self):
        sum_p_q = 0
        sum_prob = 0
        sum_visit = 0
        probs: list[float] = F.softmax(tensor(self.logits()), 0).tolist()
        for child in self.children:
            if child.expanded():
                sum_p_q += probs[child.index] * (
                    child.reward + 1.0 * child.value()
                )
                sum_prob += probs[child.index]
                sum_visit += child.visit_count
        if sum_prob < 1e-6:
            v_mix = self.value()
        else:
            v_mix = (1 / (1 + sum_visit)) * (
                self.value() + sum_visit / sum_prob * sum_p_q
            )

        completed_Qs = [
            (
                (child.reward + 1.0 * child.value())
                if child.expanded()
                else v_mix
            )
            for child in self.children
        ]
        completed_Qs = np.array(completed_Qs)
        child_visit_counts = np.array([child.visit_count for child in self.children])
        max_child_visit_count = child_visit_counts.max()
        return tensor((50 + max_child_visit_count) * 0.02 * completed_Qs)

    def improved_policy(self):
        logits = tensor(self.logits())
        sigma_Qs = self.sigma_Qs()
        # if self.depth == 0:
        #     print(logits, sigma_Qs)
        if logits.numel() > 1:
            logits = (logits - logits.mean()) / logits.std()
        else:
            logits = logits - logits.mean()
        if not (sigma_Qs == sigma_Qs[0]).all():
            sigma_Qs = (sigma_Qs - sigma_Qs.mean()) / sigma_Qs.std()
        else:
            sigma_Qs = sigma_Qs - sigma_Qs.mean()

        return F.softmax(logits + sigma_Qs, 0)

    def select_action(self):
        child_visit_counts = tensor([child.visit_count for child in self.children])
        finished_adj = tensor(
            [
                -torch.inf if child.expanded() and child.finished else 0
                for child in self.children
            ]
        )
        idx: int = torch.argmax(
            self.improved_policy()
            - child_visit_counts / (1 + child_visit_counts.sum())
            + finished_adj
        ).item()
        return self.children[idx]


class RootNode(Node):
    def __init__(self):
        super().__init__(None, -1, 1)

    def set_remain_actions_idx(self, idx: list[int]):
        self.remain_actions_idx = idx
        if all(
            [self.children[i].expanded() and self.children[i].finished for i in idx]
        ):
            self.finished = True

    def select_action(self):
        min_visit_count = self.visit_count + 1
        min_visit_idx = -1
        for idx in self.remain_actions_idx:
            if not self.children[idx].expanded():
                return self.children[idx]
            if (
                c := self.children[idx].visit_count < min_visit_count
                and not self.children[idx].finished
            ):
                min_visit_count = c
                min_visit_idx = idx
        return self.children[min_visit_idx]


class MCTS:
    def __init__(
        self,
        extractor: StateExtract,
        action_encoder: ActionEncoder,
        value_net: ValueNet,
        policy_net: PolicyNet,
    ):
        self.extractor = extractor
        self.action_encoder = action_encoder
        self.value_net = value_net
        self.policy_net = policy_net

    @torch.no_grad()
    def search(
        self,
        graphs: list[Graph],
        root_sample_count: int,
        simulation_count: int,
    ):
        assert root_sample_count & (root_sample_count - 1) == 0  # 2的幂

        device = next(self.extractor.parameters()).device

        obs = [Observation.from_env(g) for g in graphs]

        batched_graph = Batch.from_data_list([build_graph(o.feature) for o in obs])
        batched_graph.to(device)

        (
            node_states,
            _,
            graph_states,
        ) = self.extractor.train_forward(batched_graph)

        actions_list = [o.action_list for o in obs]
        actions_embs = self.action_encoder.train_forward(
            actions_list,
            node_states,
            get_offsets(batched_graph),
            [o.mapper for o in obs],
        )

        if root_sample_count == 1:
            idxs = []
            policies = []
            for state, action_embs in zip(graph_states, actions_embs):
                logits = self.policy_net(state, action_embs)
                idxs.append(torch.argmax(logits).item())
                policies.append(logits)
            return (
                idxs,
                self.value_net(graph_states),
                policies,
            )

        roots: list[RootNode] = []
        for ((graph, state, actions, actions_emb, value),) in zip(
            zip(
                graphs,
                graph_states,
                actions_list,
                actions_embs,
                self.value_net(graph_states),
            ),
        ):
            new_root = RootNode()
            new_root.visit_count += 1
            new_root.value_list.append(value.item())
            new_root.expand(
                graph,
                actions,
                0,
                self.policy_net(state, actions_emb),
            )
            roots.append(new_root)
        gs: list[Tensor] = []
        remaining_actions_index: list[list[int]] = []
        sim_SH_idxs: list[list[int]] = []
        sim_stages: list[int] = []
        for i, actions in enumerate(actions_embs):
            gumbel = torch.distributions.Gumbel(0, 1)
            g: Tensor = gumbel.sample([actions.size(0)]).to(device)
            gs.append(g)
            logits = tensor(roots[i].logits(), device=device)
            begin_count = min(root_sample_count, actions.size(0))
            sim_SH_idxs.append(
                self.SH_idx(begin_count, root_sample_count, simulation_count)
            )
            sim_stages.append(0)
            topk_indices = torch.topk(g + logits, begin_count).indices.tolist()
            roots[i].set_remain_actions_idx(topk_indices)
            remaining_actions_index.append(topk_indices)

        for sim_idx in count():
            target_leaves: list[Node] = []
            target_actions = []
            parent_graphs = []
            sim_state_idx: list[int] = []
            for i, (root, actions_index, counts, stage) in enumerate(
                zip(roots, remaining_actions_index, sim_SH_idxs, sim_stages)
            ):
                if stage == len(counts):
                    assert len(actions_index) == 1
                    continue

                sim_state_idx.append(i)
                if root.finished:
                    continue

                node = root.select_action()
                while node.expanded():
                    node = node.select_action()
                target_leaves.append(node)
                target_actions.append(node.parent.actions[node.index])
                parent_graphs.append(node.parent.graph)

            if len(sim_state_idx) == 0:
                break

            if len(target_leaves) > 0:
                env = Environment.from_graphs(parent_graphs)
                rewards, dones, obs = env.step(target_actions)
                batched_graph = Batch.from_data_list(
                    [build_graph(o.feature) for o in obs],
                ).to(device)
                (
                    node_states,
                    _,
                    graph_states,
                ) = self.extractor.train_forward(batched_graph)

                values = self.value_net(graph_states) * (
                    1 - tensor(dones, dtype=torch.float, device=device)
                )
                actions_list = [o.action_list for o in obs]
                actions_embs = self.action_encoder.train_forward(
                    actions_list,
                    node_states,
                    get_offsets(batched_graph),
                    [o.mapper for o in obs],
                )

                for node, (
                    (
                        graph,
                        states,
                        actions,
                        actions_emb,
                        reward,
                    ),
                    value,
                ) in zip(
                    target_leaves,
                    zip(
                        zip(
                            env.envs,
                            graph_states,
                            actions_list,
                            actions_embs,
                            rewards,
                        ),
                        values,
                    ),
                ):
                    node.expand(
                        graph,
                        actions,
                        reward,
                        self.policy_net(states, actions_emb),
                    )
                    while True:
                        node.visit_count += 1
                        node.value_list.append(value.item())
                        if (p := node.parent) is not None:
                            value = node.reward + 1.0 * value
                            node = p
                        else:
                            break

            for idx in sim_state_idx:
                if sim_SH_idxs[idx][sim_stages[idx]] == sim_idx:
                    new_idx = self.sequential_halving(
                        roots[idx], gs[idx], remaining_actions_index[idx]
                    )
                    roots[idx].set_remain_actions_idx(new_idx)
                    remaining_actions_index[idx] = new_idx
                    sim_stages[idx] += 1

        assert all(len(idxs) == 1 for idxs in remaining_actions_index)
        target_values = tensor(
            [root.value() for root in roots],
            dtype=torch.float,
            device=device,
        )
        target_policies = [root.improved_policy().float().to(device) for root in roots]
        return (
            [idxs[0] for idxs in remaining_actions_index],
            target_values,
            target_policies,
        )

    @staticmethod
    def SH_idx(m, old_m, old_n):
        if old_m < 2:
            return []
        m = 2 ** math.ceil(math.log2(m))
        # n = math.floor(m * old_n / old_m * math.log(m, old_m))
        n = math.floor(old_n * math.log(m, old_m))
        counts = []
        remained = m
        sum_count = 0
        while remained > 1:
            sum_count += math.floor(n / (math.log2(m) * remained)) * remained
            counts.append(sum_count)
            remained //= 2
        return counts

    def phase_step_num(
        self, root_sample_count, remaining_action_count, simulation_count
    ):
        return (
            0
            if root_sample_count == 1
            else (
                math.floor(
                    simulation_count
                    / (math.log2(root_sample_count) * remaining_action_count)
                )
                * remaining_action_count
            )
        )

    def sequential_halving(self, root: Node, gs: Tensor, action_idx: list[int]):
        g = gs[action_idx]
        logits = tensor(root.logits())[action_idx].to(gs.device)
        sigma_Qs = root.sigma_Qs()[action_idx].to(gs.device)
        remain_idx = torch.topk(
            g + logits + sigma_Qs, math.ceil(len(action_idx) / 2)
        ).indices
        return [action_idx[idx] for idx in remain_idx.tolist()]


class Model(nn.Module):

    def __init__(
        self,
        node_channels: tuple[int, int, int],
        type_channels: tuple[int, int, int],
        graph_channels: int,
        state_layer_num: int,
        action_channels: int,
        action_hidden_channel: tuple[int, int, int],
        action_stack_num: tuple[int, int, int],
        value_hidden_channel: int,
        value_stack_num: int,
        policy_hidden_channel: int,
        policy_stack_num: int,
    ):
        super().__init__()

        self.extractor = StateExtract(
            node_channels,
            type_channels,
            graph_channels,
            state_layer_num,
        )

        self.action_encoder = ActionEncoder(
            action_channels,
            node_channels,
            action_stack_num,
            action_hidden_channel,
        )

        self.value_net = ValueNet(
            graph_channels,
            value_hidden_channel,
            value_stack_num,
        )

        self.policy_net = PolicyNet(
            graph_channels,
            action_channels,
            policy_hidden_channel,
            policy_stack_num,
        )

        self.mcts = MCTS(
            self.extractor,
            self.action_encoder,
            self.value_net,
            self.policy_net,
        )

    @torch.no_grad
    def single_step_predict(
        self, graphs: list[Graph], sample_count: int, sim_count: int
    ):
        (
            act_idxs,
            _,
            _,
        ) = self.mcts.search(
            graphs,
            sample_count,
            sim_count,
        )
        return [
            g.get_available_actions()[idx] for g, idx in zip(graphs, act_idxs)
        ], act_idxs

    @torch.inference_mode()
    async def predict(
        self,
        graph: Graph,
        sample_count: int,
        sim_count: int,
    ):
        env = Environment.from_graphs([graph], True)

        for round_count in count(1):
            actions, _ = self.single_step_predict(
                env.envs,
                sample_count,
                sim_count,
            )
            env.step(actions, False)

            finished_step, total_step = env.envs[0].progress()

            yield {
                "round_count": round_count,
                "finished_step": finished_step,
                "total_step": total_step,
                "graph_state": env.envs[0],
                "action": actions[0],
            }

            if env.envs[0].finished():
                break

            await asyncio.sleep(0)


class Agent(L.LightningModule):
    
    class TrainStage(IntEnum):
        encode = auto()
        _value = auto()
        policy = auto()
        explore = auto()

    def __init__(
        self,
        envs: Environment | None,
        lr: float,
        opt_step_size: int,
        seq_len: int,
        buffer_size: int,
        epoch_size: int,
        batch_size: int,
        node_channels: tuple[int, int, int],
        type_channels: tuple[int, int, int],
        graph_channels: int,
        state_layer_num: int,
        action_channels: int,
        action_hidden_channel: tuple[int, int, int],
        action_stack_num: tuple[int, int, int],
        value_hidden_channel: int,
        value_stack_num: int,
        policy_hidden_channel: int,
        policy_stack_num: int,
        predict_hidden_channel: int,
        predict_stack_num: int,
        root_sample_count: int,
        simulation_count: int,
        val_predict_num: int,
        stage: TrainStage,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["envs", "finished_batch_count"])

        self.envs = envs

        self.lr = lr
        self.opt_step_size = opt_step_size

        self.seq_len = seq_len

        self.epoch_size = epoch_size
        self.batch_size = batch_size

        self.model = Model(
            node_channels,
            type_channels,
            graph_channels,
            state_layer_num,
            action_channels,
            action_hidden_channel,
            action_stack_num,
            value_hidden_channel,
            value_stack_num,
            policy_hidden_channel,
            policy_stack_num,
        )
        self.predictor = PredictNet(
            graph_channels,
            action_channels,
            predict_hidden_channel,
            predict_stack_num,
        )
        self.action_decoder = ActionDecoder(action_channels, node_channels)
        self.value_target = ValueNet(
            graph_channels,
            value_hidden_channel,
            value_stack_num,
        )
        self.policy_target = PolicyNet(
            graph_channels,
            action_channels,
            policy_hidden_channel,
            policy_stack_num,
        )
        self.mcts = MCTS(
            self.model.extractor,
            self.model.action_encoder,
            self.value_target,
            self.policy_target,
        )

        self.extractor_norm_clipper = NormClipper(self.model.extractor)
        self.predictor_norm_clipper = NormClipper(self.predictor)
        self.action_encoder_norm_clipper = NormClipper(self.model.action_encoder)
        self.action_decoder_norm_clipper = NormClipper(self.action_decoder)
        self.value_net_norm_clipper = NormClipper(self.model.value_net)
        self.policy_net_norm_clipper = NormClipper(self.model.policy_net)

        self.root_sample_count = root_sample_count
        self.simulation_count = simulation_count

        self.buffer = ReplayBuffer(seq_len, buffer_size)

        self.val_predict_num = val_predict_num

        self.stage = stage

    def load(self, path: str, stage: TrainStage):
        ckpt = torch.load(path, weights_only=False)
        states: dict[str, Tensor] = ckpt["state_dict"]
        load_modules = []
        if stage >= Agent.TrainStage.encode:
            load_modules.extend(
                [
                    "extractor",
                    "action_encoder",
                ]
            )

        if stage >= Agent.TrainStage._value:
            load_modules.extend(["value_net"])

        if stage >= Agent.TrainStage.policy:
            load_modules.extend(["policy_net"])

        for module_name in load_modules:
            getattr(self.model, module_name).load_state_dict(
                {
                    k[len(module_name) + 7 :]: v
                    for (k, v) in states.items()
                    if k.startswith(f"model.{module_name}.")
                }
            )

    def compile_modules(self):
        torch.compile(self.model, fullgraph=True)
        torch.compile(self.predictor, fullgraph=True)
        torch.compile(self.action_decoder, fullgraph=True)
        torch.compile(self.value_target, fullgraph=True)
        torch.compile(self.policy_target, fullgraph=True)

    def prepare_data(self):
        assert self.envs is not None
        self.eval()
        self.init_buffer()
        self.train()

        self.baseline, self.val_data = self.get_baseline()

    def get_baseline(self) -> tuple[np.ndarray | None, list[Graph]]:
        if self.stage == Agent.TrainStage.encode:
            return None, [None]
        if self.stage == Agent.TrainStage._value:
            return None, [None]
        if self.stage >= Agent.TrainStage.policy:
            temp_envs = [
                Environment(1, [params], False) for params in self.envs.generate_params
            ]
            timestamps: list[float] = []
            graphs: list[Graph] = []
            for env_i, env in enumerate(temp_envs):
                env_timestamps: list[float] = []
                for _ in tqdm(
                    range(self.val_predict_num), f"Baseline_{env_i}", leave=False
                ):
                    env.reset(True)
                    while True:
                        obs = env.observe()
                        act, _ = single_step_useful_only_predict(obs[0])
                        _, done, _ = env.step([act])
                        if done[0]:
                            break
                    env_timestamps.append(env.envs[0].get_timestamp())
                timestamps.append(np.mean(env_timestamps))
                graphs.append(env.envs[0].reset())
            return np.array(timestamps), graphs

    def init_buffer(self):
        progress = tqdm(total=self.epoch_size, desc="Init Buffer")
        while True:
            obs = self.envs.observe()
            graphs = [g.copy() for g in self.envs.envs]
            if self.stage == Agent.TrainStage.explore:
                acts, act_idxs = self.model.single_step_predict(
                    [g.copy() for g in self.envs.envs],
                    self.root_sample_count,
                    self.simulation_count,
                )
            else:
                acts = []
                act_idxs = []
                for ob in obs:
                    act, act_idx = single_step_useful_only_predict(ob, 0.1)
                    acts.append(act)
                    act_idxs.append(act_idx)
            rewards, dones, _ = self.envs.step(acts)
            prev_buffer_size = len(self.buffer.buffer)
            self.buffer.append(
                graphs,
                act_idxs,
                rewards,
                dones,
                [g.copy() for g in self.envs.envs],
            )
            buffer_size = len(self.buffer.buffer)
            progress.update(buffer_size - prev_buffer_size)
            if progress.n >= progress.total:
                break

    def configure_optimizers(self):
        encode_stage_opt = optim.Adam(
            chain(
                self.model.extractor.parameters(),
                self.predictor.parameters(),
                self.model.action_encoder.parameters(),
                self.action_decoder.parameters(),
            ),
            self.lr,
        )
        encode_stage_sch = lr_scheduler.StepLR(
            encode_stage_opt,
            self.opt_step_size,
            0.99,
        )

        value_opt = optim.SGD(
            self.model.value_net.parameters(),
            self.lr,
        )
        value_sch = lr_scheduler.StepLR(
            value_opt,
            self.opt_step_size,
            0.95,
        )

        policy_opt = optim.SGD(
            chain(
                self.model.value_net.parameters(), self.model.policy_net.parameters()
            ),
            self.lr,
        )
        policy_sch = lr_scheduler.StepLR(
            policy_opt,
            self.opt_step_size,
            0.95,
        )

        explore_opt = optim.Adam(
            self.model.parameters(),
            self.lr,
            weight_decay=1e-2,
        )
        explore_sch = lr_scheduler.StepLR(
            explore_opt,
            self.opt_step_size,
            0.93,
        )

        return [
            encode_stage_opt,
            value_opt,
            policy_opt,
            explore_opt,
        ], [
            encode_stage_sch,
            value_sch,
            policy_sch,
            explore_sch,
        ]

    def train_dataloader(self):
        return DataLoader(
            ReplayDataset(self.buffer, self.epoch_size),
            self.batch_size,
            collate_fn=lambda batch: batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            len(self.val_data),
            collate_fn=lambda batch: batch,
        )

    def on_train_start(self):
        match self.stage:
            case Agent.TrainStage._value:
                self.value_target.load_state_dict(self.model.value_net.state_dict())
            case Agent.TrainStage.policy | Agent.TrainStage.explore:
                self.value_target.load_state_dict(self.model.value_net.state_dict())
                self.policy_target.load_state_dict(self.model.policy_net.state_dict())

    @torch.no_grad
    def play_step(self):
        for _ in range(5):
            obs = self.envs.observe()
            graphs = [g.copy() for g in self.envs.envs]
            if self.stage == Agent.TrainStage.explore:
                acts, act_idxs = self.model.single_step_predict(
                    [g.copy() for g in self.envs.envs],
                    1,
                    0,
                )
            else:
                acts = []
                act_idxs = []
                for ob in obs:
                    act, act_idx = single_step_useful_only_predict(ob, 0.1)
                    acts.append(act)
                    act_idxs.append(act_idx)
            rewards, dones, _ = self.envs.step(acts)
            self.buffer.append(
                graphs,
                act_idxs,
                rewards,
                dones,
                [g.copy() for g in self.envs.envs],
            )

    def encode_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):
        obs = [[Observation.from_env(g) for g in item.graphs] for item in items]
        graphs = [build_graph(o.feature) for o in chain(*obs)]
        batched_graph = Batch.from_data_list(graphs).to(self.device)

        next_obs = [
            [Observation.from_env(g) for g in item.next_graphs] for item in items
        ]
        next_graphs = [build_graph(o.feature) for o in chain(*next_obs)]
        batched_next_graph = Batch.from_data_list(next_graphs).to(self.device)

        (
            node_states,
            _,
            graph_states,
        ) = self.model.extractor.train_forward(batched_graph)
        graph_states = rearrange(
            graph_states,
            "(n l) s -> n l s",
            l=self.seq_len,
        )
        with torch.no_grad():
            next_graph_states = rearrange(
                self.model.extractor.train_forward(batched_next_graph)[2],
                "(n l) s -> n l s",
                l=self.seq_len,
            )

        graph_diff_loss = (
            torch.clip(0.5 - graph_states.var((0, 1)), 0)
            + torch.clip(graph_states.var((0, 1)) - 1, 0)
        ).mean()

        offsets = get_offsets(batched_graph)
        mappers = sum(
            [[o.mapper for o in os] for os in obs],
            [],
        )
        actions = sum(
            [[o.action_list for o in os] for os in obs],
            [],
        )
        types = [[action.action_type for action in actions] for actions in actions]
        raw_actions_tensors = self.model.action_encoder.action_tensor_cat(
            actions,
            node_states,
            offsets,
            mappers,
        )
        actions_embs = self.model.action_encoder.encode(
            types, [[*tensors] for tensors in raw_actions_tensors]
        )
        cated_actions_embs = torch.cat(actions_embs)

        pred_cls = self.action_decoder.cls(cated_actions_embs)
        target_cls = torch.tensor([t.value for t in sum(types, [])], device=self.device)
        cls_loss = F.cross_entropy(pred_cls, target_cls)

        decoded_picks, decoded_transports, decoded_moves = self.action_decoder.decode(
            cated_actions_embs[target_cls == ActionType.pick.value],
            cated_actions_embs[target_cls == ActionType.transport.value],
            cated_actions_embs[target_cls == ActionType.move.value],
        )
        true_picks = []
        true_transports = []
        true_moves = []
        for ts, acts in zip(types, raw_actions_tensors):
            for t, a in zip(ts, acts):
                match t:
                    case ActionType.pick:
                        true_picks.append(a)
                    case ActionType.transport:
                        true_transports.append(a)
                    case ActionType.move:
                        true_moves.append(a)

        decode_loss = (
            F.mse_loss(decoded_picks, torch.stack(true_picks))
            + F.mse_loss(decoded_transports, torch.stack(true_transports))
            + F.mse_loss(decoded_moves, torch.stack(true_moves))
        )

        sim_mat = [
            F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), 2)
            for embs in actions_embs
        ]
        type_idxs = [
            tensor([action.action_type.value for action in acts]).to(self.device)
            for acts in actions
        ]
        type_diff_mask = [
            type_idx.unsqueeze(0).expand(type_idx.size(0), type_idx.size(0))
            != type_idx.unsqueeze(1).expand(type_idx.size(0), type_idx.size(0))
            for type_idx in type_idxs
        ]

        type_diff_loss = torch.mean(
            torch.stack(
                [
                    (
                        tensor(0).to(self.device)
                        if mat.size(0) == 1
                        else (torch.sum(mat * mask) / torch.sum(mask))
                    )
                    for mat, mask in zip(sim_mat, type_diff_mask)
                ]
            )
        )

        action_diff_loss = torch.mean(
            torch.stack(
                [
                    (
                        tensor(0).to(self.device)
                        if mat.size(0) == 1
                        else (
                            torch.sum(
                                mat * (1 - torch.eye(mat.size(0)).to(self.device))
                            )
                            / (mat.size(0) * (mat.size(0) - 1))
                        )
                    )
                    for mat in sim_mat
                ]
            )
        )

        action_embs = torch.stack(
            [
                torch.stack(
                    [
                        actions_embs[i * self.seq_len + seq_step][
                            item.action_idxs[seq_step]
                        ]
                        for seq_step in range(self.seq_len)
                    ]
                )
                for i, item in enumerate(items)
            ]
        )
        pred_next_states = self.predictor(graph_states, action_embs)
        # pred_next_states = []
        # prev_states = graph_states[:, 0]
        # for pred_step in range(self.seq_len):
        #     action_embs = torch.stack(
        #         [
        #             actions_embs[i * self.seq_len + pred_step][
        #                 item.action_idxs[pred_step]
        #             ]
        #             for i, item in enumerate(items)
        #         ]
        #     )
        #     new_pred_next_states = self.predictor(prev_states, action_embs)
        #     pred_next_states.append(new_pred_next_states)
        #     prev_states = new_pred_next_states
        # pred_next_states = torch.stack(pred_next_states, 1)

        self.log_dict(
            {
                "info/next_state_scale": next_graph_states.abs().mean(),
                "info/pred_state_scale": pred_next_states.abs().mean(),
            }
        )

        # pred_state_sim_losses = -torch.stack(
        #     [
        #         F.cosine_similarity(p, n, -1).mean()
        #         for p, n in zip(
        #             torch.unbind(pred_next_states, 1),
        #             torch.unbind(next_graph_states, 1),
        #         )
        #     ]
        # )

        pred_state_sim_loss = F.mse_loss(pred_next_states, next_graph_states)

        # self.log_dict(
        #     {f"info/pred_sim_{i}": v for i, v in enumerate(pred_state_sim_losses)}
        # )

        # pred_state_mse_losses = torch.stack(
        #     [
        #         F.mse_loss(p, n)
        #         for p, n in zip(
        #             torch.unbind(pred_next_states, 1),
        #             torch.unbind(next_graph_states, 1),
        #         )
        #     ]
        # )
        # self.log_dict(
        #     {f"info/pred_mse_{i}": v for i, v in enumerate(pred_state_mse_losses)}
        # )

        opt.zero_grad()
        self.manual_backward(
            graph_diff_loss / graph_diff_loss.detach().abs()
            + cls_loss / cls_loss.detach().abs()
            + decode_loss / decode_loss.detach().abs()
            + type_diff_loss / type_diff_loss.detach().abs()
            + action_diff_loss / action_diff_loss.detach().abs()
            + pred_state_sim_loss / pred_state_sim_loss.detach().abs()
        )
        self.extractor_norm_clipper.clip()
        self.action_encoder_norm_clipper.clip()
        self.action_decoder_norm_clipper.clip()
        self.predictor_norm_clipper.clip()
        opt.step()

        self.log_dict(
            {
                "loss/graph_diff": graph_diff_loss,
                "loss/cls": cls_loss,
                "loss/decode": decode_loss,
                "loss/type_diff": type_diff_loss,
                "loss/action_diff": action_diff_loss,
                "loss/pred_state_sim": pred_state_sim_loss,
            }
        )

    def value_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):
        self.eval()
        with torch.no_grad():
            obs = [[Observation.from_env(g) for g in item.graphs] for item in items]
            graphs = [build_graph(o.feature) for o in chain(*obs)]
            batched_graph = Batch.from_data_list(graphs).to(self.device)
            graph_states = rearrange(
                self.model.extractor.train_forward(batched_graph)[2],
                "(n l) s -> n l s",
                l=self.seq_len,
            )

            last_obs = [Observation.from_env(item.next_graphs[-1]) for item in items]
            last_graphs = [build_graph(o.feature) for o in last_obs]
            batched_last_graph = Batch.from_data_list(last_graphs).to(self.device)
            last_graph_states = self.model.extractor.train_forward(batched_last_graph)[
                2
            ]

            dones = tensor(
                [item.dones[-1] for item in items],
                dtype=torch.float,
                device=self.device,
            )
            target_value = self.value_target(last_graph_states) * dones
            for i in reversed(range(self.seq_len)):
                target_value += tensor(
                    [item.rewards[i] for item in items], device=self.device
                )
        self.train()

        pred_value = self.model.value_net(graph_states[:, 0])
        value_loss = F.mse_loss(pred_value, target_value)
        self.log_dict(
            {
                "loss/value": value_loss,
            }
        )
        opt.zero_grad()
        self.manual_backward(value_loss / value_loss.detach().abs())
        self.value_net_norm_clipper.clip()
        self.policy_net_norm_clipper.clip()
        opt.step()

        eps = 0.005
        for t, n in zip(
            self.value_target.parameters(), self.model.value_net.parameters()
        ):
            t.data.copy_(t.data * (1 - eps) + n.data * eps)

    def policy_stage(
        self,
        items: list[SequenceReplayItem],
        opt: LightningOptimizer,
    ):
        self.eval()
        with torch.no_grad():
            obs = [[Observation.from_env(g) for g in item.graphs] for item in items]
            graphs = [build_graph(o.feature) for o in chain(*obs)]
            batched_graph = Batch.from_data_list(graphs).to(self.device)
            (
                node_states,
                _,
                graph_states,
            ) = self.model.extractor.train_forward(batched_graph)

            offsets = get_offsets(batched_graph)
            mappers = sum(
                [[o.mapper for o in os] for os in obs],
                [],
            )
            actions = sum(
                [[o.action_list for o in os] for os in obs],
                [],
            )

            actions_embs = self.model.action_encoder.train_forward(
                actions,
                node_states,
                offsets,
                mappers,
            )
            (
                _,
                target_value,
                target_policy,
            ) = self.mcts.search(
                sum([item.graphs for item in items], []),
                self.root_sample_count,
                self.simulation_count,
            )
        self.train()

        pred_value = self.model.value_net(graph_states)
        pred_policy = []
        policy_logits = []
        for state, embs in zip(graph_states, actions_embs):
            logits = self.model.policy_net(state, embs)
            pred_policy.append(logits.softmax(-1))
            policy_logits.append(logits)
        policy_logits = torch.cat(policy_logits)
        self.log_dict(
            {
                "info/policy_logits_mean": policy_logits.mean(),
                "info/policy_logits_std": policy_logits.std(),
            }
        )

        value_loss = F.mse_loss(pred_value, target_value)
        policy_loss = torch.mean(
            torch.stack(
                [
                    F.kl_div(torch.log(target), pred, reduction="batchmean")
                    for pred, target in zip(pred_policy, target_policy)
                ]
            )
        )
        self.log_dict(
            {
                "loss/value": value_loss,
                "loss/policy": policy_loss,
            }
        )

        opt.zero_grad()
        self.manual_backward(
            value_loss / value_loss.detach().abs()
            + policy_loss / policy_loss.detach().abs()
        )
        self.value_net_norm_clipper.clip()
        self.policy_net_norm_clipper.clip()
        opt.step()

        eps = 0.005
        for t, n in zip(
            self.value_target.parameters(), self.model.value_net.parameters()
        ):
            t.data.copy_(t.data * (1 - eps) + n.data * eps)
        for t, n in zip(
            self.policy_target.parameters(), self.model.policy_net.parameters()
        ):
            t.data.copy_(t.data * (1 - eps) + n.data * eps)

    def explore_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):

        obs = [[Observation.from_env(g) for g in item.graphs] for item in items]
        graphs = [build_graph(o.feature) for o in chain(*obs)]
        batched_graph = Batch.from_data_list(graphs).to(self.device)
        (
            node_states,
            _,
            graph_states,
        ) = self.model.extractor.train_forward(batched_graph)

        offsets = get_offsets(batched_graph)
        mappers = sum(
            [[o.mapper for o in os] for os in obs],
            [],
        )
        actions = sum(
            [[o.action_list for o in os] for os in obs],
            [],
        )

        actions_embs = self.model.action_encoder.train_forward(
            actions,
            node_states,
            offsets,
            mappers,
        )
        self.eval()
        with torch.no_grad():
            (
                _,
                target_value,
                target_policy,
            ) = self.mcts.search(
                sum([item.graphs for item in items], []),
                self.root_sample_count,
                self.simulation_count,
            )
        self.train()

        pred_value = self.model.value_net(graph_states)
        pred_policy = []
        for state, embs in zip(graph_states, actions_embs):
            logits = self.model.policy_net(state, embs)
            pred_policy.append(logits.softmax(-1))

        value_loss = F.mse_loss(pred_value, target_value)
        policy_loss = torch.mean(
            torch.stack(
                [
                    F.kl_div(torch.log(target), pred, reduction="batchmean")
                    for pred, target in zip(pred_policy, target_policy)
                ]
            )
        )
        self.log_dict(
            {
                "loss/value": value_loss,
                "loss/policy": policy_loss,
            }
        )

        opt.zero_grad()
        self.manual_backward(
            value_loss / value_loss.detach() + policy_loss / policy_loss.detach()
        )
        state_norm = self.extractor_norm_clipper.clip()
        action_norm = self.action_encoder_norm_clipper.clip()
        value_norm = self.value_net_norm_clipper.clip()
        policy_norm = self.policy_net_norm_clipper.clip()
        self.log_dict(
            {
                "info/state_norm": state_norm,
                "info/action_norm": action_norm,
                "info/value_norm": value_norm,
                "info/policy_norm": policy_norm,
            }
        )
        self.log_dict(
            {
                "info/state_max_norm": self.extractor_norm_clipper.max_norm,
                "info/action_max_norm": self.action_encoder_norm_clipper.max_norm,
                "info/value_max_norm": self.value_net_norm_clipper.max_norm,
                "info/policy_max_norm": self.policy_net_norm_clipper.max_norm,
            }
        )
        opt.step()

        eps = 0.005
        for t, n in zip(
            self.value_target.parameters(), self.model.value_net.parameters()
        ):
            t.data.copy_(t.data * (1 - eps) + n.data * eps)
        for t, n in zip(
            self.policy_target.parameters(), self.model.policy_net.parameters()
        ):
            t.data.copy_(t.data * (1 - eps) + n.data * eps)

    def training_step(self, items: list[SequenceReplayItem]):
        (
            encode_stage_opt,
            value_opt,
            policy_opt,
            explore_opt,
        ) = self.optimizers()
        (
            encode_stage_sch,
            value_sch,
            policy_sch,
            explore_sch,
        ) = self.lr_schedulers()
        match self.stage:
            case Agent.TrainStage.encode:
                self.encode_stage(items, encode_stage_opt)
                # encode_stage_sch.step()
            case Agent.TrainStage._value:
                self.value_stage(items, value_opt)
            case Agent.TrainStage.policy:
                # self.policy_stage(items, policy_opt)
                # value_stage_sch.step()
                # policy_sch.step()
                self.explore_stage(items, explore_opt)
                explore_sch.step()
            case Agent.TrainStage.explore:
                self.explore_stage(items, explore_opt)
                # explore_sch.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.eval()
        self.play_step()
        self.train()

    def validation_step(self, batch: list[Graph]):
        match self.stage:
            case Agent.TrainStage.policy | Agent.TrainStage.explore:
                bar: tqdm = self.trainer.progress_bar_callback.val_progress_bar
                for sample_count, sim_count in [(1, 0), (2, 8), (4, 32)]:
                    bar.set_description(f"Validation [{sample_count}/{sim_count}]")
                    env = Environment.from_graphs(
                        [
                            graph.copy()
                            for graph in batch
                            for _ in range(self.val_predict_num)
                        ],
                        True,
                    )
                    while any(
                        [
                            np.mean([g.get_timestamp() for g in gs])
                            < self.baseline[i] * 1.2
                            for i, gs in enumerate(
                                batched(env.envs, self.val_predict_num)
                            )
                        ]
                    ):
                        obs = env.observe()
                        if len(obs) == 0:
                            break

                        actions, _ = self.model.single_step_predict(
                            [g for g in env.envs if not g.finished()],
                            sample_count,
                            sim_count,
                        )
                        env.step(actions)

                        total_step = 0
                        finished_step = 0
                        for e in env.envs:
                            f, t = e.progress()
                            total_step += t
                            finished_step += f

                        bar.total = total_step
                        bar.n = finished_step
                        bar.refresh()

                    self.log_dict(
                        {
                            f"val/env_{idx} [{sample_count}/{sim_count}]": (
                                (
                                    np.mean([g.get_timestamp() for g in graphs])
                                    / self.baseline[idx]
                                )
                                if all(g.finished() for g in graphs)
                                else np.nan
                            )
                            for idx, graphs in enumerate(
                                batched(
                                    env.envs,
                                    self.val_predict_num,
                                )
                            )
                        },
                        batch_size=len(batch),
                    )
