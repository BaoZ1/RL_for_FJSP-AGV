import torch
from torch import nn, Tensor, tensor
from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric import nn as gnn
from torch_geometric.typing import NodeType, EdgeType
from FJSP_env import (
    Graph,
    Environment,
    Action,
    ActionType,
    IdIdxMapper,
    Observation,
    single_step_useful_only_predict,
    single_step_useful_first_predict,
)
from .utils import *
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
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

    def forward(self, x):
        return self.project(x) + self.model(x)


class ExtractLayer(nn.Module):
    def __init__(
        self,
        node_channels: tuple[int, int, int],
        dropout: float = 0,
        residual: bool = True,
    ):
        super().__init__()

        self.residual = residual

        self.operation_relation_extract = gnn.HeteroConv(
            {
                ("operation", name, "operation"): gnn.GATv2Conv(
                    node_channels[0],
                    node_channels[0],
                    dropout=dropout,
                    add_self_loops=False,
                )
                for name in ["predecessor", "successor"]
            }
        )

        type_channel_map = {
            "operation": node_channels[0],
            "machine": node_channels[1],
            "AGV": node_channels[2],
        }
        self.hetero_relation_extract = gnn.HeteroConv(
            {
                edge: gnn.GATv2Conv(
                    (type_channel_map[edge[0]], type_channel_map[edge[2]]),
                    type_channel_map[edge[2]],
                    edge_dim=Metadata.edge_attrs.get(edge, None),
                    dropout=dropout,
                    add_self_loops=False,
                )
                for edge in Metadata.edge_types
                if edge[0] != edge[2] != "operation"
            }
        )

        self.norm = nn.ModuleDict(
            {k: nn.BatchNorm1d(v) for k, v in type_channel_map.items()}
        )

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

        for k, v in res.items():
            res[k] = self.norm[k](v)

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

        self.convs = gnn.HeteroConv(
            {
                (name, "global", f"{name}_global"): gnn.GATv2Conv(
                    (node_channels[idx], global_channels[idx]),
                    global_channels[idx],
                    add_self_loops=False,
                )
                for idx, name in enumerate(["operation", "machine", "AGV"])
            }
        )
        self.type_norm = nn.ModuleDict(
            {
                f"{name}_global": nn.BatchNorm1d(global_channels[idx])
                for idx, name in enumerate(["operation", "machine", "AGV"])
            }
        )

        self.graph_mix = nn.Sequential(
            ResidualLinear(
                sum(global_channels) + Graph.global_feature_size,
                graph_global_channels * 2,
            ),
            nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            ResidualLinear(graph_global_channels * 2, graph_global_channels * 2),
            nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            ResidualLinear(graph_global_channels * 2, graph_global_channels),
            # nn.BatchNorm1d(graph_global_channels),
        )

    def forward(
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
                tensor(list(enumerate(batch))).T.contiguous().to(batch.device)
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
                "operation": nn.Linear(
                    Graph.operation_feature_size, hidden_channels[0]
                ),
                "machine": nn.Linear(Graph.machine_feature_size, hidden_channels[1]),
                "AGV": nn.Linear(Graph.AGV_feature_size, hidden_channels[2]),
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

    def forward(self, data: HeteroGraph | Batch) -> StateEmbedding:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict: dict[EdgeType, Tensor] = data.collect(
            "edge_attr", allow_empty=True
        )
        if isinstance(data, Batch):
            batch_dict = {k: data[k].batch for k in x_dict}
        else:
            batch_dict = None

        for k, v in x_dict.items():
            x_dict[k] = self.init_project[k](v)

        for layer in self.backbone:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            for key in x_dict:
                x_dict[key] = F.tanh(x_dict[key])

        global_dict, graph_feature = self.mix(x_dict, data.global_attr, batch_dict)

        return x_dict, global_dict, graph_feature

    __call__: Callable[[HeteroGraph | Batch], StateEmbedding]


class ActionEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        node_channels: tuple[int, int, int],
        stack_num: tuple[int, int, int],
        hidden_channels: tuple[int, int, int],
    ):
        super().__init__()

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
            ResidualLinear(hidden_channels[0], out_channels),
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
            ResidualLinear(hidden_channels[1], out_channels),
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
            ResidualLinear(hidden_channels[2], out_channels),
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

    def forward(
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

    __call__: Callable[
        [
            list[list[Action]],
            dict[str, Tensor],
            dict[str, dict[int, int]],
            list[IdIdxMapper],
        ],
        list[Tensor],
    ]

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
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, 4),
        )

        self.pick_decoder = nn.Sequential(
            nn.Linear(in_channels, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
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
            nn.Linear(in_channels, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(
                in_channels * 5,
                (node_channels[1] + node_channels[2]),
            ),
        )

        self.move_decoder = nn.Sequential(
            nn.Linear(in_channels, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
            nn.Tanh(),
            nn.Linear(in_channels * 5, in_channels * 5),
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

        self.model = nn.Sequential(
            ResidualLinear(state_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                )
                for _ in range(stack_num)
            ],
            ResidualLinear(hidden_channels, 1),
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
            ResidualLinear(hidden_channels, 1),
            nn.Flatten(-2),
        )

    def forward(self, state: Tensor, actions: Tensor):
        if state.dim() < actions.dim():
            s = list(state.shape)
            s.insert(-1, actions.size(-2))
            state = state.unsqueeze(-2).expand(s)
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
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.Tanh(),
                )
                for _ in range(stack_num)
            ],
            ResidualLinear(hidden_channels, state_channels),
        )

    def forward(self, state: Tensor, action: Tensor):
        if state.dim() < action.dim():
            s = list(state.shape)
            s.insert(-1, action.size(-2))
            state = state.unsqueeze(-2).expand(s)
        return self.model(torch.cat([state, action], -1))

    __call__: Callable[[Tensor, Tensor], Tensor]


class ValuePrefixNet(nn.Module):
    def __init__(self, state_channels: int, hidden_size: int, layer_num: int):
        super().__init__()

        self.prev_proj = nn.Sequential(
            nn.Linear(state_channels, state_channels),
            # Rearrange("n l s -> n s l"),
            # nn.BatchNorm1d(state_channels),
            # Rearrange("n s l -> n l s"),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(state_channels, hidden_size, layer_num, batch_first=True)
        self.succ_proj = nn.Sequential(
            nn.Linear(hidden_size, state_channels),
            # Rearrange("n l s -> n s l"),
            # nn.BatchNorm1d(state_channels),
            # Rearrange("n s l -> n l s"),
            nn.Tanh(),
            nn.Linear(state_channels, 1),
            Rearrange("n l 1 -> n l"),
        )

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor] | None):
        # N L S
        x = self.prev_proj(x)
        x, hx = self.lstm(x, hx)
        x = self.succ_proj(x)
        return x, hx

    __call__: Callable[
        [Tensor, tuple[Tensor, Tensor] | None], tuple[Tensor, tuple[Tensor, Tensor]]
    ]


# class ActionGenerator(nn.Module):
#     def __init__(
#         self,
#         state_channels: int,
#         action_channels: int,
#         out_num: int,
#         gen_hidden_channels: int,
#         gen_stack_num: int,
#         dis_hidden_channels: int,
#         dis_stack_num: int,
#         cls_emb_dim: int,
#         cls_hidden_channels: int,
#         cls_stack_num: int,
#     ):
#         super().__init__()

#         self.action_channels = action_channels
#         self.out_num = out_num

#         base_channels = state_channels + action_channels

#         self.embeddings = nn.Embedding(out_num, cls_emb_dim)

#         self.generator = nn.Sequential(
#             nn.Linear(state_channels + cls_emb_dim, gen_hidden_channels),
#             nn.BatchNorm1d(gen_hidden_channels),
#             nn.Tanh(),
#             *[
#                 nn.Sequential(
#                     nn.Linear(gen_hidden_channels, gen_hidden_channels),
#                     nn.BatchNorm1d(gen_hidden_channels),
#                     nn.Tanh(),
#                 )
#                 for _ in range(gen_stack_num)
#             ],
#             nn.Linear(gen_hidden_channels, action_channels),
#         )

#         self.discriminator = nn.Sequential(
#             nn.Linear(base_channels, dis_hidden_channels),
#             nn.BatchNorm1d(dis_hidden_channels),
#             nn.Tanh(),
#             *[
#                 nn.Sequential(
#                     nn.Linear(dis_hidden_channels, dis_hidden_channels),
#                     nn.BatchNorm1d(dis_hidden_channels),
#                     nn.Tanh(),
#                 )
#                 for _ in range(dis_stack_num)
#             ],
#             nn.Linear(dis_hidden_channels, 1),
#             nn.Sigmoid(),
#             nn.Flatten(-2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(base_channels, cls_hidden_channels),
#             nn.BatchNorm1d(cls_hidden_channels),
#             nn.Tanh(),
#             *[
#                 nn.Sequential(
#                     nn.Linear(cls_hidden_channels, cls_hidden_channels),
#                     nn.BatchNorm1d(cls_hidden_channels),
#                     nn.Tanh(),
#                 )
#                 for _ in range(cls_stack_num)
#             ],
#             nn.Linear(cls_hidden_channels, out_num),
#         )

#         self.eval_generator = nn.Sequential(
#             Rearrange("n o s -> (n o) s"),
#             self.generator,
#             Rearrange("(n o) s -> n o s", o=self.out_num),
#         )

#     def train_generate(self, states: Tensor) -> tuple[Tensor, Tensor]:
#         class_target = torch.randint(
#             self.out_num, (states.size(0),), device=states.device
#         )
#         class_embeddings = self.embeddings(class_target)
#         return self.generator(torch.cat([states, class_embeddings], -1)), class_target

#     def discriminate(self, states: Tensor, actions: Tensor):
#         return self.discriminator(torch.cat([states, actions], -1))

#     def classify(self, states: Tensor, actions: Tensor):
#         return self.classifier(torch.cat([states, actions], -1))

#     @torch.no_grad
#     def generate(self, states: Tensor) -> Tensor:
#         class_target = torch.randint(
#             self.out_num, (states.size(0), self.out_num), device=states.device
#         )
#         class_embeddings = self.embeddings(class_target)
#         generated: Tensor = self.eval_generator(
#             torch.cat(
#                 [
#                     states.unsqueeze(1).expand(-1, self.out_num, -1),
#                     class_embeddings,
#                 ],
#                 -1,
#             )
#         )
#         return generated


class ActionGenerator(nn.Module):
    def __init__(
        self,
        state_channels: int,
        action_channels: int,
        out_num: int,
        gen_hidden_channels: int,
        gen_stack_num: int,
        dis_hidden_channels: int,
        dis_stack_num: int,
        cls_emb_dim: int,
        cls_hidden_channels: int,
        cls_stack_num: int,
    ):
        super().__init__()

        self.action_channels = action_channels
        self.out_num = out_num
        self.emb_dim = cls_emb_dim

        base_channels = state_channels + action_channels

        self.generator = nn.Sequential(
            ResidualLinear(state_channels + cls_emb_dim, gen_hidden_channels),
            # nn.BatchNorm1d(gen_hidden_channels),
            nn.Tanh(),
            *[
                nn.Sequential(
                    ResidualLinear(gen_hidden_channels, gen_hidden_channels),
                    # nn.BatchNorm1d(gen_hidden_channels),
                    nn.Tanh(),
                )
                for _ in range(gen_stack_num)
            ],
            ResidualLinear(gen_hidden_channels, action_channels),
        )

        self.discriminator = nn.Sequential(
            ResidualLinear(base_channels, dis_hidden_channels),
            # nn.BatchNorm1d(dis_hidden_channels),
            nn.LeakyReLU(),
            *[
                nn.Sequential(
                    ResidualLinear(dis_hidden_channels, dis_hidden_channels),
                    # nn.BatchNorm1d(dis_hidden_channels),
                    nn.LeakyReLU(),
                )
                for _ in range(dis_stack_num)
            ],
            ResidualLinear(dis_hidden_channels, 1),
            nn.Flatten(-2),
        )

    def train_generate(self, states: Tensor) -> Tensor:
        noise = torch.rand([*states.shape[:-1], self.emb_dim], device=states.device)
        return self.generator(torch.cat([states, noise], -1))

    def discriminate(self, states: Tensor, actions: Tensor):
        return self.discriminator(torch.cat([states, actions], -1))

    @torch.no_grad
    def generate(self, states: Tensor) -> Tensor:
        noise = torch.rand(
            states.size(0) * self.out_num, self.emb_dim, device=states.device
        )
        generated: Tensor = self.generator(
            torch.cat(
                [
                    repeat(states, "b s -> (b o) s", o=self.out_num),
                    noise,
                ],
                -1,
            )
        )
        return rearrange(generated, "(b o) s -> b o s", o=self.out_num)


class Node:
    def __init__(self, parent: Self | None, index: int, logit: float):
        self.parent = parent
        self.index = index
        self.logit = logit
        self.children: list[Self] = []
        self.value_list: list[float] = []
        self.visit_count = 0
        self.depth = 0 if parent is None else parent.depth + 1

    def expand(
        self,
        state: Tensor,
        actions: Tensor,
        value_prefix: float,
        hidden: tuple[Tensor, Tensor],
        logits: Tensor,
    ):
        self.state = state
        self.actions = actions
        self.value_prefix = value_prefix
        self.hidden = hidden
        for i in range(actions.size(0)):
            self.children.append(Node(self, i, logits[i].item()))

    def logits(self):
        return [child.logit for child in self.children]

    def expanded(self):
        return len(self.children) > 0

    def reward(self):
        if self.parent is not None:
            return (self.value_prefix - self.parent.value_prefix) / (
                Agent.discount**self.depth
            )
        return self.value_prefix

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
                    child.reward() + Agent.discount * child.value()
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
                (child.reward() + Agent.discount * child.value())
                if child.expanded()
                else v_mix
            )
            for child in self.children
        ]
        completed_Qs = np.array(completed_Qs)
        child_visit_counts = np.array([child.visit_count for child in self.children])
        max_child_visit_count = child_visit_counts.max()
        return tensor((50 + max_child_visit_count) * 0.1 * completed_Qs)

    def improved_policy(self):
        logits = tensor(self.logits())
        # if logits.numel() > 1:
        #     logits = (logits - logits.mean()) / logits.std()
        # else:
        #     logits = logits - logits.mean()
        sigma_Qs = self.sigma_Qs()
        # if not (sigma_Qs == sigma_Qs[0]).all():
        #     sigma_Qs = (sigma_Qs - sigma_Qs.mean()) / sigma_Qs.std()
        # else:
        #     sigma_Qs = sigma_Qs - sigma_Qs.mean()
        return F.softmax(logits + sigma_Qs, 0)

    def select_action(self):
        child_visit_counts = torch.tensor(
            [child.visit_count for child in self.children]
        )
        idx: int = torch.argmax(
            self.improved_policy() - child_visit_counts / (1 + child_visit_counts.sum())
        ).item()
        return self.children[idx]

    def select_root_action(self, available_index: list[int]):
        min_visit_count = self.visit_count + 1
        min_visit_idx = -1
        for idx in available_index:
            if not self.children[idx].expanded():
                return self.children[idx]
            if (c := self.children[idx].visit_count) < min_visit_count:
                min_visit_count = c
                min_visit_idx = idx
        return self.children[min_visit_idx]


class MCTS:
    def __init__(
        self,
        policy_net: PolicyNet,
        value_net: ValueNet,
        predict_net: PredictNet,
        generate_net: ActionGenerator,
        value_prefix_net: ValuePrefixNet,
    ):
        self.device: torch.device | None = None

        self.policy_net = policy_net
        self.value_net = value_net
        self.predict_net = predict_net
        self.generate_net = generate_net
        self.value_prefix_net = value_prefix_net

    @torch.no_grad()
    def search(
        self,
        states: Tensor,
        action_lists: list[Tensor],
        root_sample_count: int,
        simulation_count: int,
    ):
        self.device = states.device
        assert root_sample_count & (root_sample_count - 1) == 0  # 2的幂

        roots: list[Node] = []
        for state, actions, value, hidden in zip(
            states,
            action_lists,
            self.value_net(states),
            zip(
                *[
                    torch.unbind(h, 1)
                    for h in self.value_prefix_net(states.unsqueeze(1), None)[1]
                ]
            ),
        ):
            logits = self.policy_net(state, actions)
            new_root = Node(None, -1, 1)
            new_root.visit_count += 1
            new_root.value_list.append(value.item())
            new_root.expand(state, actions, 0, hidden, logits)
            roots.append(new_root)
        gs: list[Tensor] = []
        remaining_actions_index: list[list[int]] = []
        sim_SH_idxs: list[list[int]] = []
        sim_stages: list[int] = []
        for i, actions in enumerate(action_lists):
            gumbel = torch.distributions.Gumbel(0, 1)
            g: Tensor = gumbel.sample([actions.size(0)]).to(self.device)
            gs.append(g)
            logits = tensor(roots[i].logits(), device=self.device)
            begin_count = min(root_sample_count, actions.size(0))
            sim_SH_idxs.append(
                self.SH_idx(begin_count, root_sample_count, simulation_count)
            )
            sim_stages.append(0)
            topk_indices = torch.topk(g + logits, begin_count).indices
            remaining_actions_index.append(topk_indices.tolist())

        # remaining_action_count = root_sample_count
        # phase_finish_idx = self.phase_step_num(
        #     root_sample_count,
        #     remaining_action_count,
        #     simulation_count,
        # )
        for sim_idx in count():
            target_leaves: list[Node] = []
            target_actions = []
            parent_states = []
            hiddens = ([], [])
            sim_state_idx = []
            for i, (root, actions_index, counts, stage) in enumerate(
                zip(roots, remaining_actions_index, sim_SH_idxs, sim_stages)
            ):
                if stage == len(counts):
                    assert len(actions_index) == 1
                    continue
                sim_state_idx.append(i)
                node = root.select_root_action(actions_index)
                while node.expanded():
                    node = node.select_action()
                target_leaves.append(node)
                target_actions.append(node.parent.actions[node.index])
                parent_states.append(node.parent.state)
                hiddens[0].append(node.parent.hidden[0])
                hiddens[1].append(node.parent.hidden[1])

            if len(sim_state_idx) == 0:
                break

            states = self.predict_net(
                torch.stack(parent_states),
                torch.stack(target_actions),
            )
            values = self.value_net(states)
            value_prefixs, new_hiddens = self.value_prefix_net(
                states.unsqueeze(1),
                (torch.stack(hiddens[0], 1), torch.stack(hiddens[1], 1)),
            )
            actions = self.generate_net.generate(states)
            logit_lists = self.policy_net(states, actions)

            for node, (
                state,
                action,
                value_prefix,
                hidden_0,
                hidden_1,
                value,
                logits,
            ) in zip(
                target_leaves,
                zip(
                    states,
                    actions,
                    value_prefixs,
                    torch.unbind(new_hiddens[0], 1),
                    torch.unbind(new_hiddens[1], 1),
                    values,
                    logit_lists,
                ),
            ):
                node.expand(
                    state,
                    action,
                    value_prefix.item(),
                    (hidden_0, hidden_1),
                    logits,
                )
                while True:
                    node.visit_count += 1
                    node.value_list.append(value.item())
                    if (p := node.parent) is not None:
                        value = node.reward() + Agent.discount * value
                        node = p
                    else:
                        break

            for idx in sim_state_idx:
                if sim_SH_idxs[idx][sim_stages[idx]] == sim_idx:
                    remaining_actions_index[idx] = self.sequential_halving(
                        roots[idx], gs[idx], remaining_actions_index[idx]
                    )
                    sim_stages[idx] += 1

            # if sim_idx == phase_finish_idx:
            #     remaining_actions_index = self.sequential_halving(
            #         roots, gs, remaining_actions_index
            #     )
            #     remaining_action_count /= 2
            #     if remaining_action_count > 2:
            #         phase_finish_idx += self.phase_step_num(
            #             root_sample_count,
            #             remaining_action_count,
            #             simulation_count,
            #         )
            #     else:
            #         phase_finish_idx = simulation_count - 1

        assert all(len(idxs) == 1 for idxs in remaining_actions_index)
        target_values = tensor(
            [root.value() for root in roots], dtype=torch.float, device=self.device
        )
        target_policies = [
            root.improved_policy().float().to(self.device) for root in roots
        ]
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
        n = math.floor(m * old_n / old_m * math.log(m, old_m))
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
        logits = tensor(root.logits())[action_idx].to(self.device)
        sigma_Qs = root.sigma_Qs()[action_idx].to(self.device)
        remain_idx = torch.topk(
            g + logits + sigma_Qs, math.ceil(len(action_idx) / 2)
        ).indices
        return [action_idx[idx] for idx in remain_idx.tolist()]

    # def sequential_halving(
    #     self, roots: list[Node], gs: list[Tensor], actions_index: list[list[int]]
    # ):
    #     ret = []
    #     for root, all_g, idxs in zip(roots, gs, actions_index):
    #         g = all_g[idxs]
    #         logits = tensor(root.logits())[idxs].to(self.device)
    #         sigma_Qs = root.sigma_Qs()[idxs].to(self.device)
    #         remain_idx = torch.topk(
    #             g + logits + sigma_Qs, math.ceil(len(idxs) / 2)
    #         ).indices
    #         ret.append([idxs[idx] for idx in remain_idx.tolist()])
    #     return ret


class Agent(L.LightningModule):
    discount = 0.997

    class TrainStage(IntEnum):
        encode = auto()
        _value = auto()
        generate = auto()
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
        prefix_hidden_channel: int,
        prefix_stack_num: int,
        generate_count: int,
        gen_hidden_channels: int,
        gen_stack_num: int,
        dis_hidden_channels: int,
        dis_stack_num: int,
        cls_emb_dim: int,
        cls_hidden_channels: int,
        cls_stack_num: int,
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
        self.action_decoder = ActionDecoder(action_channels, node_channels)
        self.value_net = ValueNet(
            graph_channels,
            value_hidden_channel,
            value_stack_num,
        )
        self.value_target = ValueNet(
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
        self.policy_target = PolicyNet(
            graph_channels,
            action_channels,
            policy_hidden_channel,
            policy_stack_num,
        )
        self.predictor = PredictNet(
            graph_channels,
            action_channels,
            predict_hidden_channel,
            predict_stack_num,
        )
        self.value_prefix_net = ValuePrefixNet(
            graph_channels, prefix_hidden_channel, prefix_stack_num
        )
        self.action_generator = ActionGenerator(
            graph_channels,
            action_channels,
            generate_count,
            gen_hidden_channels,
            gen_stack_num,
            dis_hidden_channels,
            dis_stack_num,
            cls_emb_dim,
            cls_hidden_channels,
            cls_stack_num,
        )

        self.mcts = MCTS(
            self.policy_net,
            self.value_net,
            self.predictor,
            self.action_generator,
            self.value_prefix_net,
        )

        self.root_sample_count = root_sample_count
        self.simulation_count = simulation_count

        self.buffer = ReplayBuffer(seq_len, buffer_size)

        self.val_predict_num = val_predict_num

        self.stage = stage

    def save(self, path: Path, name: str):
        target_dir = path / self.stage.name / f"{name}.ckpt"
        state_dict = {}
        if self.stage >= Agent.TrainStage.encode:
            state_dict.update(
                {
                    "extractor": self.extractor.state_dict(),
                    "action_encoder": self.action_encoder.state_dict(),
                    "action_decoder": self.action_decoder.state_dict(),
                    "value_net": self.value_net.state_dict(),
                    "value_prefix_net": self.value_prefix_net.state_dict(),
                    "predictor": self.predictor.state_dict(),
                }
            )
        if self.stage >= Agent.TrainStage.generate:
            state_dict.update(
                {
                    "action_generator": self.action_generator.state_dict(),
                }
            )
        if self.stage >= Agent.TrainStage.policy:
            state_dict.update(
                {
                    "policy_net": self.policy_net.state_dict(),
                }
            )
        torch.save(state_dict, target_dir)

    def load(self, path: str, stages: list[TrainStage]):
        ckpt = torch.load(path)
        states: dict[str, Tensor] = ckpt["state_dict"]
        load_modules = []
        if Agent.TrainStage.encode in stages:
            load_modules.extend(
                [
                    "extractor",
                    "action_encoder",
                    "action_decoder",
                    "value_prefix_net",
                    "predictor",
                ]
            )
        if Agent.TrainStage._value in stages:
            load_modules.extend(["value_net"])
        if Agent.TrainStage.generate in stages:
            load_modules.extend(["action_generator"])
        if Agent.TrainStage.policy in stages:
            load_modules.extend(["policy_net"])

        for module_name in load_modules:
            getattr(self, module_name).load_state_dict(
                {
                    k[len(module_name) + 1 :]: v
                    for (k, v) in states.items()
                    if k.startswith(f"{module_name}.")
                }
            )

    def compile_modules(self):
        torch.compile(self.extractor, fullgraph=True)
        torch.compile(self.action_encoder, fullgraph=True)
        torch.compile(self.action_decoder, fullgraph=True)
        torch.compile(self.value_net, fullgraph=True)
        torch.compile(self.value_target, fullgraph=True)
        torch.compile(self.policy_net, fullgraph=True)
        torch.compile(self.policy_target, fullgraph=True)
        torch.compile(self.predictor, fullgraph=True)
        torch.compile(self.value_prefix_net, fullgraph=True)
        torch.compile(self.action_generator, fullgraph=True)

    def prepare_data(self):
        assert self.envs is not None
        self.init_buffer()

        self.baseline, self.val_data = self.get_baseline()

    def get_baseline(self) -> tuple[np.ndarray | None, list[Graph]]:
        if self.stage == Agent.TrainStage.encode:
            return None, [None]
        if self.stage == Agent.TrainStage._value:
            return None, [
                Environment(1, [params], False).envs[0]
                for params in self.envs.generate_params
            ]
        if self.stage == Agent.TrainStage.generate:
            return None, [
                Environment(1, [params], False).envs[0]
                for params in self.envs.generate_params
            ]
        if self.stage >= Agent.TrainStage.policy:
            temp_envs = [
                Environment(1, [params], False) for params in self.envs.generate_params
            ]
            timestamps: list[float] = []
            graphs: list[Graph] = []
            for env_i, env in enumerate(temp_envs):
                env_timestamps: list[float] = []
                env.reset()
                graph = env.envs[0]
                for _ in tqdm(
                    range(self.val_predict_num), f"Baseline_{env_i}", leave=False
                ):
                    env.reset([graph])
                    while True:
                        obs = env.observe()
                        act, _ = single_step_useful_only_predict(obs[0])
                        _, done, _ = env.step([act])
                        if done[0]:
                            break
                    env_timestamps.append(env.envs[0].get_timestamp())
                timestamps.append(np.mean(env_timestamps))
                graphs.append(graph)
            return np.array(timestamps), graphs

    def init_buffer(self):
        progress = tqdm(total=self.epoch_size, desc="Init Buffer")
        while True:
            obs = self.envs.observe()
            if self.stage == Agent.TrainStage.explore:
                acts, act_idxs = self.single_step_predict(
                    obs,
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
            rewards, dones, next_states = self.envs.step(acts)
            prev_buffer_size = len(self.buffer.buffer)
            self.buffer.append(
                obs,
                act_idxs,
                rewards,
                dones,
                next_states,
            )
            buffer_size = len(self.buffer.buffer)
            progress.update(buffer_size - prev_buffer_size)
            if progress.n >= progress.total:
                break

    def configure_optimizers(self):
        encode_stage_opt = optim.Adam(
            chain(
                self.extractor.parameters(),
                self.action_encoder.parameters(),
                self.action_decoder.parameters(),
                self.value_prefix_net.parameters(),
                self.predictor.parameters(),
            ),
            self.lr,
        )
        encode_stage_sch = lr_scheduler.StepLR(
            encode_stage_opt,
            self.opt_step_size,
            0.99,
        )

        value_stage_opt = optim.SGD(
            self.value_net.parameters(),
            self.lr,
        )
        value_stage_sch = lr_scheduler.StepLR(
            value_stage_opt,
            self.opt_step_size,
            0.99,
        )

        discriminator_opt = optim.Adam(
            self.action_generator.discriminator.parameters(),
            self.lr,
            (0, 0.9),
            weight_decay=1e-3,
        )
        discriminator_sch = lr_scheduler.StepLR(
            discriminator_opt,
            self.opt_step_size,
            0.99,
        )

        generator_opt = optim.Adam(
            self.action_generator.generator.parameters(),
            self.lr,
            (0, 0.9),
            weight_decay=1e-3,
        )
        generator_sch = lr_scheduler.StepLR(
            generator_opt,
            self.opt_step_size,
            0.99,
        )

        policy_opt = optim.SGD(
            self.policy_net.parameters(),
            self.lr,
        )
        policy_sch = lr_scheduler.StepLR(
            policy_opt,
            self.opt_step_size,
            0.9,
        )

        explore_opt = optim.Adam(
            chain(self.value_net.parameters(), self.policy_net.parameters()),
            self.lr * 1e-2,
        )
        explore_sch = lr_scheduler.StepLR(
            explore_opt,
            0.9,
        )

        return [
            encode_stage_opt,
            value_stage_opt,
            discriminator_opt,
            generator_opt,
            policy_opt,
            explore_opt,
        ], [
            encode_stage_sch,
            value_stage_sch,
            discriminator_sch,
            generator_sch,
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
            case Agent.TrainStage.encode:
                self.value_target.load_state_dict(self.value_net.state_dict())
            case Agent.TrainStage.policy:
                self.value_target.load_state_dict(self.value_net.state_dict())
                self.policy_target.load_state_dict(self.policy_net.state_dict())

    @torch.no_grad
    def play_step(self):
        for _ in range(5):
            obs = self.envs.observe()
            if self.stage == Agent.TrainStage.explore:
                acts, act_idxs = self.single_step_predict(
                    obs,
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
            rewards, dones, next_states = self.envs.step(acts)
            self.buffer.append(
                obs,
                act_idxs,
                rewards,
                dones,
                next_states,
            )

    def encode_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):
        graphs = [
            build_graph(obs.feature) for obs in sum([item.states for item in items], [])
        ]
        batched_graph = Batch.from_data_list(graphs).to(self.device)

        next_graphs = [
            build_graph(obs.feature)
            for obs in sum([item.next_states for item in items], [])
        ]
        batched_next_graph = Batch.from_data_list(next_graphs).to(self.device)

        (
            node_states,
            _,
            graph_states,
        ) = self.extractor(batched_graph)
        graph_states = rearrange(
            graph_states,
            "(n l) s -> n l s",
            l=self.seq_len,
        )
        with torch.no_grad():
            next_graph_states = rearrange(
                self.extractor(batched_next_graph)[2],
                "(n l) s -> n l s",
                l=self.seq_len,
            )

        offsets = get_offsets(batched_graph)
        mappers = sum(
            [[state.mapper for state in item.states] for item in items],
            [],
        )
        actions = sum(
            [[state.action_list for state in item.states] for item in items],
            [],
        )
        types = [[action.action_type for action in actions] for actions in actions]
        raw_actions_tensors = self.action_encoder.action_tensor_cat(
            actions,
            node_states,
            offsets,
            mappers,
        )
        actions_embs = self.action_encoder.encode(
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

        target_value_prefix = []
        current_prefix = torch.zeros(
            next_graph_states.size(0), dtype=torch.float, device=self.device
        )
        rewards = tensor([item.rewards for item in items], device=self.device)
        for i in range(self.seq_len):
            current_prefix = current_prefix + rewards[:, i] * Agent.discount**i
            target_value_prefix.append(current_prefix)
        target_value_prefix = torch.stack(target_value_prefix, 1)

        pred_value_prefix = self.value_prefix_net(
            torch.cat([graph_states, next_graph_states[:, 0:1]], 1), None
        )[0][:, 1:]

        value_prefix_loss = F.mse_loss(pred_value_prefix, target_value_prefix)

        pred_next_states = []
        prev_states = graph_states[:, 0]
        for pred_step in range(self.seq_len):
            action_embs = torch.stack(
                [
                    actions_embs[i * self.seq_len + pred_step][
                        item.action_idxs[pred_step]
                    ]
                    for i, item in enumerate(items)
                ]
            )
            new_pred_next_states = self.predictor(prev_states, action_embs)
            pred_next_states.append(new_pred_next_states)
            prev_states = new_pred_next_states
        pred_next_states = torch.stack(pred_next_states, 1)

        self.log_dict(
            {
                "info/next_state_scale": next_graph_states.abs().mean(),
                "info/pred_state_scale": pred_next_states.abs().mean(),
            }
        )

        pred_state_sim_losses = -torch.stack(
            [
                F.cosine_similarity(p, n, -1).mean()
                for p, n in zip(
                    torch.unbind(pred_next_states, 1),
                    torch.unbind(next_graph_states, 1),
                )
            ]
        )

        pred_state_sim_loss = torch.mean(pred_state_sim_losses)

        self.log_dict(
            {f"info/pred_sim_{i}": v for i, v in enumerate(pred_state_sim_losses)}
        )

        pred_state_mse_losses = torch.stack(
            [
                F.mse_loss(p, n)
                for p, n in zip(
                    torch.unbind(pred_next_states, 1),
                    torch.unbind(next_graph_states, 1),
                )
            ]
        )
        self.log_dict(
            {f"info/pred_mse_{i}": v for i, v in enumerate(pred_state_mse_losses)}
        )

        opt.zero_grad()
        self.manual_backward(
            cls_loss
            + decode_loss
            + type_diff_loss
            + action_diff_loss
            + value_prefix_loss
            + pred_state_sim_loss
        )
        clip_grad_norm_(self.extractor.parameters(), 0.5)
        clip_grad_norm_(self.action_encoder.parameters(), 0.5)
        clip_grad_norm_(self.action_decoder.parameters(), 0.5)
        clip_grad_norm_(self.value_net.parameters(), 0.5)
        clip_grad_norm_(self.value_prefix_net.parameters(), 0.5)
        clip_grad_norm_(self.predictor.parameters(), 0.5)
        opt.step()

        self.log_dict(
            {
                "loss/cls": cls_loss,
                "loss/decode": decode_loss,
                "loss/type_diff": type_diff_loss,
                "loss/action_diff": action_diff_loss,
                "loss/value_prefix": value_prefix_loss,
                "loss/pred_state_sim": pred_state_sim_loss,
            }
        )

    def value_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):
        self.eval()
        with torch.no_grad():
            graphs = [
                build_graph(obs.feature)
                for obs in sum([item.states for item in items], [])
            ]
            batched_graph = Batch.from_data_list(graphs).to(self.device)

            next_graphs = [
                build_graph(obs.feature)
                for obs in sum([item.next_states for item in items], [])
            ]
            batched_next_graph = Batch.from_data_list(next_graphs).to(self.device)

            graph_states = self.extractor(batched_graph)[2]
            next_graph_states = self.extractor(batched_next_graph)[2]

            rewards = tensor(
                sum([item.rewards for item in items], []),
                device=self.device,
            )
            dones = tensor(
                sum([item.dones for item in items], []),
                dtype=torch.float,
                device=self.device,
            )
            next_value = self.value_target(next_graph_states)
        self.train()

        value_loss = F.mse_loss(
            self.value_net(graph_states),
            rewards + Agent.discount * next_value * (1 - dones),
        )
        self.log("loss/value", value_loss)

        opt.zero_grad()
        self.manual_backward(value_loss)
        clip_grad_norm_(self.value_net.parameters(), 0.5)
        opt.step()

    def generate_stage(
        self,
        items: list[SequenceReplayItem],
        dis_opt: LightningOptimizer,
        gen_opt: LightningOptimizer,
    ):
        self.eval()
        with torch.no_grad():
            graphs = [
                build_graph(obs.feature)
                for obs in sum([item.states for item in items], [])
            ]
            batched_graph = Batch.from_data_list(graphs).to(self.device)

            (
                node_states,
                _,
                graph_states,
            ) = self.extractor(batched_graph)

            offsets = get_offsets(batched_graph)
            mappers = sum(
                [[state.mapper for state in item.states] for item in items],
                [],
            )
            actions = sum(
                [[state.action_list for state in item.states] for item in items],
                [],
            )

            actions_embs = self.action_encoder(
                actions,
                node_states,
                offsets,
                mappers,
            )
        self.train()

        # wasserstein

        for _ in range(5):
            true_action_embs = torch.stack(
                [emb[np.random.randint(emb.size(0))] for emb in actions_embs]
            )

            with torch.no_grad():
                gen_action_embs = self.action_generator.train_generate(graph_states)

            e = torch.rand(true_action_embs.size(0), 1).to(self.device)
            mixed_action_embs = (
                true_action_embs * e + gen_action_embs * (1 - e)
            ).requires_grad_()

            dis_true, dis_gen, dis_mix = torch.chunk(
                self.action_generator.discriminate(
                    repeat(graph_states, "b s -> (r b) s", r=3),
                    torch.cat([true_action_embs, gen_action_embs, mixed_action_embs]),
                ),
                3,
            )

            grads = torch.autograd.grad(
                dis_mix,
                mixed_action_embs,
                torch.ones_like(dis_mix),
                True,
                True,
            )[0]

            n = torch.norm(grads, 2, 1)
            gp = (n - 0.5) ** 2

            d_loss = torch.mean(dis_gen - dis_true + 80 * gp)

            dis_opt.zero_grad()
            self.manual_backward(d_loss)
            # clip_grad_norm_(self.action_generator.discriminator.parameters(), 0.5)
            dis_opt.step()

            self.log_dict(
                {
                    "info/norm_min": n.min(),
                    "info/norm_max": n.max(),
                }
            )

            self.log_dict(
                {
                    "loss/dis_gen": torch.mean(dis_gen),
                    "loss/dis_true": torch.mean(dis_true),
                    "loss/gp": torch.mean(gp),
                    "loss/critic": d_loss,
                }
            )

        gens = self.action_generator.train_generate(graph_states)

        g_loss = -torch.mean(self.action_generator.discriminate(graph_states, gens))
        self.log("loss/gen", g_loss)

        g_sim = torch.mean(
            torch.stack(
                [
                    torch.min(((gen - true) ** 2).mean(-1))
                    for gen, true in zip(gens, actions_embs)
                ]
            )
        )
        self.log("info/g_sim", g_sim)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.eval()
        with torch.no_grad():
            true_values = [
                self.value_net(self.predictor(state, actions))
                for state, actions in zip(graph_states, actions_embs)
            ]
            gen_values = self.value_net(self.predictor(graph_states, gens))
        self.train()

        value_sim = torch.mean(
            torch.stack(
                [
                    torch.min((gen - true) ** 2)
                    for gen, true in zip(gen_values, true_values)
                ]
            )
        )
        self.log("info/value_sim", value_sim)

    def policy_stage(
        self,
        items: list[SequenceReplayItem],
        v_opt: LightningOptimizer,
        p_opt: LightningOptimizer,
    ):
        self.eval()
        with torch.no_grad():
            graphs = [
                build_graph(obs.feature)
                for obs in sum([item.states for item in items], [])
            ]
            batched_graph = Batch.from_data_list(graphs).to(self.device)

            (
                node_states,
                _,
                graph_states,
            ) = self.extractor(batched_graph)

            offsets = get_offsets(batched_graph)
            mappers = sum(
                [[state.mapper for state in item.states] for item in items],
                [],
            )
            actions = sum(
                [[state.action_list for state in item.states] for item in items],
                [],
            )

            actions_embs = self.action_encoder(
                actions,
                node_states,
                offsets,
                mappers,
            )
            self.mcts.value_net = self.value_target
            self.mcts.policy_net = self.policy_target
            (
                _,
                target_value,
                target_policy,
            ) = self.mcts.search(
                graph_states,
                actions_embs,
                self.root_sample_count,
                self.simulation_count,
            )
            self.mcts.value_net = self.value_net
            self.mcts.policy_net = self.policy_net
        self.train()

        pred_value = self.value_net(graph_states)
        pred_policy = []
        policy_logits = []
        for state, embs in zip(graph_states, actions_embs):
            logits = self.policy_net(state, embs)
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
                    F.cosine_similarity(pred, target, dim=0)
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

        v_opt.zero_grad()
        self.manual_backward(value_loss)
        v_opt.step()

        p_opt.zero_grad()
        self.manual_backward(policy_loss)
        p_opt.step()

    def explore_stage(self, items: list[SequenceReplayItem], opt: LightningOptimizer):
        self.eval()
        with torch.no_grad():
            graphs = [
                build_graph(obs.feature)
                for obs in sum([item.states for item in items], [])
            ]
            batched_graph = Batch.from_data_list(graphs).to(self.device)

            (
                node_states,
                _,
                graph_states,
            ) = self.extractor(batched_graph)

            offsets = get_offsets(batched_graph)
            mappers = sum(
                [[state.mapper for state in item.states] for item in items],
                [],
            )
            actions = sum(
                [[state.action_list for state in item.states] for item in items],
                [],
            )

            actions_embs = self.action_encoder(
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
                graph_states,
                actions_embs,
                self.root_sample_count,
                self.simulation_count,
            )
        self.train()

        pred_value = self.value_net(graph_states)
        pred_policy = []
        policy_logits = []
        for state, embs in zip(graph_states, actions_embs):
            logits = self.policy_net(state, embs)
            pred_policy.append(logits.softmax(-1))
            policy_logits.append(logits)
        policy_logits = torch.cat(policy_logits)
        self.log("info/policy_logits_min_var", policy_logits.var(-1).min())

        value_loss = F.mse_loss(pred_value, target_value)
        policy_loss = torch.mean(
            torch.stack(
                [
                    F.kl_div(torch.log(pred), target, reduction="batchmean")
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
        self.manual_backward(value_loss + policy_loss)
        clip_grad_norm_(self.value_net.parameters(), 0.5)
        clip_grad_norm_(self.policy_net.parameters(), 0.5)
        opt.step()

    def training_step(self, items: list[SequenceReplayItem]):
        (
            encode_stage_opt,
            value_stage_opt,
            discriminator_opt,
            generator_opt,
            policy_opt,
            explore_opt,
        ) = self.optimizers()
        (
            encode_stage_sch,
            value_stage_sch,
            discriminator_sch,
            generator_sch,
            policy_sch,
            explore_sch,
        ) = self.lr_schedulers()
        match self.stage:
            case Agent.TrainStage.encode:
                self.encode_stage(items, encode_stage_opt)
                encode_stage_sch.step()
            case Agent.TrainStage._value:
                self.value_stage(items, value_stage_opt)
                value_stage_sch.step()
                pass
            case Agent.TrainStage.generate:
                self.generate_stage(items, discriminator_opt, generator_opt)
                # discriminator_sch.step()
                # generator_sch.step()
            case Agent.TrainStage.policy:
                self.policy_stage(items, value_stage_opt, policy_opt)
            case Agent.TrainStage.explore:
                self.explore_stage(items, explore_opt)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.eval()
        self.play_step()
        self.train()

    def on_train_epoch_end(self):
        match self.stage:
            case Agent.TrainStage._value:
                if (self.current_epoch + 1) % 3 == 0:
                    self.value_target.load_state_dict(self.value_net.state_dict())
            case Agent.TrainStage.policy:
                if (self.current_epoch + 1) % 3 == 0:
                    self.value_target.load_state_dict(self.value_net.state_dict())
                    self.policy_target.load_state_dict(self.policy_net.state_dict())

    @torch.no_grad
    def single_step_predict(
        self, obs: list[Observation], sample_count: int, sim_count: int
    ):
        batched_graph = Batch.from_data_list(
            [build_graph(ob.feature) for ob in obs]
        ).to(self.device)

        (
            node_states,
            _,
            graph_states,
        ) = self.extractor(batched_graph)

        offsets = get_offsets(batched_graph)
        mappers = [ob.mapper for ob in obs]
        action_lists = [ob.action_list for ob in obs]

        action_embs = self.action_encoder(
            action_lists,
            node_states,
            offsets,
            mappers,
        )
        (
            act_idxs,
            _,
            _,
        ) = self.mcts.search(
            graph_states,
            action_embs,
            sample_count,
            sim_count,
        )
        return [ob.action_list[idx] for ob, idx in zip(obs, act_idxs)], act_idxs

    @torch.inference_mode()
    async def predict(
        self,
        graph: Graph,
        sample_count: int,
        sim_count: int,
    ):
        env = Environment.from_graphs([graph])

        for round_count in count(1):
            obs = env.observe()

            actions, _ = self.single_step_predict(
                obs,
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

    def validation_step(self, batch: list[Graph]):
        match self.stage:
            case Agent.TrainStage._value:
                obs = Environment.from_graphs(
                    [graph.copy() for graph in batch]
                ).observe()
                batched_graph = Batch.from_data_list(
                    [build_graph(ob.feature) for ob in obs]
                ).to(self.device)

                (
                    node_states,
                    type_states,
                    graph_states,
                ) = self.extractor(batched_graph)

                offsets = get_offsets(batched_graph)
                mappers = [ob.mapper for ob in obs]
                action_lists = [ob.action_list for ob in obs]
                action_embs = self.action_encoder(
                    action_lists,
                    node_states,
                    offsets,
                    mappers,
                )
                action_types = [
                    [action.action_type.value for action in ob.action_list]
                    for ob in obs
                ]
                stack_action_embs, ps = pack(action_embs, "* a")
                repeated_states, _ = pack(
                    [
                        repeat(states, "s -> a s", a=p[0])
                        for states, p in zip(graph_states, ps)
                    ],
                    "* s",
                )
                pred_next_states = self.predictor(repeated_states, stack_action_embs)
                pred_rewards = self.value_prefix_net(
                    torch.stack([repeated_states, pred_next_states], 1), None
                )[0][:, 1]
                values = self.value_net(pred_next_states)
                for i, (ts, rs, vs) in enumerate(
                    zip(
                        action_types,
                        unpack(pred_rewards, ps, "*"),
                        unpack(values, ps, "*"),
                    )
                ):
                    type_Qs: dict[int, list[float]] = {}
                    for t, r, v in zip(ts, rs, vs, strict=True):
                        type_Qs.setdefault(t, []).append(r.item() + v.item())
                    self.log_dict(
                        {
                            f"val/env_{i}_pick_Q": np.mean(
                                type_Qs[ActionType.pick.value]
                            ),
                            f"val/env_{i}_move_Q": np.mean(
                                type_Qs[ActionType.move.value]
                            ),
                        },
                        batch_size=len(batch),
                    )
            case Agent.TrainStage.generate:
                env = Environment.from_graphs([graph.copy() for graph in batch])
                for _ in range(np.random.randint(30)):
                    obs = env.observe()
                    acts = []
                    act_idxs = []
                    for ob in obs:
                        act, act_idx = single_step_useful_only_predict(ob, 0.1)
                        acts.append(act)
                        act_idxs.append(act_idx)
                    env.step(acts)

                obs = env.observe()
                batched_graph = Batch.from_data_list(
                    [build_graph(ob.feature) for ob in obs]
                ).to(self.device)
                (
                    node_states,
                    _,
                    graph_states,
                ) = self.extractor(batched_graph)
                offsets = get_offsets(batched_graph)
                mappers = [ob.mapper for ob in obs]
                action_lists = [ob.action_list for ob in obs]
                action_embs = self.action_encoder(
                    action_lists,
                    node_states,
                    offsets,
                    mappers,
                )
                stack_action_embs, ps = pack(action_embs, "* a")
                repeated_states, _ = pack(
                    [
                        repeat(states, "s -> a s", a=p[0])
                        for states, p in zip(graph_states, ps)
                    ],
                    "* s",
                )
                true_pred_next_states = self.predictor(
                    repeated_states, stack_action_embs
                )
                true_pred_rewards = self.value_prefix_net(
                    torch.stack([repeated_states, true_pred_next_states], 1), None
                )[0][:, 1]
                true_pred_values = self.value_net(true_pred_next_states)
                for i, t_Q in enumerate(
                    unpack(true_pred_rewards + true_pred_values, ps, "*"),
                ):
                    self.log(
                        f"val/env_{i}_true_Q",
                        torch.mean(t_Q),
                        batch_size=len(batch),
                    )

                gen_actions = rearrange(
                    self.action_generator.generate(graph_states), "b o s -> (b o) s"
                )
                repeated_states = repeat(
                    graph_states, "b s -> (b o) s", o=self.action_generator.out_num
                )
                gen_pred_next_states = self.predictor(repeated_states, gen_actions)
                gen_pred_rewards = self.value_prefix_net(
                    torch.stack([repeated_states, gen_pred_next_states], 1), None
                )[0][:, 1]
                gen_pred_values = self.value_net(gen_pred_next_states)
                for i, g_Q in enumerate(
                    torch.chunk(
                        gen_pred_rewards + gen_pred_values, graph_states.size(0)
                    )
                ):
                    self.log(
                        f"val/env_{i}_gen_Q",
                        torch.mean(g_Q),
                        batch_size=len(batch),
                    )

            case Agent.TrainStage.policy | Agent.TrainStage.explore:
                bar: tqdm = self.trainer.progress_bar_callback.val_progress_bar
                for sample_count, sim_count in [(1, 0), (4, 20), (8, 40)]:
                    bar.set_description(f"Validation [{sample_count}/{sim_count}]")
                    env = Environment.from_graphs(
                        [
                            graph.copy()
                            for graph in batch
                            for _ in range(self.val_predict_num)
                        ]
                    )
                    while any(
                        [
                            np.min([g.get_timestamp() for g in gs])
                            < self.baseline[i] * 2
                            for i, gs in enumerate(
                                batched(env.envs, self.val_predict_num)
                            )
                        ]
                    ):
                        obs = env.observe()
                        if len(obs) == 0:
                            break

                        actions, _ = self.single_step_predict(
                            obs,
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
