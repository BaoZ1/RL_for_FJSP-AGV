import torch
from torch import nn, Tensor, tensor
from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric import nn as gnn
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeType, EdgeType
from FJSP_env import (
    Graph,
    Environment,
    Action,
    ActionType,
    IdIdxMapper,
    Observation,
    single_step_simple_predict,
)
from .utils import *
import lightning as L
from lightning.pytorch.trainer.states import RunningStage
from collections.abc import Callable
from tqdm import tqdm
from typing import Self, AsyncGenerator, Any, Literal
import math
from itertools import chain, batched, count
from einops import rearrange
from einops.layers.torch import Rearrange
import asyncio


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


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.norm(x, 2, dim=-1, keepdim=True)


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

        for k, v in f1.items():
            if k in res:
                res[k] = res[k] + v
            else:
                res[k] = v

        for k, v in f2.items():
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
            nn.Linear(
                sum(global_channels) + Graph.global_feature_size,
                graph_global_channels * 2,
            ),
            nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            nn.Linear(graph_global_channels * 2, graph_global_channels * 2),
            nn.BatchNorm1d(graph_global_channels * 2),
            nn.Tanh(),
            nn.Linear(graph_global_channels * 2, graph_global_channels),
            nn.BatchNorm1d(graph_global_channels),
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
        global_dict = {k: self.type_norm[k](v) for k, v in global_dict.items()}

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
    ):
        super().__init__()

        self.wait_emb = nn.Parameter(torch.zeros(out_channels))
        self.pick_encoder = nn.Sequential(
            nn.Linear(
                (
                    node_channels[2]  # AGV
                    + node_channels[0]  # operation_from
                    + node_channels[0]  # operation_to
                    + node_channels[1]  # machine_target
                ),
                out_channels * 2,
            ),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.transport_encoder = nn.Sequential(
            nn.Linear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                out_channels * 2,
            ),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                out_channels * 2,
            ),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.Tanh(),
            nn.Linear(out_channels * 2, out_channels),
        )

    def forward(
        self,
        batch_actions: list[list[Action]],
        embeddings: dict[str, Tensor],
        offsets: dict[str, dict[int, int]],
        idx_mappers: list[IdIdxMapper],
    ):
        ret: list[Tensor] = []
        for i, (actions, idx_mapper) in enumerate(zip(batch_actions, idx_mappers)):
            encoded: list[Tensor] = []
            for action in actions:
                match action.action_type:
                    case ActionType.wait:
                        encoded.append(self.wait_emb)
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
                        encoded.append(
                            self.pick_encoder(
                                torch.cat(
                                    [
                                        AGV_emb,
                                        operation_from_emb,
                                        operation_to_emb,
                                        machine_target_emb,
                                    ]
                                )
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
                        encoded.append(
                            self.transport_encoder(
                                torch.cat(
                                    [
                                        AGV_emb,
                                        machine_target_emb,
                                    ]
                                )
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
                        encoded.append(
                            self.move_encoder(
                                torch.cat(
                                    [
                                        AGV_emb,
                                        machine_target_emb,
                                    ]
                                )
                            )
                        )
            ret.append(torch.stack(encoded))
        return ret

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


class ValueNet(nn.Module):
    def __init__(self, state_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_channels, state_channels * 2),
            nn.BatchNorm1d(state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels * 2),
            nn.BatchNorm1d(state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels * 2),
            nn.BatchNorm1d(state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels * 2),
            nn.BatchNorm1d(state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels * 2),
            nn.BatchNorm1d(state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, 1),
        )

    def forward(self, state: Tensor):
        return self.model(state).squeeze(-1)

    __call__: Callable[[Tensor], Tensor]


class PolicyNet(nn.Module):
    def __init__(self, state_channels: int, action_channels: int):
        super().__init__()

        base_channels = state_channels + action_channels
        self.model = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, 1),
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
    def __init__(self, state_channels: int, action_channels: int):
        super().__init__()
        base_channels = state_channels + action_channels
        self.model = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, state_channels),
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
            Rearrange("n l s -> n s l"),
            nn.BatchNorm1d(state_channels),
            Rearrange("n s l -> n l s"),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(state_channels, hidden_size, layer_num, batch_first=True)
        self.succ_proj = nn.Sequential(
            nn.Linear(hidden_size, state_channels),
            Rearrange("n l s -> n s l"),
            nn.BatchNorm1d(state_channels),
            Rearrange("n s l -> n l s"),
            nn.Tanh(),
            nn.Linear(state_channels, 1),
            Rearrange("n l 1 -> n l"),
        )

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor] | None = None):
        # N L S
        x = self.prev_proj(x)
        x, hx = self.lstm(x, hx)
        x = self.succ_proj(x)
        return x, hx

    __call__: Callable[
        [Tensor, tuple[Tensor, Tensor] | None], tuple[Tensor, tuple[Tensor, Tensor]]
    ]


class ActionGenerator(nn.Module):
    def __init__(self, state_channels: int, action_channels: int, out_num: int):
        super().__init__()

        self.action_channels = action_channels
        self.out_num = out_num

        base_channels = state_channels + action_channels

        embedding_dim = 8

        self.embeddings = nn.Embedding(out_num, embedding_dim)

        self.generator = nn.Sequential(
            nn.Linear(state_channels + embedding_dim, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.Tanh(),
            nn.Linear(base_channels, action_channels),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(base_channels, action_channels * 2),
            nn.BatchNorm1d(action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels * 2),
            nn.BatchNorm1d(action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels),
            nn.BatchNorm1d(action_channels),
            nn.Tanh(),
            nn.Linear(action_channels, 1),
            nn.Sigmoid(),
            nn.Flatten(-2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_channels, action_channels * 2),
            nn.BatchNorm1d(action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels * 2),
            nn.BatchNorm1d(action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels),
            nn.BatchNorm1d(action_channels),
            nn.Tanh(),
            nn.Linear(action_channels, out_num),
        )

        self.eval_generator = nn.Sequential(
            Rearrange("n o s -> (n o) s"),
            self.generator,
            Rearrange("(n o) s -> n o s", o=self.out_num),
        )

    def train_generate(self, states: Tensor) -> tuple[Tensor, Tensor]:
        class_target = torch.randint(
            self.out_num, (states.size(0),), device=states.device
        )
        class_embeddings = self.embeddings(class_target)
        return self.generator(torch.cat([states, class_embeddings], -1)), class_target

    def discriminate(self, states: Tensor, actions: Tensor):
        return self.discriminator(torch.cat([states, actions], -1))

    def classify(self, states: Tensor, actions: Tensor):
        return self.classifier(torch.cat([states, actions], -1))

    @torch.no_grad
    def generate(self, states: Tensor) -> Tensor:
        class_target = torch.randint(
            self.out_num, (states.size(0), self.out_num), device=states.device
        )
        class_embeddings = self.embeddings(class_target)
        generated: Tensor = self.eval_generator(
            torch.cat(
                [
                    states.unsqueeze(1).expand(-1, self.out_num, -1),
                    class_embeddings,
                ],
                -1,
            )
        )
        return generated


class Node:
    discount = 0.95

    def __init__(self, parent: Self | None, index: int, logit: float):
        self.parent = parent
        self.index = index
        self.logit = logit
        self.children: list[Self] = []
        self.value_list: list[float] = []
        self.visit_count = 0

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
            return self.value_prefix - self.parent.value_prefix
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
        return tensor((50 + max_child_visit_count) * 1.0 * completed_Qs)

    def improved_policy(self):
        return F.softmax(tensor(self.logits()) + self.sigma_Qs(), 0)

    def select_action(self):
        child_visit_counts = torch.tensor(
            [child.visit_count for child in self.children]
        )
        idx: int = torch.argmax(
            self.improved_policy() - child_visit_counts / (1 + child_visit_counts.max())
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
        init_hidden = (
            torch.zeros(
                self.value_prefix_net.lstm.num_layers,
                self.value_prefix_net.lstm.hidden_size,
                dtype=torch.float,
                device=self.device,
            ),
            torch.zeros(
                self.value_prefix_net.lstm.num_layers,
                self.value_prefix_net.lstm.hidden_size,
                dtype=torch.float,
                device=self.device,
            ),
        )
        for state, actions, value in zip(states, action_lists, self.value_net(states)):
            logits = self.policy_net(state, actions)
            new_root = Node(None, -1, 1)
            new_root.visit_count += 1
            new_root.value_list.append(value.item())
            new_root.expand(state, actions, 0, init_hidden, logits)
            roots.append(new_root)
        gs: list[Tensor] = []
        remaining_actions_index: list[list[int]] = []
        for i, actions in enumerate(action_lists):
            gumbel = torch.distributions.Gumbel(0, 1)
            g: Tensor = gumbel.sample([actions.size(0)]).to(self.device)
            gs.append(g)
            logits = tensor(roots[i].logits(), device=self.device)
            topk_indices = torch.topk(
                g + logits, min(root_sample_count, actions.size(0))
            ).indices
            remaining_actions_index.append(topk_indices.tolist())

        remaining_action_count = root_sample_count
        phase_finish_idx = self.phase_step_num(
            root_sample_count,
            remaining_action_count,
            simulation_count,
        )
        for sim_idx in range(simulation_count):
            target_leaves: list[Node] = []
            target_actions = []
            parent_states = []
            hiddens = ([], [])
            for root, actions_index in zip(roots, remaining_actions_index):
                node = root.select_root_action(actions_index)
                while node.expanded():
                    node = node.select_action()
                target_leaves.append(node)
                target_actions.append(node.parent.actions[node.index])
                parent_states.append(node.parent.state)
                hiddens[0].append(node.parent.hidden[0])
                hiddens[1].append(node.parent.hidden[1])
            target_actions = torch.stack(target_actions)
            parent_states = torch.stack(parent_states)
            hiddens = (
                torch.stack(hiddens[0], 1),
                torch.stack(hiddens[1], 1),
            )

            states = self.predict_net(parent_states, target_actions)
            values = self.value_net(states)
            value_prefixs, new_hiddens = self.value_prefix_net(
                states.unsqueeze(1),
                hiddens,
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
                    value = node.reward() + Agent.discount * value
                    if (p := node.parent) is not None:
                        node = p
                    else:
                        break

            if sim_idx == phase_finish_idx:
                remaining_actions_index = self.sequential_halving(
                    roots, gs, remaining_actions_index
                )
                remaining_action_count /= 2
                if remaining_action_count > 2:
                    phase_finish_idx += self.phase_step_num(
                        root_sample_count,
                        remaining_action_count,
                        simulation_count,
                    )
                else:
                    phase_finish_idx = simulation_count - 1

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

    def sequential_halving(
        self, roots: list[Node], gs: list[Tensor], actions_index: list[list[int]]
    ):
        ret = []
        for root, all_g, idxs in zip(roots, gs, actions_index):
            g = all_g[idxs]
            logits = tensor(root.logits())[idxs].to(self.device)
            sigma_Qs = root.sigma_Qs()[idxs].to(self.device)
            remain_idx = torch.topk(
                g + logits + sigma_Qs, math.ceil(len(idxs) / 2)
            ).indices
            ret.append([idxs[idx] for idx in remain_idx.tolist()])
        return ret


class Agent(L.LightningModule):
    discount = 0.997

    def __init__(
        self,
        envs: Environment | None,
        lr: float,
        seq_len: int,
        buffer_size: int,
        epoch_size: int,
        batch_size: int,
        val_num: int,
        node_channels: tuple[int, int, int],
        type_channels: tuple[int, int, int],
        graph_channels: int,
        state_layer_num: int,
        action_channels: int,
        generate_count: int,
        root_sample_count: int,
        simulation_count: int,
        val_predict_num: int,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["envs"])

        self.envs = envs

        self.lr = lr

        self.seq_len = seq_len

        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.val_num = val_num

        self.extractor = StateExtract(
            node_channels,
            type_channels,
            graph_channels,
            state_layer_num,
        )
        self.action_encoder = ActionEncoder(action_channels, node_channels)
        self.value_net = ValueNet(graph_channels)
        self.policy_net = PolicyNet(graph_channels, action_channels)
        self.predictor = PredictNet(graph_channels, action_channels)
        self.value_prefix_net = ValuePrefixNet(graph_channels, 512, 3)
        self.action_generator = ActionGenerator(
            graph_channels,
            action_channels,
            generate_count,
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

    def compile_modules(self):
        torch.compile(self.extractor, fullgraph=True)
        torch.compile(self.action_encoder, fullgraph=True)
        torch.compile(self.value_net, fullgraph=True)
        torch.compile(self.policy_net, fullgraph=True)
        torch.compile(self.predictor, fullgraph=True)
        torch.compile(self.value_prefix_net, fullgraph=True)

    def prepare_data(self):
        assert self.envs is not None
        self.baseline, self.val_graphs = self.get_baseline(self.val_num)
        self.init_buffer()

    def get_baseline(self, env_count) -> tuple[np.ndarray, list[Graph]]:
        temp_envs = [
            Environment(1, self.envs.generate_params, False) for _ in range(env_count)
        ]
        timestamps: list[float] = []
        graphs: list[Graph] = []
        for env_i, env in enumerate(temp_envs):
            env_timestamps: list[float] = []
            while True:
                env_timestamps.clear()
                env.reset()
                graph = env.envs[0]
                for _ in tqdm(range(8), f"Baseline_{env_i}", leave=False):
                    env.reset([graph])
                    while True:
                        obs = env.observe()
                        act, _ = single_step_simple_predict(obs[0])
                        _, done, _ = env.step([act])
                        if done[0]:
                            break

                    env_timestamps.append(env.envs[0].get_timestamp())
                if len(env_timestamps) > 5:
                    break
            timestamps.append(np.mean(env_timestamps))
            graphs.append(graph)
        return timestamps, graphs

    def init_buffer(self):
        progress = tqdm(total=self.epoch_size, desc="Init Buffer")
        while True:
            obs = self.envs.observe()
            acts = []
            act_idxs = []
            for ob in obs:
                act, act_idx = single_step_simple_predict(ob)
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
        encode_predict_opt = optim.Adam(
            chain(
                self.extractor.parameters(),
                self.action_encoder.parameters(),
                self.predictor.parameters(),
            ),
            self.lr,
        )
        encode_predict_sch = lr_scheduler.StepLR(encode_predict_opt, 5, 0.5)

        value_prefix_opt = optim.Adam(self.value_prefix_net.parameters(), self.lr)
        value_prefix_sch = lr_scheduler.StepLR(value_prefix_opt, 5, 0.8)

        discriminator_opt = optim.Adam(
            self.action_generator.discriminator.parameters(), self.lr
        )
        discriminator_sch = lr_scheduler.StepLR(discriminator_opt, 5, 0.8)

        generator_opt = optim.Adam(
            chain(
                self.action_generator.generator.parameters(),
                self.action_generator.classifier.parameters(),
            ),
            self.lr,
        )
        generator_sch = lr_scheduler.StepLR(generator_opt, 5, 0.8)

        actor_critic_opt = optim.SGD(
            chain(
                self.value_net.parameters(),
                self.policy_net.parameters(),
            ),
            self.lr,
        )
        actor_critic_sch = lr_scheduler.StepLR(actor_critic_opt, 5, 0.8)

        return [
            encode_predict_opt,
            value_prefix_opt,
            discriminator_opt,
            generator_opt,
            actor_critic_opt,
        ], [
            encode_predict_sch,
            value_prefix_sch,
            discriminator_sch,
            generator_sch,
            actor_critic_sch,
        ]

    def train_dataloader(self):
        return DataLoader(
            ReplayDataset(self.buffer, self.epoch_size),
            self.batch_size,
            collate_fn=lambda batch: batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_graphs,
            self.val_num,
            collate_fn=lambda batch: batch,
        )

    @torch.no_grad
    def play_step(self):
        for _ in range(5):
            obs = self.envs.observe()
            act, idx = self.single_step_predict(obs, 1, 0)
            reward, done, next_obs = self.envs.step(act)
            self.buffer.append(obs, idx, reward, done, next_obs)

    def encode_predict_part(self, items: list[SequenceReplayItem]):
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
        next_graph_states = self.extractor(batched_next_graph)[2]

        offsets = get_offsets(batched_graph)
        mappers = sum(
            [[state.mapper for state in item.states] for item in items],
            [],
        )
        actions = sum(
            [[state.action_list for state in item.states] for item in items],
            [],
        )
        actions_embs = self.action_encoder(actions, node_states, offsets, mappers)

        target_states = rearrange(
            next_graph_states,
            "(n l) s -> l n s",
            l=self.seq_len,
        )

        pred_states = []
        prev_states = graph_states[:: self.seq_len]
        for pred_step in range(self.seq_len):
            action_embs = torch.stack(
                [
                    actions_embs[i * self.seq_len + pred_step][
                        item.action_idxs[pred_step]
                    ]
                    for i, item in enumerate(items)
                ]
            )
            new_pred_states = self.predictor(prev_states, action_embs)
            pred_states.append(new_pred_states)
            prev_states = new_pred_states
        pred_states = torch.stack(pred_states)

        state_loss = -torch.mean(
            F.cosine_similarity(pred_states, target_states.detach(), -1)
        )

        return (
            state_loss,
            graph_states.detach(),
            [emb.detach() for emb in actions_embs],
            next_graph_states.detach(),
        )

    def value_prefix_part(self, next_states: Tensor, rewards: Tensor):
        start_idxs = torch.arange(0, next_states.size(0), self.seq_len)
        target_value_prefix = []
        current_prefix = torch.zeros_like(
            start_idxs, dtype=torch.float, device=self.device
        )
        for i in reversed(range(self.seq_len)):
            current_prefix = rewards[start_idxs + i] + Agent.discount * current_prefix
            target_value_prefix.append(current_prefix)
        target_value_prefix = torch.stack(target_value_prefix[::-1], 1)

        states = rearrange(
            next_states,
            "(n l) s -> n l s",
            l=self.seq_len,
        )
        pred_value_prefix, _ = self.value_prefix_net(states)

        value_prefix_loss = F.mse_loss(pred_value_prefix, target_value_prefix.detach())
        return value_prefix_loss

    def generate_d_part(self, graph_states: Tensor, actions_embs: list[Tensor]):
        true_action_embs = torch.stack(
            [emb[np.random.randint(emb.size(0))] for emb in actions_embs]
        )

        generated_action_embs, cls_target = self.action_generator.train_generate(
            graph_states
        )

        true_loss = F.binary_cross_entropy_with_logits(
            self.action_generator.discriminate(
                graph_states.detach(), true_action_embs.detach()
            ),
            torch.ones(graph_states.size(0), device=self.device),
        )
        gen_loss = F.binary_cross_entropy_with_logits(
            self.action_generator.discriminate(
                graph_states.detach(), generated_action_embs.detach()
            ),
            torch.zeros(graph_states.size(0), device=self.device),
        )

        return (true_loss, gen_loss), generated_action_embs, cls_target

    def generate_g_part(
        self, graph_states: Tensor, generated_action_embs: Tensor, cls_target: Tensor
    ):
        g_loss = F.binary_cross_entropy(
            self.action_generator.discriminate(
                graph_states.detach(), generated_action_embs
            ),
            torch.ones(graph_states.size(0), device=self.device),
        )
        cls_loss = F.cross_entropy(
            self.action_generator.classify(
                graph_states.detach(), generated_action_embs
            ),
            cls_target,
        )

        return g_loss, cls_loss

    def actor_critic_part(
        self,
        states: Tensor,
        actions_embs: list[Tensor],
    ):
        (
            _,
            target_value,
            target_policy,
        ) = self.mcts.search(
            states,
            actions_embs,
            self.root_sample_count,
            self.simulation_count,
        )

        pred_value = self.value_net(states)
        pred_policy = []
        for state, embs in zip(states, actions_embs):
            pred_policy.append(self.policy_net(state, embs).softmax(-1))

        value_loss = F.mse_loss(pred_value, target_value.detach())
        policy_loss = torch.mean(
            torch.stack(
                [
                    F.mse_loss(pred, target.detach())
                    for pred, target in zip(pred_policy, target_policy)
                ]
            )
        )

        return (value_loss, policy_loss)

    def training_step(self, items: list[SequenceReplayItem]):
        (
            encode_predict_opt,
            value_prefix_opt,
            discriminate_opt,
            generate_opt,
            actor_critic_opt,
        ) = self.optimizers()

        (
            state_loss,
            states,
            actions_embs,
            next_states,
        ) = self.encode_predict_part(items)
        encode_predict_opt.zero_grad()
        self.manual_backward(state_loss)
        encode_predict_opt.step()
        self.log_dict(
            {
                "loss/state": state_loss,
            }
        )

        value_prefix_loss = self.value_prefix_part(
            next_states,
            tensor(sum([item.rewards for item in items], []), device=self.device),
        )
        value_prefix_opt.zero_grad()
        self.manual_backward(value_prefix_loss)
        value_prefix_opt.step()
        self.log_dict(
            {
                "loss/value_prefix": value_prefix_loss,
            }
        )

        (
            (d_true_loss, d_gen_loss),
            generated_action_embs,
            cls_target,
        ) = self.generate_d_part(states, actions_embs)
        discriminate_opt.zero_grad()
        self.manual_backward(d_true_loss + d_gen_loss)
        discriminate_opt.step()
        self.log_dict(
            {
                "loss/d_true": d_true_loss,
                "loss/d_gen": d_gen_loss,
            }
        )

        (
            g_loss,
            cls_loss,
        ) = self.generate_g_part(
            states,
            generated_action_embs,
            cls_target,
        )
        generate_opt.zero_grad()
        self.manual_backward(g_loss + cls_loss)
        generate_opt.step()
        self.log_dict(
            {
                "loss/g": g_loss,
                "loss/cls": cls_loss,
            }
        )

        (value_loss, policy_loss) = self.actor_critic_part(
            states,
            actions_embs,
        )
        actor_critic_opt.zero_grad()
        self.manual_backward(value_loss + policy_loss)
        actor_critic_opt.step()
        self.log_dict(
            {
                "loss/value": value_loss,
                "loss/policy": policy_loss,
            }
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for sch in self.lr_schedulers():
            sch.step()

        self.eval()
        self.play_step()
        self.train()

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
        env = Environment.from_graphs(
            [graph for graph in batch for _ in range(self.val_predict_num)]
        )
        while True:
            obs = env.observe()
            if len(obs) == 0:
                break

            actions, _ = self.single_step_predict(
                obs,
                self.root_sample_count,
                self.simulation_count,
            )
            env.step(actions)

            total_step = 0
            finished_step = 0
            for e in env.envs:
                f, t = e.progress()
                total_step += t
                finished_step += f

            bar = self.trainer.progress_bar_callback.val_progress_bar
            bar.total = total_step
            bar.n = finished_step
            bar.refresh()

        self.log_dict(
            {
                f"val/env_{idx}": (np.mean(ts) / self.baseline[idx])
                for idx, ts in enumerate(
                    batched(
                        [graph.get_timestamp() for graph in env.envs],
                        self.val_predict_num,
                    )
                )
            },
            batch_size=len(batch),
        )
