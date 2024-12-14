import torch
from torch import nn, Tensor, tensor
from torch.nn import functional as F
from torch import optim
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
)
from .utils import *
import lightning as L
from collections.abc import Callable
from tqdm import tqdm


class ExtractLayer(nn.Module):
    def __init__(
        self,
        node_channels: tuple[int, int, int],
        dropout: float = 0.2,
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

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_attr_dict: dict[EdgeType, Tensor] | None = None,
    ):
        res: dict[NodeType, Tensor] = x_dict if self.residual else {}

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

        self.graph_mix = nn.Sequential(
            nn.Linear(sum(global_channels), graph_global_channels),
            nn.LeakyReLU(),
            nn.Linear(graph_global_channels, graph_global_channels),
        )

    def forward(
        self, x_dict: dict[NodeType, Tensor], batch_dict: dict[NodeType, Tensor]
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

        graph_feature = self.graph_mix(
            torch.cat(
                [
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
                x_dict[key] = F.leaky_relu(x_dict[key])

        global_dict, graph_feature = self.mix(x_dict, batch_dict)

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
            nn.LeakyReLU(),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.transport_encoder = nn.Sequential(
            nn.Linear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                out_channels * 2,
            ),
            nn.LeakyReLU(),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(
                (node_channels[1] + node_channels[2]),  # AGV  # machine_target
                out_channels * 2,
            ),
            nn.LeakyReLU(),
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


class PredictNet(nn.Module):
    def __init__(self, state_channels: int, action_channels: int):
        super().__init__()
        base_channels = state_channels + action_channels
        self.base = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
        )

        self.value = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels // 2),
            nn.Tanh(),
            nn.Linear(base_channels // 2, 1),
        )

        self.next_state = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, state_channels),
        )

    def forward(self, state: Tensor, action: Tensor):
        base_data: Tensor = self.base(torch.cat([state, action], 1))
        return self.value(base_data).flatten(), self.next_state(base_data)

    __call__: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


class StatePredictor(nn.Module):
    def __init__(self, state_channels: int, action_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_channels + action_channels, state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels * 2),
            nn.Tanh(),
            nn.Linear(state_channels * 2, state_channels),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.model(torch.cat([states, actions], 1))

    __call__: Callable[[Tensor, Tensor], Tensor]


class ActionGenerator(nn.Module):
    def __init__(self, state_channels: int, action_channels: int, out_num: int):
        super().__init__()

        self.action_channels = action_channels
        self.out_num = out_num

        base_channels = state_channels + action_channels

        self.generator = nn.Sequential(
            nn.Linear(state_channels, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.Tanh(),
            nn.Linear(base_channels * 2, base_channels),
            nn.Tanh(),
            nn.Linear(base_channels, action_channels * out_num),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(action_channels, action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels * 2),
            nn.Tanh(),
            nn.Linear(action_channels * 2, action_channels),
            nn.Tanh(),
            nn.Linear(action_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, states: Tensor, true_samples: Tensor):
        generated: Tensor = self.generator(states)
        generated = generated.reshape(-1, self.out_num, self.action_channels)
        return self.discriminator(torch.cat([true_samples, generated], 1)).squeeze(-1), generated

    __call__: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]

    def generate(self, state: Tensor):
        return self.generator(state).reshape(self.out_num, self.action_channels)


class BackboneTrainer(L.LightningModule):

    def __init__(
        self,
        envs: Environment,
        extractor: StateExtract,
        action_encoder: ActionEncoder,
        predictor: PredictNet,
        action_generator: ActionGenerator,
    ):
        super().__init__()

        self.envs = envs

        self.extractor = extractor
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.action_generator = action_generator

        self.buffer = ReplayBuffer[Observation]()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch: list[ReplayItem[Observation]], batch_idx):
        batched_graph = Batch.from_data_list(
            [build_graph(item.state.feature) for item in batch]
        )
        batched_graph = batched_graph.to(self.device)
        offsets = get_offsets(batched_graph)
        mappers = [item.state.mapper for item in batch]

        states = self.extractor(batched_graph)
        action_embs = self.action_encoder.single_action_encode(
            [item.action for item in batch],
            states[0],
            offsets,
            mappers,
        )

        value_pred, state_pred = self.predictor(states[2], action_embs)

        value_target = torch.tensor([item.reward for item in batch], device=self.device)
        value_loss = F.mse_loss(value_pred, value_target)

        true_action_count = min(
            self.action_generator.out_num,
            min(len(item.state.action_list) for item in batch),
        )
        true_action_lists = [
            item.state.action_list[:true_action_count] for item in batch
        ]
        true_actions = self.action_encoder(
            true_action_lists, states[0], offsets, mappers
        )
        true_actions = torch.stack(true_actions)
        discriminate_result, generated_actions = self.action_generator(states[2], true_actions)
        discriminate_target = torch.cat(
            [
                torch.ones(len(batch), true_action_count, device=self.device),
                torch.zeros(len(batch), self.action_generator.out_num, device=self.device),
            ],
            1,
        )
        generate_loss = F.binary_cross_entropy(discriminate_result, discriminate_target)
        
        batched_next_graph = Batch.from_data_list(
            [build_graph(item.next_state.feature) for item in batch]
        )
        batched_next_graph = batched_next_graph.to(self.device)
        state_target = self.extractor(batched_next_graph)[2]
        state_loss = F.mse_loss(state_pred, state_target)

        self.log_dict(
            {
                "loss/value": value_loss,
                "loss/generate": generate_loss,
                "loss/state": state_loss,
            },
            True,
        )
        if batch_idx == 0:
            self.log_dict(
                {"value/pred": value_pred[0], "value/target": value_target[0]}
            )
            self.log_dict(
                {
                    "generate/var/true": true_actions.var(-1).mean(),
                    "generate/var/generated": generated_actions.var(-1).mean(),
                }
            )
            self.log_dict(
                {
                    "state/var/pred": state_pred.var(),
                    "state/var/target": state_target.var(),
                }
            )
        return value_loss + generate_loss + state_loss

    def update_buffer(self):
        progress = tqdm(range(200), desc="Update Buffer")
        for _ in progress:
            obs = self.envs.observe()
            act: list[Action] = []
            for env_action in [ob.action_list for ob in obs]:
                has_useful = False
                for a in env_action:
                    if a.action_type != ActionType.move:
                        act.append(a)
                        has_useful = True
                        break
                if not has_useful:
                    act.append(np.random.choice(env_action))
            lb, done, next_obs = self.envs.step(act)
            for data in zip(obs, act, lb, done, next_obs):
                self.buffer.append(*data)

    def on_train_epoch_end(self):
        self.update_buffer()

    def train_dataloader(self):
        self.update_buffer()
        return DataLoader(
            ReplayDataset(self.buffer, 500),
            32,
            collate_fn=lambda batch: batch,
        )
