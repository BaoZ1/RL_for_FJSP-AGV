import torch
from torch import nn, Tensor, tensor
from torch_geometric.data import HeteroData, Batch
from torch_geometric import nn as gnn
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeType, EdgeType
from FJSP_env import Graph


class BidiGATv2Conv(nn.Module):

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int | tuple[int, int],
        edge_dim: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)

        self.forward_conv = (
            gnn.GATv2Conv(
                in_channels,
                out_channels[0],
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            ),
        )
        self.backward_conv = (
            gnn.GATv2Conv(
                in_channels[::-1],
                out_channels[1],
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            ),
        )

    def forward(
        self,
        x: tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ):
        return (
            self.backward_conv(x, edge_index, edge_attr),
            self.forward_conv(x[::-1], edge_index.flip(0), edge_attr),
        )


class BidiHeteroConv(nn.Module):
    def __init__(
        self,
        convs: dict[tuple[str, str, str], nn.Module],
        aggr: str | None = "sum",
    ):
        super().__init__()

        self.convs = ModuleDict(convs)
        self.aggr = aggr

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        edge_attr_dict: dict[EdgeType, Tensor] | None = None,
    ):
        out_dict: dict[str, list[Tensor]] = {}

        for edge_type, conv in self.convs.items():
            src, _, dst = edge_type

            x = (x_dict.get(src), x_dict.get(src))
            edge_index = edge_index_dict.get(edge_type)
            edge_attr = edge_attr_dict.get(edge_type) if edge_attr_dict else None

            if not edge_index:
                continue

            out_src, out_dst = conv(x, edge_index, edge_attr)

            out_dict.setdefault(src, []).append(out_src)
            out_dict.setdefault(dst, []).append(out_dst)

        for key, value in out_dict.items():
            out_dict[key] = gnn.conv.hetero_conv.group(value, self.aggr)

        return out_dict


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

        self.hetero_relation_extract = BidiHeteroConv(
            {
                **{
                    ("machine", name, "operation"): BidiGATv2Conv(
                        (node_channels[1], node_channels[0]),
                        (node_channels[1], node_channels[0]),
                        edge_dim,
                        dropout,
                    )
                    for name, edge_dim in [
                        ("processable", None),
                        ("processing", 1),
                        ("waiting", 2),
                    ]
                },
                **{
                    ("AGV", name, "machine"): BidiGATv2Conv(
                        (node_channels[2], node_channels[1]),
                        (node_channels[2], node_channels[1]),
                        edge_dim,
                        dropout,
                    )
                    for name, edge_dim in [
                        ("position", None),
                        ("target", 1),
                    ]
                },
                **{
                    ("AGV", name, "operation"): BidiGATv2Conv(
                        (node_channels[2], node_channels[0]),
                        (node_channels[2], node_channels[0]),
                        edge_dim,
                        dropout,
                    )
                    for name, edge_dim in [
                        ("position", None),
                        ("target", None),
                    ]
                },
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


class Mixer(nn.Module):
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
    ):
        edge_index_dict: dict[EdgeType, Tensor] = {}

        for k, batch in batch_dict.items():
            x_dict[f"{k}_global"] = self.global_tokens[k].repeat(
                batch.max().item() + 1, 1
            )
            edge_index_dict[(k, "global", f"{k}_global")] = (
                tensor([[i, b] for i, b in enumerate(batch)])
                .T.contiguous()
                .to(batch.device)
            )

        global_dict: dict[NodeType, Tensor] = self.convs(x_dict, edge_index_dict)

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


class Model(nn.Module):

    def __init__(
        self,
        operation_hidden_channels: int,
        machine_hidden_channels: int,
        AGV_hidden_channels: int,
        extract_num_layers: int,
        global_channels: tuple[int, int, int],
        graph_global_channels: int,
    ):
        super().__init__()

        self.init_project = nn.ModuleDict(
            {
                "operation": nn.Linear(
                    Graph.operation_feature_size, operation_hidden_channels
                ),
                "machine": nn.Linear(
                    Graph.machine_feature_size, machine_hidden_channels
                ),
                "AGV": nn.Linear(Graph.AGV_feature_size, AGV_hidden_channels),
            }
        )

        self.backbone = nn.ModuleList(
            [
                ExtractLayer(
                    (
                        operation_hidden_channels,
                        machine_hidden_channels,
                        AGV_hidden_channels,
                    )
                )
                for _ in range(extract_num_layers)
            ]
        )

        self.mix = Mixer(
            (
                operation_hidden_channels,
                machine_hidden_channels,
                AGV_hidden_channels,
            ),
            global_channels,
            graph_global_channels,
        )

    def forward(self, data: HeteroData | Batch):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        if isinstance(data, Batch):
            batch_dict = {k: data[k].batch for k in x_dict}
        else:
            batch_dict = None

        for k, v in x_dict.items():
            x_dict[k] = self.init_project[k](v)

        for layer in self.backbone:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        global_dict, graph_feature = self.mix(x_dict, batch_dict)
        
        return x_dict, global_dict, graph_feature
