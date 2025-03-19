from modules import *
from utils import Metadata
from pathlib import Path
from torch.export import Dim


def export_extractor(m: StateExtract, dir: str):
    m.eval()

    input_names: list[str] = ["global_attr"]
    input_names.extend(Metadata.node_types)
    input_names.extend(["__".join(e) + "__edge" for e in Metadata.edge_types])
    input_names.extend(["__".join(a) + "__attr" for a in Metadata.edge_attrs.keys()])

    output_names = (
        [n + "_embeds" for n in Metadata.node_types]
        + [n + "_global_embed" for n in Metadata.node_types]
        + ["graph_embed"]
    )

    edge_count_dims = {
        e: Dim(f"{"__".join(e)}__count") for e in Metadata.edge_types
    }
    dynamic_shapes = {
        "global_attr": (Dim.STATIC,),
        "x_dict": {
            n: (
                Dim(f"{n}__count", min=(2 if n == "operation" else 1)),
                Dim.STATIC,
            )
            for n in Metadata.node_types
        },
        "edge_index_dict": {
            e: (
                Dim.STATIC,
                edge_count_dims[e],
            )
            for e in Metadata.edge_types
        },
        "edge_attr_dict": {
            a: (
                edge_count_dims[a],
                Dim.STATIC,
            )
            for a in Metadata.edge_attrs
        },
    }

    torch.onnx.export(
        torch.export.export(
            m,
            (
                torch.empty(3),
                {
                    "operation": torch.empty(3, Graph.operation_feature_size),
                    "machine": torch.empty(3, Graph.machine_feature_size),
                    "AGV": torch.empty(3, Graph.AGV_feature_size),
                },
                {e: torch.empty(2, 3, dtype=torch.int64) for e in Metadata.edge_types},
                {a: torch.empty(3, x) for a, x in Metadata.edge_attrs.items()},
            ),
            dynamic_shapes=dynamic_shapes,
        ),
        (),
        Path(dir) / "state.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        external_data=False,
    )


def export_encoder(m: ActionEncoder, dir: str):
    m.eval()

    input_names = []
    input_names.extend([f"{n}_embeds" for n in Metadata.node_types])
    input_names.extend([f"{t}_actions" for t in ["wait", "pick", "transport", "move"]])

    output_names = ["encoded_actions"]

    dynamic_shapes = {
        "embeds": {
            n: (
                Dim(f"{n}__count", min=(2 if n == "operation" else 1)),
                Dim.STATIC,
            )
            for n in Metadata.node_types
        },
        "actions": {
            t: (
                Dim(f"{t}__count"),
                Dim.STATIC,
            )
            for t in ActionType.__members__
        },
    }

    torch.onnx.export(
        torch.export.export(
            m,
            (
                {
                    "operation": torch.empty(3, m.node_channels[0]),
                    "machine": torch.empty(3, m.node_channels[1]),
                    "AGV": torch.empty(3, m.node_channels[2]),
                },
                {
                    ActionType.wait.name: torch.empty(3, 0, dtype=torch.int64),
                    ActionType.pick.name: torch.empty(3, 4, dtype=torch.int64),
                    ActionType.transport.name: torch.empty(3, 2, dtype=torch.int64),
                    ActionType.move.name: torch.empty(3, 2, dtype=torch.int64),
                },
            ),
            dynamic_shapes=dynamic_shapes,
        ),
        (),
        Path(dir) / "action.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        external_data=False,
    )


def export_value(m: ValueNet, dir: str):
    m.eval()

    input_names = ["state"]
    output_names = ["value"]

    dynamic_shapes = {
        "state": (Dim.STATIC,),
    }

    torch.onnx.export(
        torch.export.export(
            m,
            (torch.empty(m.state_channels),),
            dynamic_shapes=dynamic_shapes,
        ),
        (),
        Path(dir) / "value.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        external_data=False,
    )


def export_policy(m: PolicyNet, dir: str):
    m.eval()

    input_names = ["state", "actions"]
    output_names = ["logits"]

    dynamic_shapes = {
        "state": (Dim.STATIC,),
        "actions": (Dim("action_count"), Dim.STATIC),
    }

    torch.onnx.export(
        torch.export.export(
            m,
            (
                torch.empty(m.state_channels),
                torch.empty(3, m.action_channels),
            ),
            dynamic_shapes=dynamic_shapes,
        ),
        (),
        Path(dir) / "policy.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        external_data=False,
    )

def export(m: Model, dir: str):
    export_extractor(m.extractor, dir)
    export_encoder(m.action_encoder, dir)
    export_value(m.value_net, dir)
    export_policy(m.policy_net, dir)