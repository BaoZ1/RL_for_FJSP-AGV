import asyncio
from itertools import count
import onnxruntime
from pathlib import Path
import numpy as np
from FJSP_env import Graph, Action, ActionType, GraphFeature, IdIdxMapper


class Agent:
    def __init__(self, dir: Path):
        self.state_model = onnxruntime.InferenceSession(dir / "state.onnx")
        self.action_model = onnxruntime.InferenceSession(dir / "action.onnx")
        self.value_model = onnxruntime.InferenceSession(dir / "value.onnx")
        self.policy_model = onnxruntime.InferenceSession(dir / "policy.onnx")

    @staticmethod
    def state_inputs(feature: GraphFeature):
        d: dict[str, np.ndarray] = {}
        d["global_attr"] = np.array(feature.global_feature)

        d["operation"] = np.array(feature.operation_features)
        d["machine"] = np.array(feature.machine_features)
        d["AGV"] = np.array(feature.AGV_features)

        d["operation__predecessor__operation__edge"] = np.array(
            feature.predecessor_idx, dtype=np.int64
        ).transpose()

        d["operation__successor__operation__edge"] = np.array(
            feature.successor_idx, dtype=np.int64
        ).transpose()

        d["machine__processable__operation__edge"] = np.array(
            feature.processable_idx, dtype=np.int64
        ).transpose()

        d["machine__processing__operation__edge"] = np.array(
            [[x[0], x[1]] for x in feature.processing], dtype=np.int64
        ).transpose()
        d["machine__processing__operation__attr"] = np.array(
            [[x[2]] for x in feature.processing], dtype=np.float32
        )

        d["machine__waiting__operation__edge"] = np.array(
            [[x[0], x[1]] for x in feature.waiting], dtype=np.int64
        ).transpose()
        d["machine__waiting__operation__attr"] = np.array(
            [[x[2], x[3]] for x in feature.waiting], dtype=np.float32
        )

        d["machine__distance__machine__edge"] = np.array(
            [[x[0], x[1]] for x in feature.distance], dtype=np.int64
        ).transpose()

        d["machine__distance__machine__attr"] = np.array(
            [[x[2]] for x in feature.distance], dtype=np.float32
        )

        d["AGV__position__machine__edge"] = np.array(
            feature.AGV_position, dtype=np.int64
        ).transpose()

        d["AGV__target__machine__edge"] = np.array(
            [[x[0], x[1]] for x in feature.AGV_target], dtype=np.int64
        ).transpose()
        d["AGV__target__machine__attr"] = np.array(
            [[x[2]] for x in feature.AGV_target], dtype=np.float32
        )

        d["AGV__load_from__operation__edge"] = np.array(
            [[x[0], x[1]] for x in feature.AGV_loaded], dtype=np.int64
        ).transpose()
        d["AGV__load_to__operation__edge"] = np.array(
            [[x[0], x[2]] for x in feature.AGV_loaded], dtype=np.int64
        ).transpose()

        rev_d = {}
        for k, v in d.items():
            if "__" not in k:
                continue
            src, name, dst, typ = k.split("__")
            if src == dst:
                continue
            if typ == "edge":
                rev_d[f"{src}__{name}_rev__{src}__{typ}"] = v[::-1]
            else:
                rev_d[f"{src}__{name}_rev__{src}__{typ}"] = v[:, ::-1]

        d.update(rev_d)
        return d

    @staticmethod
    def action_inputs(actions: list[Action], mapper: IdIdxMapper):
        wait_idxs: list[int] = []
        wait_arrays: list[np.ndarray] = []
        pick_idxs: list[int] = []
        pick_arrays: list[np.ndarray] = []
        transport_idxs: list[int] = []
        transport_arrays: list[np.ndarray] = []
        move_idxs: list[int] = []
        move_arrays: list[np.ndarray] = []

        for i, action in enumerate(actions):
            match action.action_type:
                case ActionType.wait:
                    wait_idxs.append(i)
                    wait_arrays.append(np.zeros((0,), dtype=np.int64))
                case ActionType.pick:
                    pick_idxs.append(i)
                    pick_arrays.append(
                        np.array(
                            [
                                mapper.AGV[action.AGV_id],
                                mapper.operation[action.target_product.operation_from],
                                mapper.operation[action.target_product.operation_to],
                                mapper.machine[action.target_machine],
                            ],
                            dtype=np.int64,
                        ),
                    )
                case ActionType.transport:
                    transport_idxs.append(i)
                    transport_arrays.append(
                        np.array(
                            [
                                mapper.AGV[action.AGV_id],
                                mapper.machine[action.target_machine],
                            ],
                            dtype=np.int64,
                        ),
                    )
                case ActionType.move:
                    move_idxs.append(i)
                    move_arrays.append(
                        np.array(
                            [
                                mapper.AGV[action.AGV_id],
                                mapper.machine[action.target_machine],
                            ],
                            dtype=np.int64,
                        ),
                    )

        d = {}
        idxs = []
        d[ActionType.wait.name] = (
            np.stack(wait_arrays) if len(wait_arrays) > 0 else np.zeros((0, 0))
        )
        idxs.extend(wait_idxs)
        d[ActionType.pick.name] = (
            np.stack(pick_arrays) if len(pick_arrays) > 0 else np.zeros((0, 4))
        )
        idxs.extend(pick_idxs)
        d[ActionType.transport.name] = (
            np.stack(transport_arrays) if len(transport_arrays) > 0 else np.zeros((0, 2))
        )
        idxs.extend(transport_idxs)
        d[ActionType.move.name] = (
            np.stack(move_arrays) if len(move_arrays) > 0 else np.zeros((0, 2))
        )
        idxs.extend(move_idxs)
        
        return d, idxs

    def search(self, graph: Graph, sample_num: int, sim_num: int):
        pass

    async def predict(self, graph: Graph, sample_num: int, sim_num: int):
        for round_count in count(1):
            action = self.search(graph, sample_num, sim_num)
            graph = graph.act(action)

            finished_step, total_step = graph.progress()

            yield {
                "round_count": round_count,
                "finished_step": finished_step,
                "total_step": total_step,
                "graph_state": graph,
                "action": action,
            }

            if graph.finished():
                break

            await asyncio.sleep(0)
