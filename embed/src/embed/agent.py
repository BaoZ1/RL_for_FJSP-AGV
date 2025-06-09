import asyncio
from typing import Self
from itertools import count
import math
import onnxruntime
from pathlib import Path
import numpy as np
from fjsp_env import Graph, Action, ActionType, GraphFeature, IdIdxMapper


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp, axis)


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
        logits: np.ndarray,
    ):
        self.graph = graph.copy()
        self.actions = actions
        self.reward = reward
        for i in range(len(actions)):
            self.children.append(Node(self, i, logits[i]))

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
        return np.array([child.logit for child in self.children])

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return np.mean(self.value_list)

    def sigma_Qs(self):
        sum_p_q = 0
        sum_prob = 0
        sum_visit = 0
        probs = softmax(np.array(self.logits()), 0)
        for child in self.children:
            if child.expanded():
                sum_p_q += probs[child.index] * (child.reward + 1.0 * child.value())
                sum_prob += probs[child.index]
                sum_visit += child.visit_count
        if sum_prob < 1e-6:
            v_mix = self.value()
        else:
            v_mix = (1 / (1 + sum_visit)) * (
                self.value() + sum_visit / sum_prob * sum_p_q
            )

        completed_Qs = np.array(
            [
                ((child.reward + 1.0 * child.value()) if child.expanded() else v_mix)
                for child in self.children
            ]
        )
        child_visit_counts = np.array([child.visit_count for child in self.children])
        max_child_visit_count: int = child_visit_counts.max()
        return (50 + max_child_visit_count) * 0.02 * completed_Qs

    def improved_policy(self):
        logits = np.array(self.logits())
        sigma_Qs = self.sigma_Qs()
        # if self.depth == 0:
        #     print(logits, sigma_Qs)
        if logits.size > 1:
            logits = (logits - logits.mean()) / logits.std()
        else:
            logits = logits - logits.mean()
        if not np.all(sigma_Qs == sigma_Qs[0]):
            sigma_Qs = (sigma_Qs - sigma_Qs.mean()) / sigma_Qs.std()
        else:
            sigma_Qs = sigma_Qs - sigma_Qs.mean()

        return softmax(logits + sigma_Qs, 0)

    def select_action(self):
        child_visit_counts = np.array([child.visit_count for child in self.children])
        finished_adj = np.array(
            [
                -np.inf if child.expanded() and child.finished else 0
                for child in self.children
            ]
        )
        idx: int = np.argmax(
            self.improved_policy()
            - child_visit_counts / (1 + child_visit_counts.sum())
            + finished_adj
        )
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


class Agent:
    def __init__(self, dir: Path):
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        self.state_model = onnxruntime.InferenceSession(
            dir / "state.onnx",
            providers=providers,
        )
        self.action_model = onnxruntime.InferenceSession(
            dir / "action.onnx",
            providers=providers,
        )
        self.value_model = onnxruntime.InferenceSession(
            dir / "value.onnx",
            providers=providers,
        )
        self.policy_model = onnxruntime.InferenceSession(
            dir / "policy.onnx",
            providers=providers,
        )

    @staticmethod
    def state_inputs(feature: GraphFeature):
        d: dict[str, np.ndarray] = {}
        d["global_attr"] = np.array(feature.global_feature, dtype=np.float32)

        d["operation"] = np.array(feature.operation_features, dtype=np.float32)
        d["machine"] = np.array(feature.machine_features, dtype=np.float32)
        d["AGV"] = np.array(feature.AGV_features, dtype=np.float32)

        d["operation__predecessor__operation__edge"] = (
            np.array(
                feature.predecessor_idx,
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        d["operation__successor__operation__edge"] = (
            np.array(
                feature.successor_idx,
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        d["machine__processable__operation__edge"] = (
            np.array(
                feature.processable_idx,
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        d["machine__processing__operation__edge"] = (
            np.array(
                [[x[0], x[1]] for x in feature.processing],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )
        d["machine__processing__operation__attr"] = np.array(
            [[x[2]] for x in feature.processing],
            dtype=np.float32,
        ).reshape(-1, 1)

        d["machine__waiting__operation__edge"] = (
            np.array(
                [[x[0], x[1]] for x in feature.waiting],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )
        d["machine__waiting__operation__attr"] = np.array(
            [[x[2], x[3]] for x in feature.waiting],
            dtype=np.float32,
        ).reshape(-1, 2)

        d["machine__distance__machine__edge"] = (
            np.array(
                [[x[0], x[1]] for x in feature.distance],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        d["machine__distance__machine__attr"] = np.array(
            [[x[2]] for x in feature.distance],
            dtype=np.float32,
        ).reshape(-1, 1)

        d["AGV__position__machine__edge"] = (
            np.array(
                feature.AGV_position,
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        d["AGV__target__machine__edge"] = (
            np.array(
                [[x[0], x[1]] for x in feature.AGV_target],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )
        d["AGV__target__machine__attr"] = np.array(
            [[x[2]] for x in feature.AGV_target],
            dtype=np.float32,
        ).reshape(-1, 1)

        d["AGV__load_from__operation__edge"] = (
            np.array(
                [[x[0], x[1]] for x in feature.AGV_loaded],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )
        d["AGV__load_to__operation__edge"] = (
            np.array(
                [[x[0], x[2]] for x in feature.AGV_loaded],
                dtype=np.int64,
            )
            .reshape(-1, 2)
            .transpose()
        )

        rev_d = {}
        for k, v in d.items():
            if "__" not in k:
                continue
            src, name, dst, typ = k.split("__")
            if src == dst:
                continue
            rev_d[f"{dst}__{name}_rev__{src}__{typ}"] = np.flip(
                v, 0 if typ == "edge" else 1
            )

        d.update(rev_d)
        return d

    @staticmethod
    def state_outputs(raw: list[np.ndarray]):
        it = iter(raw)
        return (
            {
                "operation_embeds": next(it),
                "machine_embeds": next(it),
                "AGV_embeds": next(it),
            },
            {
                "operation_global_embeds": next(it),
                "machine_global_embeds": next(it),
                "AGV_global_embeds": next(it),
            },
            next(it),
        )

    def state(self, feature: GraphFeature):
        return self.state_outputs(
            self.state_model.run(None, self.state_inputs(feature))
        )

    @staticmethod
    def action_inputs(
        actions: list[Action], mapper: IdIdxMapper, embeds: dict[str, np.ndarray]
    ):
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

        d: dict[str, np.ndarray] = {}
        idxs: list[int] = []
        d[f"{ActionType.wait.name}_actions"] = (
            np.stack(wait_arrays)
            if len(wait_arrays) > 0
            else np.zeros((0, 0), dtype=np.int64)
        )
        idxs.extend(wait_idxs)
        d[f"{ActionType.pick.name}_actions"] = (
            np.stack(pick_arrays)
            if len(pick_arrays) > 0
            else np.zeros((0, 4), dtype=np.int64)
        )
        idxs.extend(pick_idxs)
        d[f"{ActionType.transport.name}_actions"] = (
            np.stack(transport_arrays)
            if len(transport_arrays) > 0
            else np.zeros((0, 2), dtype=np.int64)
        )
        idxs.extend(transport_idxs)
        d[f"{ActionType.move.name}_actions"] = (
            np.stack(move_arrays)
            if len(move_arrays) > 0
            else np.zeros((0, 2), dtype=np.int64)
        )
        idxs.extend(move_idxs)

        d.update(embeds)

        return d, idxs

    @staticmethod
    def action_outputs(raw: list[np.ndarray]):
        return raw[0]

    def action(
        self, actions: list[Action], mapper: IdIdxMapper, embeds: dict[str, np.ndarray]
    ):
        d, idxs = self.action_inputs(actions, mapper, embeds)
        return self.action_outputs(self.action_model.run(None, d))[idxs]

    @staticmethod
    def value_inputs(global_state: np.ndarray):
        return {"state": global_state}

    @staticmethod
    def value_outputs(raw: list[np.ndarray]) -> float:
        return raw[0]

    def value(self, global_state: np.ndarray):
        return self.value_outputs(
            self.value_model.run(None, self.value_inputs(global_state))
        )

    @staticmethod
    def policy_inputs(global_state: np.ndarray, actions_emb: np.ndarray):
        return {
            "state": global_state,
            "actions": actions_emb,
        }

    @staticmethod
    def policy_outputs(raw: list[np.ndarray]):
        return raw[0]

    def policy(self, global_state: np.ndarray, actions_emb: np.ndarray):
        return self.policy_outputs(
            self.policy_model.run(
                None,
                self.policy_inputs(global_state, actions_emb),
            )
        )

    def search(self, graph: Graph, sample_num: int, sim_num: int):
        assert sample_num & (sample_num - 1) == 0

        feature, mapper = graph.features()

        (
            node_states,
            _,
            global_state,
        ) = self.state(feature)

        actions = graph.get_available_actions()
        actions_emb = self.action(actions, mapper, node_states)

        value = self.value(global_state)

        logits = self.policy(global_state, actions_emb)

        if sample_num == 1:
            return actions[np.argpartition(logits, -1)[-1]]

        root = RootNode()
        root.visit_count += 1
        root.value_list.append(value)
        root.expand(graph, actions, 0, logits)

        g = np.random.gumbel(0, 1, len(actions))
        begin_count = min(sample_num, len(actions))

        m: int = 2 ** math.ceil(math.log2(begin_count))
        # n = math.floor(m * old_n / old_m * math.log(m, old_m))
        n = math.floor(sim_num * math.log(m, sample_num))
        counts: list[int] = []
        remained = m
        sum_count = 0
        while remained > 1:
            sum_count += math.floor(n / (math.log2(m) * remained)) * remained
            counts.append(sum_count)
            remained //= 2
        sim_SH_idxs = counts

        sim_stage = 0
        remaining_actions_index: list[int] = np.argpartition(
            g + logits,
            -begin_count,
        )[-begin_count:].tolist()
        root.set_remain_actions_idx(remaining_actions_index)

        for sim_idx in count():
            if sim_stage == len(sim_SH_idxs):
                assert len(remaining_actions_index) == 1
                break

            if not root.finished:
                node = root.select_action()
                while node.expanded():
                    node = node.select_action()
                target_action = node.parent.actions[node.index]
                parent_graph = node.parent.graph

                graph, reward = parent_graph.act(target_action)

                feature, mapper = graph.features()

                (
                    node_states,
                    _,
                    global_state,
                ) = self.state(feature)

                value = self.value(global_state) * (1 - graph.finished())
                actions = graph.get_available_actions()
                actions_emb = self.action(actions, mapper, node_states)

                logits = self.policy(global_state, actions_emb)

                node.expand(graph, actions, reward, logits)
                while True:
                    node.visit_count += 1
                    node.value_list.append(value)
                    if (p := node.parent) is not None:
                        value = node.reward + 1.0 * value
                        node = p
                    else:
                        break

            if sim_SH_idxs[sim_stage] == sim_idx:
                remain_g = g[remaining_actions_index]
                logits = root.logits()[remaining_actions_index]
                sigma_Qs = root.sigma_Qs()[remaining_actions_index]
                remain_count = math.ceil(len(remaining_actions_index) / 2)
                remain_idx = np.argpartition(
                    remain_g + logits + sigma_Qs,
                    -remain_count,
                )[-remain_count:]
                new_idx = [remaining_actions_index[idx] for idx in remain_idx.tolist()]
                root.set_remain_actions_idx(new_idx)
                remaining_actions_index = new_idx
                sim_stage += 1

        assert len(remaining_actions_index) == 1
        return root.actions[remaining_actions_index[0]]

    async def predict(self, graph: Graph, sample_num: int, sim_num: int):
        for round_count in count(1):
            action = self.search(graph, sample_num, sim_num)
            graph = graph.act(action)[0]

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
