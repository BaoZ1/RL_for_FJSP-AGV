from fjsp_env import Graph, Action, ActionType
import numpy as np
import asyncio
from itertools import count
from typing import Callable


def single_step_useful_first_predict(actions: list[Action], rand_prob: float = 0):
    useful_action_idxs = (
        [
            i
            for i, action in enumerate(actions)
            if action.action_type in (ActionType.pick, ActionType.transport)
        ]
        if np.random.rand() > rand_prob
        else []
    )
    if len(useful_action_idxs) == 0:
        i = np.random.choice(len(actions))
        act_idx = i
    else:
        i = np.random.choice(len(useful_action_idxs))
        act_idx = useful_action_idxs[i]
    act = actions[act_idx]
    return act, act_idx


def single_step_useful_only_predict(actions: list[Action], rand_prob: float=0):
    useful_action_idxs = (
        [
            i
            for i, action in enumerate(actions)
            if action.action_type in (ActionType.pick, ActionType.transport)
        ]
        if np.random.rand() > rand_prob
        else [*range(len(actions))]
    )

    if len(useful_action_idxs) == 0:
        i = actions.index(Action(ActionType.wait))
        act_idx = i
    else:
        i = np.random.choice(len(useful_action_idxs))
        act_idx = useful_action_idxs[i]
    act = actions[act_idx]
    return act, act_idx


async def simple_predict(
    graph: Graph, rule: Callable[[list[Action]], tuple[Action, int]]
):

    for round_count in count(1):

        action, _ = rule(graph.get_available_actions())
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
