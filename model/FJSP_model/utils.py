from FJSP_env import GraphFeature, build_graph, Action
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class ReplayBuffer:
    def __init__(self):
        self.buffer: list[tuple[GraphFeature, Action, float, bool, GraphFeature]] = []

    def append(
        self,
        state: GraphFeature,
        action: Action,
        reward: float,
        done: bool,
        next_state: GraphFeature,
    ):
        self.buffer.append((state, action, reward, done, next_state))

    def sample(
        self, num
    ) -> list[tuple[GraphFeature, Action, float, bool, GraphFeature]]:
        assert len(self.buffer) > num

        return np.random.choice(self.buffer, num, False)


class ReplayDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int):
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for item in self.buffer.sample(self.sample_size):
            yield item
