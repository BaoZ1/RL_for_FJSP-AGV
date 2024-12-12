from FJSP_env import GraphFeature, build_graph, Action
import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class ReplayBuffer[T]:
    def __init__(self, max_len: int=1000):
        self.buffer: list[tuple[T, Action, float, bool, T]] = []
        self.max_len = max_len

    def append(
        self,
        state: T,
        action: Action,
        reward: float,
        done: bool,
        next_state: T,
    ):
        self.buffer.append((state, action, reward, done, next_state))
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def sample(
        self, num
    ) -> list[tuple[T, Action, float, bool, T]]:
        assert len(self.buffer) > num

        return np.random.choice(self.buffer, num, False)


class ReplayDataset[T](IterableDataset):
    def __init__(self, buffer: ReplayBuffer[T], sample_size: int):
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for item in self.buffer.sample(self.sample_size):
            yield item
