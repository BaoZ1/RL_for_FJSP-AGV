import torch
from torch import Tensor, tensor, nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()

        hidden_dim = hidden_dim or dim

        self.w = nn.Linear(dim, hidden_dim * 3)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        features: Tensor,  # [B,C,dim]
        mask: Tensor,  # [B,C,C]
    ):
        _, C, DIM = features.size()

        q, k, v = torch.tensor_split(self.w(features), 3, -1)
        q = q.unsqueeze(-2).expand(-1, -1, C, -1)  # [B,C,C,hidden_dim]
        k = k.unsqueeze(-3).expand(-1, C, -1, -1)  # [B,C,C,hidden_dim]
        v = v.unsqueeze(-3).expand(-1, C, -1, -1)  # [B,C,C,hidden_dim]

        scores = self.score(torch.cat([q, k], -1)).squeeze(-1)  # [B,C,C]
        masked_scores = scores * mask
        normalized_scores = F.softmax(masked_scores, -1)  # [B,C,C]; (-1).sum() == 1

        weighted_features = torch.sum(
            normalized_scores.unsqueeze(-1) * v, -2
        )  # [B,C,hidden_dim]
        return self.out(weighted_features)


class BeginEndEncoder(nn.Module):
    def __init__(self, operation_feature_len: int):
        super().__init__()

        self.begin_projection = nn.Linear(operation_feature_len, operation_feature_len)
        self.end_projection = nn.Linear(operation_feature_len, operation_feature_len)

    def forward(
        self,
        operation_features: Tensor,  # [B,C,F]
        begins_idx: Tensor,  # [B,J]
        ends_idx: Tensor,  # [B,J]
    ):
        DIM = operation_features.size(-1)
        begin_idx = begins_idx.unsqueeze(-1).expand(-1, -1, DIM)
        end_idx = ends_idx.unsqueeze(-1).expand(-1, -1, DIM)
        job_begin_features = operation_features.gather(1, begin_idx)
        job_end_features = operation_features.gather(1, end_idx)
        begin_feature = self.begin_projection(job_begin_features)
        end_feature = self.end_projection(job_end_features)
        return torch.mean(begin_feature, 1), torch.mean(end_feature, 1)


class OperationTimeRelationEncoder(nn.Module):
    def __init__(self, operation_feature_len: int):
        super().__init__()

        self.begin_end = BeginEndEncoder(operation_feature_len)
        self.mix = nn.Linear(operation_feature_len * 3, operation_feature_len)

    def forward(
        self,
        operations: Tensor,  # [B,C,F]
        relations: Tensor,  # [B,C-2,2]  (-1 if need begin or end)
        begins_idx: Tensor,  # [B,J]
        ends_idx: Tensor,  # [B,J]
    ):
        DIM = operations.size(-1)
        p_relations, s_relations = [
            r.expand(-1, -1, DIM) for r in relations.tensor_split(2, -1)
        ]
        predecessors = operations.gather(1, p_relations)
        successors = operations.gather(1, s_relations)

        concated = torch.cat([operations[:, :-2, :], predecessors, successors], dim=-1)
        mixed = self.mix(concated)
        begin, end = self.begin_end(operations, begins_idx, ends_idx)
        return torch.cat([mixed, begin.unsqueeze(1), end.unsqueeze(1)], 1)


class OperationFeatureBlock(nn.Module):
    def __init__(self, operation_feature_len: int):
        super().__init__()

        self.seq = OperationTimeRelationEncoder(operation_feature_len)
        self.same_type = SelfAttention(operation_feature_len)

        self.mix = nn.Linear(operation_feature_len * 2, operation_feature_len)

    def forward(
        self,
        operation_features: Tensor,  # [B,C,F] (last 2 are begin & end)
        relations: Tensor,  # [B,C,2]
        begins_idx: Tensor,  # [B,J]
        ends_idx: Tensor,  # [B,J]
        same_type_mask: Tensor,  # [B,C,C]
    ):
        seq_features = self.seq(operation_features, relations, begins_idx, ends_idx)
        same_type_features = self.same_type(operation_features, same_type_mask)
        return self.mix(torch.cat([seq_features, same_type_features], -1))


class OperationFeatureExtractor(nn.Module):
    def __init__(self, operation_feature_len: int, block_count: int):
        super().__init__()

        self.begin_end_init = BeginEndEncoder(operation_feature_len)

        self.blocks = nn.ModuleList(
            [OperationFeatureBlock(operation_feature_len) for _ in range(block_count)]
        )

    def forward(
        self,
        operation_features: Tensor,  # [B,C-2,F]
        relations: Tensor,  # [B,C-2,2]  (-1 if need begin or end)
        begins_idx: Tensor,  # [B,J]
        ends_idx: Tensor,  # [B,J]
        same_type_mask: Tensor,  # [B,C-2,C-2]
    ):
        begin, end = self.begin_end_init(operation_features, begins_idx, ends_idx)
        begin_idx = operation_features.size(1)
        end_idx = begin_idx + 1
        operation_features = torch.cat(
            [operation_features, begin.unsqueeze(1), end.unsqueeze(1)], 1
        )

        relations[:, :, 0][relations[:, :, 0] == -1] = begin_idx
        relations[:, :, 1][relations[:, :, 1] == -1] = end_idx

        B, c, _ = relations.size()
        C = c + 2

        extended_same_type_mask = torch.zeros(B, C, C)
        extended_same_type_mask[:, :-2, :-2] = same_type_mask
        extended_same_type_mask[:, -2, -2] = 1
        extended_same_type_mask[:, -1, -1] = 1

        for block in self.blocks:
            operation_features = block(
                operation_features,
                relations,
                begins_idx,
                ends_idx,
                extended_same_type_mask,
            )

        return torch.mean(operation_features, dim=1), operation_features


class OperationMachineAttention(nn.Module):
    def __init__(
        self, operation_dim: int, machine_dim: int, hidden_dim: int | None = None
    ):
        super().__init__()

        hidden_dim = hidden_dim or machine_dim

        self.q = nn.Linear(machine_dim, hidden_dim)
        self.kv = nn.Linear(operation_dim, hidden_dim * 2)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(hidden_dim, machine_dim)

    def forward(
        self,
        operation_features: Tensor,  # [B,C,Fc]
        machine_features: Tensor,  # [B,M,Fm]
        type_match_mask: Tensor,  # [B,M,C]
    ):
        _, C, M = type_match_mask.size()
        q: Tensor = self.q(machine_features)  # [B,M,hidden_dim]
        k, v = torch.tensor_split(self.kv(operation_features), 2, -1)
        q = q.unsqueeze(-2).expand(-1, -1, C, -1)
        k = k.unsqueeze(-3).expand(-1, M, -1, -1)
        v = v.unsqueeze(-3).expand(-1, M, -1, -1)  # [B,M,C,hidden_dim]

        scores = self.score(torch.cat([q, k], -1)).squeeze(-1)  # [B,M,C]
        masked_scores = scores * type_match_mask
        normalized_scores = F.softmax(masked_scores, -1)  # [B,M,C]; (-1).sum() == 1

        weighted_features = torch.sum(
            normalized_scores.unsqueeze(-1) * v, -2
        )  # [B,M,hidden_dim]
        return self.out(weighted_features)


class MachineFeatureBlock(nn.Module):
    def __init__(self, operation_dim: int, machine_dim: int):
        super().__init__()

        self.operation_machine = OperationMachineAttention(operation_dim, machine_dim)
        self.machine_machine = SelfAttention(machine_dim)
        self.mix = nn.Linear(machine_dim * 2, machine_dim)

    def forward(
        self,
        operation_features: Tensor,  # [B,C,Fc]
        machine_features: Tensor,  # [B,M,Fm]
        type_match_mask: Tensor,  # [B,M,C]
        same_type_mask: Tensor,  # [B,M,M]
    ):
        operation_machine = self.operation_machine(
            operation_features, machine_features, type_match_mask
        )
        machine_machine = self.machine_machine(machine_features, same_type_mask)

        return self.mix(torch.cat([operation_machine, machine_machine], -1))


class MachineFeatureExtractor(nn.Module):
    def __init__(self, operation_dim: int, machine_dim: int, block_count: int):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MachineFeatureBlock(operation_dim, machine_dim)
                for _ in range(block_count)
            ]
        )

    def forward(
        self,
        operation_features: Tensor,  # [B,C,Fc]
        machine_features: Tensor,  # [B,M,Fm]
        type_match_mask: Tensor,  # [B,M,C]
        same_type_mask: Tensor,  # [B,M,M]
    ):
        for block in self.blocks:
            machine_features = block(operation_features, machine_features, type_match_mask, same_type_mask)
        return torch.mean(machine_features, 2), machine_features
