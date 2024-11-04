import torch
from torch import Tensor, tensor, nn
from torch.nn import functional as F
import einops

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
        C = features.size(1)

        q, k, v = einops.repeat(
            self.w(features), "B C (split F) -> split B copy C F", split=3, copy=C
        )
        q = einops.rearrange(q, "B copy C F -> B C copy F")

        scores = self.score(torch.cat([q, k], -1)).squeeze(-1)  # [B,C,C]
        masked_scores = scores * mask
        normalized_scores = F.softmax(masked_scores, -1)  # [B,C,C]; (-1).sum() == 1

        weighted_features = torch.sum(
            normalized_scores.unsqueeze(-1) * v, -2
        )  # [B,C,hidden_dim]
        return self.out(weighted_features)


class OperationFeatureBlock(nn.Module):
    def __init__(self, operation_feature_len: int):
        super().__init__()

        self.encoder = nn.Linear(operation_feature_len, operation_feature_len * 4)
        self.space_mix = nn.Linear(operation_feature_len * 3, operation_feature_len)
        self.same_type_mix = nn.Linear(operation_feature_len * 2, operation_feature_len)
        self.mix = nn.Linear(operation_feature_len * 2, operation_feature_len)

    def forward(
        self,
        features: Tensor,  # [B,C,F]
        pred_mask: Tensor,  # [B,C,C]
        succ_mask: Tensor,  # [B,C,C]
        same_type_mask: Tensor,  # [B,C,C]
    ):
        C, F_O = features.shape[1:]
        masks = einops.repeat(
            [pred_mask, succ_mask, same_type_mask],
            "part B idx other -> part B idx other F",
            F=F_O,
        )
        divisor = torch.maximum(may_zero := masks.sum(-2), torch.ones(may_zero.shape))
        origin, *encoded = torch.tensor_split(self.encoder(features), 4, -1)
        encoded = einops.repeat(encoded, "part B idx F -> part B copy idx F", copy=C)
        pred_f, succ_f, same_type_f = (encoded * masks).sum(-2) / divisor
        space_mixed = self.space_mix(torch.cat([pred_f, origin, succ_f], dim=-1))
        same_type_mixed = self.same_type_mix(torch.cat([origin, same_type_f], dim=-1))
        return self.mix(torch.cat([space_mixed, same_type_mixed], dim=-1))


class OperationFeatureExtractor(nn.Module):
    def __init__(self, operation_feature_len: int, block_count: int):
        super().__init__()

        self.blocks = nn.ModuleList(
            [OperationFeatureBlock(operation_feature_len) for _ in range(block_count)]
        )

    def forward(
        self,
        operation_features: Tensor,  # [B,C,F]
        pred_mask: Tensor,  # [B,C,C]
        succ_mask: Tensor,  # [B,C,C]
        same_type_mask: Tensor,  # [B,C,C]
    ):
        for block in self.blocks:
            operation_features = block(
                operation_features,
                pred_mask,
                succ_mask,
                same_type_mask,
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
        q: Tensor = self.q(machine_features)  # [B,M,H]
        k, v = torch.tensor_split(self.kv(operation_features), 2, -1)
        q = q.unsqueeze(-2).expand(-1, -1, C, -1)  # [B,M,C,H]
        k, v = einops.repeat([k, v], "part B C H -> part B M C H", M=M)

        scores = self.score(torch.cat([q, k], -1)).squeeze(-1)  # [B,M,C]
        masked_scores = scores * type_match_mask
        normalized_scores = F.softmax(masked_scores, -1)  # [B,M,C]; (-1).sum() == 1

        weighted_features = torch.sum(
            normalized_scores.unsqueeze(-1) * v, -2
        )  # [B,M,H]
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
            machine_features = block(
                operation_features, machine_features, type_match_mask, same_type_mask
            )
        return torch.mean(machine_features, 2), machine_features
