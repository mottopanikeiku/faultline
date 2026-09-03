"""Permutation-aware graph encoder with recurrent actor and critic heads."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from faultline.training.rl_env import ACTION_COUNT, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    logits: Tensor
    value: Tensor
    hidden: Tensor


class GraphRecurrentPolicy(nn.Module):
    """Small message-passing encoder followed by a GRU belief state."""

    def __init__(self, hidden_size: int = 256, message_layers: int = 3) -> None:
        super().__init__()
        if hidden_size <= 0 or message_layers <= 0:
            raise ValueError("hidden size and message layer count must be positive")
        self.hidden_size = hidden_size
        self.node_encoder = nn.Linear(NODE_FEATURE_DIM, hidden_size)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(message_layers)
        )
        self.neighbor_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(message_layers)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(message_layers)
        )
        self.global_encoder = nn.Linear(2 * hidden_size + GLOBAL_FEATURE_DIM, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self,
        nodes: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        hidden: Tensor,
        action_mask: Tensor | None = None,
    ) -> PolicyOutput:
        """Encode one recurrent step for a batch of padded graphs."""
        mask = node_mask.unsqueeze(-1)
        node_hidden = torch.relu(self.node_encoder(nodes)) * mask
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized_adjacency = adjacency / degree
        for self_layer, neighbor_layer, layer_norm in zip(
            self.self_layers,
            self.neighbor_layers,
            self.layer_norms,
            strict=True,
        ):
            messages = torch.bmm(normalized_adjacency, node_hidden)
            update = self_layer(node_hidden) + neighbor_layer(messages)
            node_hidden = torch.relu(layer_norm(update)) * mask

        count = mask.sum(dim=1).clamp_min(1.0)
        pooled = node_hidden.sum(dim=1) / count
        target_weight = nodes[:, :, 8:9] * mask
        target_count = target_weight.sum(dim=1).clamp_min(1.0)
        target = (node_hidden * target_weight).sum(dim=1) / target_count
        encoded = torch.relu(
            self.global_encoder(torch.cat((pooled, target, global_features), dim=-1))
        )
        next_hidden = self.gru(encoded, hidden)
        logits = self.actor(next_hidden)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        value = self.critic(next_hidden).squeeze(-1)
        return PolicyOutput(logits=logits, value=value, hidden=next_hidden)

    @torch.no_grad()
    def act(
        self,
        nodes: Tensor,
        adjacency: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        hidden: Tensor,
        action_mask: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self(
            nodes,
            adjacency,
            node_mask,
            global_features,
            hidden,
            action_mask,
        )
        log_probabilities = torch.log_softmax(output.logits, dim=-1)
        if deterministic:
            action = output.logits.argmax(dim=-1)
        else:
            action = torch.multinomial(log_probabilities.exp(), 1).squeeze(-1)
        selected_log_probability = log_probabilities.gather(
            -1,
            action.unsqueeze(-1),
        ).squeeze(-1)
        return action, selected_log_probability, output.value, output.hidden

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
