from __future__ import annotations

import numpy as np
import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.generation import CueCondition, build_generated_diagnostic_pair
from faultline.training.rl_env import DiagnosticAction, DiagnosticEpisode


def tensors(episode: DiagnosticEpisode) -> tuple[torch.Tensor, ...]:
    observation = episode.observe()
    return (
        torch.from_numpy(observation.nodes).unsqueeze(0),
        torch.from_numpy(observation.adjacency).unsqueeze(0),
        torch.from_numpy(observation.node_mask).unsqueeze(0),
        torch.from_numpy(observation.global_features).unsqueeze(0),
        torch.from_numpy(observation.action_mask).unsqueeze(0),
    )


def test_policy_shapes_parameter_budget_and_gradients() -> None:
    torch.manual_seed(1)
    policy = GraphRecurrentPolicy(hidden_size=256, message_layers=3)
    episode = DiagnosticEpisode(
        build_generated_diagnostic_pair(2),
        CueCondition.AMBIGUOUS,
        world_index=0,
        cue=0,
    )
    nodes, adjacency, node_mask, global_features, action_mask = tensors(episode)
    hidden = policy.initial_hidden(1, "cpu")

    output = policy(nodes, adjacency, node_mask, global_features, hidden, action_mask)
    loss = output.logits.square().mean() + output.value.square().mean()
    loss.backward()

    assert output.logits.shape == (1, 4)
    assert output.value.shape == (1,)
    assert output.hidden.shape == (1, 256)
    assert 800_000 <= policy.parameter_count <= 1_200_000
    assert policy.gru.weight_hh.grad is not None


def test_graph_encoding_is_invariant_to_padded_node_permutation() -> None:
    torch.manual_seed(2)
    policy = GraphRecurrentPolicy(hidden_size=64, message_layers=2)
    episode = DiagnosticEpisode(
        build_generated_diagnostic_pair(9),
        CueCondition.AMBIGUOUS,
        world_index=0,
        cue=1,
    )
    nodes, adjacency, node_mask, global_features, action_mask = tensors(episode)
    permutation = torch.randperm(nodes.shape[1])
    permuted_nodes = nodes[:, permutation]
    permuted_adjacency = adjacency[:, permutation][:, :, permutation]
    permuted_mask = node_mask[:, permutation]
    hidden = policy.initial_hidden(1, "cpu")

    original = policy(nodes, adjacency, node_mask, global_features, hidden, action_mask)
    permuted = policy(
        permuted_nodes,
        permuted_adjacency,
        permuted_mask,
        global_features,
        hidden,
        action_mask,
    )

    torch.testing.assert_close(original.logits, permuted.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(original.value, permuted.value, rtol=1e-5, atol=1e-6)


def test_recurrent_policy_receives_diagnostic_evidence() -> None:
    torch.manual_seed(3)
    policy = GraphRecurrentPolicy(hidden_size=64, message_layers=2)
    pair = build_generated_diagnostic_pair(21)
    worlds = [
        DiagnosticEpisode(pair, CueCondition.AMBIGUOUS, index, cue=0)
        for index in (0, 1)
    ]
    hidden = policy.initial_hidden(2, "cpu")
    initial = worlds[0].observe()
    initial_nodes = torch.from_numpy(np.stack([initial.nodes, initial.nodes]))
    initial_adjacency = torch.from_numpy(np.stack([initial.adjacency, initial.adjacency]))
    initial_mask = torch.from_numpy(np.stack([initial.node_mask, initial.node_mask]))
    initial_global = torch.from_numpy(
        np.stack([initial.global_features, initial.global_features])
    )
    initial_action_mask = torch.from_numpy(np.stack([initial.action_mask, initial.action_mask]))
    first = policy(
        initial_nodes,
        initial_adjacency,
        initial_mask,
        initial_global,
        hidden,
        initial_action_mask,
    )

    for episode in worlds:
        episode.step(DiagnosticAction.ADVANCE)
        episode.step(DiagnosticAction.INSPECT)
    observations = [episode.observe() for episode in worlds]
    second = policy(
        torch.from_numpy(np.stack([item.nodes for item in observations])),
        torch.from_numpy(np.stack([item.adjacency for item in observations])),
        torch.from_numpy(np.stack([item.node_mask for item in observations])),
        torch.from_numpy(np.stack([item.global_features for item in observations])),
        first.hidden,
        torch.from_numpy(np.stack([item.action_mask for item in observations])),
    )

    assert not torch.equal(second.logits[0], second.logits[1])
