from __future__ import annotations

from faultline.generation import CueCondition
from faultline.training.curriculum import (
    CurriculumConfig,
    CurriculumKind,
    CurriculumSampler,
)
from faultline.training.rl_env import EpisodeSummary


def config(kind: CurriculumKind) -> CurriculumConfig:
    return CurriculumConfig(
        kind=kind,
        training_seed=123,
        pool_offset=2_000,
        pool_size=16,
    )


def summary(condition: CueCondition, *, recovered: bool) -> EpisodeSummary:
    return EpisodeSummary(
        pair_id="test",
        condition=condition,
        world_index=0,
        cue=0,
        total_reward=0.0,
        recovered=recovered,
        correct_repair=recovered,
        selected_repair=None,
        decision_steps=1,
        simulator_ticks=0,
        advance_count=0,
        inspect_count=0,
        informative_inspection=False,
        false_repair_count=0,
    )


def test_counter_based_schedule_matches_base_pairs_and_worlds_across_arms() -> None:
    samplers = [
        CurriculumSampler(config(kind))
        for kind in (
            CurriculumKind.RANDOM,
            CurriculumKind.DIFFICULTY,
            CurriculumKind.EPISTEMIC,
        )
    ]

    for _ in range(40):
        episodes = [sampler.sample_episode() for sampler in samplers]
        assert len({episode.pair.pair_id for episode in episodes}) == 1
        assert len({episode.world_index for episode in episodes}) == 1


def test_epistemic_and_random_condition_policies_are_distinct_and_deterministic() -> None:
    epistemic = CurriculumSampler(config(CurriculumKind.EPISTEMIC))
    random_left = CurriculumSampler(config(CurriculumKind.RANDOM))
    random_right = CurriculumSampler(config(CurriculumKind.RANDOM))

    epistemic_conditions = [epistemic.sample_episode().condition for _ in range(50)]
    left_conditions = [random_left.sample_episode().condition for _ in range(50)]
    right_conditions = [random_right.sample_episode().condition for _ in range(50)]

    assert set(epistemic_conditions) == {CueCondition.AMBIGUOUS}
    assert set(left_conditions) == {CueCondition.AMBIGUOUS, CueCondition.REVEALED}
    assert left_conditions == right_conditions


def test_generic_difficulty_sampler_adapts_to_observed_failure() -> None:
    sampler = CurriculumSampler(config(CurriculumKind.DIFFICULTY))
    assert sampler.ambiguous_probability() == 0.5

    for _ in range(30):
        sampler.update(summary(CueCondition.AMBIGUOUS, recovered=False))
        sampler.update(summary(CueCondition.REVEALED, recovered=True))

    assert sampler.ambiguous_probability() > 0.9
    state = sampler.state_dict()
    assert state["failure_ema"]["ambiguous"] > state["failure_ema"]["revealed"]
