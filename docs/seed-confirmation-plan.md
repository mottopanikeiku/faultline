# Independent confirmation of the curriculum comparison

My first eight-seed study and the new 32-seed replication did not resolve whether ambiguous-only training beats both alternatives. I report them separately. I use neither cohort in this new confirmation's primary estimates or intervals.

## Method and independent data

I keep the policy, PPO settings, training pool, 30,000-decision-step target, validation worlds, episode-level score definition and portable checkpoint format unchanged. I use **one PyTorch thread instead of four**, requesting one CPU core and 1 GiB per Modal container. This changes arithmetic order, not the learning method; I do not assume bit-identical training across thread counts. I pilot seed 499 once in each curriculum and exclude these three runs. New matched training seeds start at 500 and are fixed in the evaluation TOML before the confirmation begins.

The primary outcomes are the paired training-seed mean differences Epistemic minus Random and Epistemic minus Difficulty. Each gets the existing percentile bootstrap: 10,000 paired-seed resamples, 95% two-sided interval, fixed RNG seed 20261008 plus 100 and comparison index. Episodes are not independent sampling units. Completed failed-learning runs remain in the analysis. There is one analysis after every selected seed completes, with no outcome-based interim stopping or replacement of poor seeds.

Five percentage points (0.05 on the score scale) is the smallest difference I consider practically important. For each contrast I report:

- **Direction:** an interval wholly above zero establishes a positive difference; one wholly below zero establishes a negative difference; otherwise its direction is unresolved.
- **Practical equivalence:** if the entire 95% interval lies strictly inside −0.05 to +0.05, I report "no difference larger than 5 points" at this interval level. A nonsignificant contrast alone is not equivalence. Direction and practical equivalence can both hold for a small effect.
- **Target superiority:** I support the original claim that ambiguous-only training beats both alternatives only when both mean advantages are at least 0.05 and both lower endpoints are above zero. This does not mean the intervals establish effects of at least five points.
- **Ruling out that target:** if either upper endpoint is below 0.05, I rule out a five-point advantage over that alternative at the reported interval level, not all possible positive effects.

I report both individual contrasts and all outcomes rather than select the easier baseline. These are individual 95% intervals, not simultaneous intervals for a full curriculum ranking.

## Precision and cost sizing

The completed 32-seed cohort is only a planning estimate of variability: paired standard deviations were 0.489400 for Random and 0.437695 for Difficulty. Under a normal approximation, `ceil((1.96 * sd / 0.05)^2)` needs 369 and 295 seeds, respectively, for a five-point interval half-width. The actual bootstrap width can differ, especially after changing thread count.

I use the new pilot only to measure runtime and memory feasibility, not to inspect scores for sizing. After its completion I choose a fixed N from 64, 96, 128, ..., 512, based on the remaining allowance and pilot runtime. With at most eight containers, the requested window is `ceil(1.25 * 3 * N * max_pilot_job_seconds / (8 * 60) + 5)` minutes, capped at 240 minutes. The upper compute estimate is window / 60 × 8 × ($0.047160 per CPU core-hour + $0.007992 per GiB-hour) × 1.10. I choose the largest N that fits after a $0.03 reserve, commit N and its predicted half-width `1.96 * sd / sqrt(N)`, and then run it. If N is below 369, I state that the budget does not permit the planning target for both contrasts. No sample size guarantees a definitive answer.

The previous runs used at most $0.8718 of the total $2 allowance, leaving $1.1282 before this pilot. All further runs stay within that remaining allowance and end before the experiment cutoff. Checkpoints stay in the Modal Volume `faultline-seed-comparison`; only hashes, locations and compressed raw evaluation JSON enter Git. No checkpoint release is published, and cohort weights are not downloaded.
