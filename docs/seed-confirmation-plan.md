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

## Fixed cohort, committed before training

I increased the total compute allowance to **$5.90** before starting the independent confirmation. The excluded one-thread pilot jobs took 163.240666, 177.371425 and 272.719381 seconds and cost at most $0.0390. A two-minute capacity check started **32 distinct containers**, all within 11.70 seconds of its client's start, and cost at most $0.0409. The earlier runs, this pilot and the capacity check together used at most **$0.9517**.

I amend the runtime-sizing calculation before the independent cohort starts to `ceil(1.25 * N * sum_pilot_job_seconds / (32 * 60) + 2)`. The cohort has exactly N jobs in each arm, so I use the three durations' sum rather than assign the slowest arm's duration to every job. The successful startup check supports a two-minute startup allowance and 32-container concurrency. I consider all integer sizes from 64 to 512, retaining the 1.25× runtime margin, 240-minute cap and $0.03 reserve. I do not use pilot scores in this choice.

This fixes **375 matched seeds, 500–874**, or **1,125 trained policies**, in a **152-minute** window. The booking is **$4.91808768**, and the total including all earlier runs is at most **$5.86978768** before the spare reserve. A 376-seed cohort needs 153 minutes and would not fit that reserve. The predicted 95% half-widths are **4.9534 points** versus Random and **4.4301 points** versus Difficulty, meeting the planning target for both. These predictions are not the observed confidence intervals and do not guarantee practical equivalence or superiority.

## Observed outcome after training

All 1,125 policies completed. The [independent analysis](../artifacts/results/seed-confirmation-analysis.json) gives Epistemic−Random +10.1042 points, 95% interval [4.6755, 15.5323], and Epistemic−Difficulty −5.9333 points, interval [−10.2929, −1.6661]. Both directions are resolved in opposite directions; the target advantage over both alternatives is ruled out. Neither contrast establishes practical equivalence within ±5 points.

The observed interval half-widths are 5.4284 points versus Random and 4.3134 versus Difficulty. The Random precision prediction missed the five-point target. I report that miss without extending the already-observed cohort. The [compute estimates](../artifacts/results/seed-compute-costs.json) total $4.2829, including $3.3312 for this confirmation; these are not invoice totals.

