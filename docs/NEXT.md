# Next: make selective probing observable

The current result supports learned diagnostic routines, not selective investigation. The policy sees
a binary cue but not whether that cue is reliable. It also lacks the reward coefficients. A fixed
`advance → inspect → repair` routine can therefore succeed without deciding whether an experiment
is worth its cost. Changing hidden costs without changing policy inputs is not a test of adaptation.
See the [observation interface](../src/faultline/training/rl_env.py) and the
[behavioral controls](../artifacts/results/behavioral-controls-v1-analysis.json).

## Proposed observation and task changes

Keep the two-fault simulator and operational reward. Add public episode-level inputs to the graph/GRU
policy before its first action:

- **Cue reliability:** the announced probability that the cue matches the hidden fault, sampled
  independently of that fault. Start with random versus perfectly reliable cues; later include
  intermediate reliability. This is a sensor specification, not the hidden cause or a label of the
  correct repair.
- **Reward terms:** expose the same production, advance/time, inspection, repair and false-repair
  coefficients used by the environment, with fixed documented normalization. Include the recovery
  horizon if it varies, since it affects the value of restoring production.

Version the policy observation and checkpoint schema; old checkpoints must not silently acquire new
input semantics. Do not edit the completed study or its split definitions.

Construct crossed reliability × probe-cost blocks using the same base factory, hidden-fault schedule,
and cue balance. Choose costs using the exact oracle so that the dataset includes both positive and
negative net value of probing. A reliable cue with a costly inspection should favor immediate repair;
an ambiguous cue with a cheap, informative inspection should favor investigation. High-cost ambiguous
tasks can rationally favor a guess rather than diagnosis. Verify these regimes with the oracle, not
with a hand-picked numerical threshold.

## Comparison and decision

First test the interface and oracle regimes on development data. Then compare the existing three
curricula at matched decision-step budgets and paired training seeds, with a fixed-probe policy,
cue-only repair policy and exact oracle as behavior baselines. Preserve repair availability and the
existing action masks across arms. Ensure every learning arm has some exposure to both optimal
probe and optimal no-probe regimes; an all-ambiguous curriculum may otherwise be unable to learn
cue-conditioned abstention. Report that curriculum change explicitly rather than relabeling it as
identical to the old comparison.

Measure per-seed return/regret relative to the oracle, probe rate in each reliability/cost cell,
correct-repair rate and paired uncertainty. Test whether the same policy changes its probing decision
when only public reliability or costs change. Include an ablation hiding those new inputs and retain
counterfactual evidence swaps for episodes that do probe. Success means both good repair choices and
less unnecessary investigation in oracle no-probe regimes, not a higher overall inspection rate.
Set the minimum relevant improvement before the new multi-seed runs. Keep the existing test split
unused during interface development; define a separate versioned evaluation protocol before opening
any held-out test data.

## Resources and sequence

This is a proposal, not a run completed in this PR. Paid compute: $0. Use local CPU, `nice -n 19`,
small PyTorch models, one training process at a time and at most two CPU threads on the shared laptop.
Check available memory first; skip training if less than 1.5 GB is available or resident memory would
exceed 1.5 GB. Keep each run below two hours.

Planning estimate: one to two days to implement/version observations, oracle regime tests and the
input ablation. The original [24 trace files](../artifacts/results/small-kill-v1-analysis.json)
record roughly 32–60 seconds of training each with six threads. That does not establish a runtime for
the new experiment. Allow up to two hours for a sequential pilot/comparison session at the old target
budget, measure pilot runtime and memory, and revise the estimate before launching the full study.

## Checkpoints: use a Release for future downloads

The current checkout contains 30 `policy.pt` files totaling 108,356,640 bytes (about 103 MiB), including
all study and pilot models. Their paths and SHA-256 hashes are recorded in the committed run results
and manifests. They remain untouched in this PR.

Proposed migration, subject to the owner's choice:

1. Tag the existing study commit and publish a GitHub Release with checkpoint assets (individual
   files or a run-ID-preserving archive), a SHA-256 inventory, source commit, license and restore
   instructions. Keep JSON traces, manifests, protocols and figures in Git.
2. Download the public assets once and check every hash against the existing manifests, then confirm
   restoration to the original `artifacts/runs/<run-id>/policy.pt` paths supports the current tools.
3. Only after public downloads work, remove tracked checkpoint binaries in a **new commit**, ignore
   future checkpoint files and document the download/restore command. No force push, history rewrite,
   deletion of result evidence or migration in this PR.

Removing binaries in a later commit will not make a full historical clone smaller. Release assets
avoid growth from future runs; shallow clones can avoid old binary history. Do not advertise this as
shrinking existing Git history.

Owner decisions: approve the revised observation/curriculum comparison and its evaluation threshold;
choose the Release/tag naming and whether to remove checkpoint files from future checkouts after
verified asset publication.
