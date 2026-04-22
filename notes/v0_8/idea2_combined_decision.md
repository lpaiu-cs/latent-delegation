# Idea 2 Combined Decision

## Branch Status

- `v0.6.0` remains the frozen current best release.
- Idea 5 remains frozen as a discovery branch only.
- Idea 2 in this task was limited to discovery-first sublayer attribution on the frozen `v0.6.0` token-wise model.

## Attribution Summary

- Full token-wise baseline remains best on the primary output metrics on both holdouts:
  - main holdout: KL=`0.255739`, NLL=`2.980182`
  - fresh holdout: KL=`0.248886`, NLL=`3.185004`
- Suppressing delegated attention materially hurts performance:
  - main holdout: `+0.103797` KL, `+0.218670` NLL vs full token-wise
  - fresh holdout: `+0.109496` KL, `+0.224608` NLL
- Suppressing delegated MLP hurts even more:
  - main holdout: `+0.182743` KL, `+0.350897` NLL
  - fresh holdout: `+0.182872` KL, `+0.379610` NLL
- Path A is more sensitive than path B for both subcomponents, with path A MLP suppression producing the largest degradation.

## Interpretation

1. Delegated attention is necessary for the current `v0.6.0` gain.
2. Delegated MLP is also necessary, and it is the stronger contributor under KL/NLL.
3. The answer is stable across both holdout policies.
4. The signal is directional toward MLP importance, but not directional enough to justify an MLP-only delegated model in this task because attention ablation is also materially harmful.

## Model-Building Decision

- No bounded Idea 2 variant was run.
- Reason: the task rule for this phase was to stop before model-building if both subcomponents are clearly needed.
- That stop rule applies here. The attribution supports a mixed-function story rather than a clean single-sublayer simplification.

## Recommendation

`Stop and preserve v0.6.0`
