# Adaptive Bridge Eval Hardening

Date: `2026-04-24`

Primary artifact:

- `outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/paired_uncertainty.json`

## Milestone Frame

Milestone 1 remains frozen as a bounded positive result.

- internal LM-style strengths are preserved relative to frozen `v0.6.0`
- `PIQA` is recovered over both bridge baselines on point estimate
- `ARC-Easy` remains unresolved
- this is not a global replacement for frozen `v0.6.0`

## Check 1: Frozen Reference Integrity

Status: `passed`

Verified for seeds `42/43/44`:

- frozen `v0.6.0` token-wise checkpoints exist locally
- warm-start was recorded as `loaded`
- the completed eval loaded `frozen_v060_tokenwise` as a live comparison model
- every bounded task result file for every seed contains the frozen reference outputs

Interpretation:

- the 3-seed bounded comparison is valid
- the adaptive models were not compared against a missing or stubbed frozen reference

## Check 2: Internal Guardrail

Frozen-reference comparison only:

### `adaptive_bridge_moe` vs frozen `v0.6.0`

| Task | KL | NLL | Guardrail |
| --- | ---: | ---: | --- |
| development holdout | `0.239405` vs `0.254750` | `2.880880` vs `2.908419` | preserved |
| confirmation holdout | `0.244074` vs `0.261244` | `2.860653` vs `2.888155` | preserved |

### `adaptive_bridge_no_small` vs frozen `v0.6.0`

| Task | KL | NLL | Guardrail |
| --- | ---: | ---: | --- |
| development holdout | `0.266798` vs `0.254750` | `2.919023` vs `2.908419` | preserved under configured tolerances |
| confirmation holdout | `0.276183` vs `0.261244` | `2.902047` vs `2.888155` | preserved under configured tolerances |

Failure rule for future variants:

- if a future variant loses both confirmation KL and confirmation NLL versus both frozen `v0.6.0` and current `adaptive_bridge_moe`, treat it as failed before external interpretation

## Check 3: No-Small Separation

Same-run comparison only:

| Task | `adaptive_bridge_moe` vs `adaptive_bridge_no_small` | Result |
| --- | --- | --- |
| development holdout | KL `-0.027393`, NLL `-0.038144` | MoE better |
| confirmation holdout | KL `-0.032110`, NLL `-0.041393` | MoE better |
| LAMBADA | KL `-0.011312`, NLL `-0.008100` | MoE better |
| PIQA | accuracy `+0.018333`, bootstrap CI `[+0.006667, +0.031667]` | MoE clearly better |
| ARC-Easy | accuracy `+0.010000`, bootstrap CI `[-0.003333, +0.023333]` | MoE slightly better on point estimate only |

Interpretation:

- delegated small-model computation matters beyond the conditioning scaffold on all LM-style tasks
- it also matters on the key recovered task, `PIQA`
- the `ARC-Easy` gap against the no-small control is positive but weak

## Check 4: Expert-Usage Stability

Token-level gate statistics for `adaptive_bridge_moe`:

| Task | bridge | path B | path A | entropy | collapse | bridge var | path B var | path A var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development holdout | `0.467` | `0.239` | `0.294` | `0.967` | `0.001` | `0.0173` | `0.0162` | `0.0206` |
| confirmation holdout | `0.461` | `0.226` | `0.313` | `0.960` | `0.001` | `0.0190` | `0.0170` | `0.0222` |
| LAMBADA | `0.412` | `0.275` | `0.313` | `1.009` | `0.000` | `0.0127` | `0.0144` | `0.0192` |
| PIQA | `0.367` | `0.273` | `0.359` | `0.989` | `0.000` | `0.0220` | `0.0184` | `0.0245` |
| ARC-Easy | `0.400` | `0.289` | `0.311` | `0.937` | `0.039` | `0.0302` | `0.0278` | `0.0376` |

Token-level gate statistics for `adaptive_bridge_no_small`:

| Task | bridge | path B | path A | entropy | collapse |
| --- | ---: | ---: | ---: | ---: | ---: |
| development holdout | `0.025` | `0.305` | `0.670` | `0.284` | `0.673` |
| confirmation holdout | `0.026` | `0.287` | `0.688` | `0.254` | `0.709` |
| LAMBADA | `0.037` | `0.321` | `0.642` | `0.345` | `0.592` |
| PIQA | `0.065` | `0.281` | `0.654` | `0.324` | `0.616` |
| ARC-Easy | `0.065` | `0.286` | `0.649` | `0.259` | `0.714` |

Interpretation:

- the full adaptive MoE is stable:
  - high entropy
  - near-zero collapse on every task except a modest rise on `ARC-Easy`
  - all three experts remain active
- the no-small control is much less stable:
  - low entropy
  - heavy route collapse
  - bridge weight nearly disappears

This weakens the generic-MoE-capacity explanation.

## Check 5: Task-Conditional Pattern

Full adaptive MoE only.

Family means:

- LM-style tasks:
  - bridge `0.447`
  - delegated total `0.553`
  - path A `0.306`
  - path B `0.247`
- multichoice tasks:
  - bridge `0.384`
  - delegated total `0.616`
  - path A `0.335`
  - path B `0.281`

Per-task pattern:

- bridge usage is highest on the internal holdouts
- delegated usage rises on `PIQA` and remains elevated on `ARC-Easy`
- path A gains most on:
  - confirmation holdout
  - `PIQA`
- path B is relatively stronger on:
  - `LAMBADA`
  - `ARC-Easy`

Interpretation:

- bridge is used more on LM-style settings
- delegated paths are used more on the multichoice tasks, especially `PIQA`
- the two delegated paths are not redundant:
  - path A is the stronger `PIQA`-leaning complement
  - path B is the stronger `LAMBADA` / `ARC-Easy` complement

## Check 6: Same-Run vs Frozen-Reference Distinction

Keep these separate:

Frozen-reference comparisons:

- `adaptive_bridge_moe` vs `frozen_v060_tokenwise`
- `adaptive_bridge_no_small` vs `frozen_v060_tokenwise`

Same-run comparisons:

- `adaptive_bridge_moe` vs `adaptive_bridge_no_small`
- `adaptive_bridge_moe` vs `bridge_only_strong`
- `adaptive_bridge_moe` vs `bridge_only_param_matched`

This note uses that separation consistently.

## Check 7: Unresolved Weakness

`ARC-Easy` remains unresolved.

- point estimate:
  - `adaptive_bridge_moe`: `0.805000`
  - `bridge_only_strong`: `0.813333`
  - `bridge_only_param_matched`: `0.818333`
- uncertainty vs bridge baselines is not favorable:
  - vs `bridge_only_param_matched`: delta `-0.013333`, CI `[-0.028333, 0.000000]`
  - vs `bridge_only_strong`: delta `-0.008333`, CI `[-0.025000, +0.008333]`

Do not treat `ARC-Easy` as recovered.

## Phase 1: Paired Uncertainty

### Robust positives

Frozen-reference comparison:

- development holdout:
  - KL delta `-0.015345`, CI `[-0.017127, -0.013713]`
  - NLL delta `-0.027539`, CI `[-0.032853, -0.022443]`
- confirmation holdout:
  - KL delta `-0.017170`, CI `[-0.019210, -0.015226]`
  - NLL delta `-0.027502`, CI `[-0.032865, -0.022319]`
- LAMBADA:
  - KL delta `-0.007184`, CI `[-0.007704, -0.006652]`
  - NLL delta `-0.004145`, CI `[-0.006290, -0.002005]`
- PIQA:
  - accuracy delta `+0.021667`, CI `[+0.008333, +0.036667]`

Same-run comparison:

- `adaptive_bridge_moe` beats `adaptive_bridge_no_small` with clear paired-bootstrap support on:
  - development holdout
  - confirmation holdout
  - LAMBADA
  - PIQA

### Point-estimate-only positives

`PIQA` vs bridge baselines is still only point-estimate positive:

- vs `bridge_only_param_matched`:
  - accuracy delta `+0.006667`
  - CI `[-0.006667, +0.020000]`
  - probability of improvement `0.816`
- vs `bridge_only_strong`:
  - accuracy delta `+0.008333`
  - CI `[-0.005000, +0.023333]`
  - probability of improvement `0.846`

Interpretation:

- the key positive task is stable versus frozen `v0.6.0` and the no-small control
- it is not yet uncertainty-robust versus the bridge baselines

## Hardening Conclusion

The bounded milestone remains genuinely positive, but only part of the claim is uncertainty-robust.

Most defensible statement:

- internal preservation is robust
- LAMBADA preservation is robust
- `PIQA` recovery is real versus frozen `v0.6.0` and the no-small control
- `PIQA` superiority over the bridge baselines is still point-estimate positive, not yet confidence-robust
- `ARC-Easy` remains unresolved
