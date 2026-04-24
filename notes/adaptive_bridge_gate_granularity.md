# Adaptive Bridge Gate Granularity

Date: `2026-04-25`

Primary artifact:

- `outputs/adaptive_bridge/gate_granularity/results.json`

## Scope

This is an inference-only same-run audit on the already-trained `42/43/44` checkpoints.

Tasks:

- confirmation holdout
- `LAMBADA`
- `PIQA`
- `ARC-Easy`

Variants:

1. full token-wise gate
2. sequence-mean gate per sample
3. global-mean gate per task slice and seed
4. bridge-only forced
5. bridge + path A only
6. bridge + path B only
7. delegated paths only
8. `adaptive_bridge_no_small`

## Same-Run vs Frozen-Reference Separation

This note is same-run only.

- every comparison here is an inference-only counterfactual on the trained adaptive checkpoints
- frozen `v0.6.0` comparisons are intentionally excluded
- frozen-reference conclusions stay in the main eval artifacts

## Granularity Definition

The two simplified gating variants are:

- sequence-mean gate:
  - average the full token-wise gate weights over valid tokens within each sample
  - reuse that single mixture over the whole sample
- global-mean gate:
  - average the full token-wise gate weights over every valid token in the task slice for one seed
  - reuse that single mixture over the whole task slice

So this audit directly asks whether token-level variation is necessary for the current gain.

## Confirmation Holdout

Same-run comparison only.

| Variant | KL | NLL |
| --- | ---: | ---: |
| full token-wise gate | `0.244074` | `2.860653` |
| sequence-mean gate | `0.256850` | `2.905120` |
| global-mean gate | `0.259980` | `2.923603` |
| adaptive no-small | `0.276183` | `2.902047` |
| bridge only forced | `0.398817` | `3.135439` |

Interpretation:

- sequence-mean and global-mean keep some KL benefit over the no-small control
- neither preserves the full token-wise NLL gain
- both surrender most of the full LM-style advantage

## LAMBADA

Same-run comparison only.

| Variant | KL | NLL |
| --- | ---: | ---: |
| full token-wise gate | `0.241733` | `3.381540` |
| sequence-mean gate | `0.245771` | `3.393293` |
| global-mean gate | `0.245553` | `3.393989` |
| adaptive no-small | `0.253045` | `3.389639` |
| bridge + path B only | `0.291546` | `3.433614` |

Interpretation:

- sequence/global averaging still help on KL relative to the no-small control
- on NLL they fall behind the no-small control
- token-wise routing remains the only setting that keeps both KL and NLL clearly best

## PIQA

Same-run comparison only.

| Variant | Accuracy |
| --- | ---: |
| full token-wise gate | `0.785000` |
| bridge + path A only | `0.783333` |
| bridge only forced | `0.780000` |
| sequence-mean gate | `0.775000` |
| global-mean gate | `0.775000` |
| adaptive no-small | `0.766667` |
| delegated paths only | `0.758333` |

Relative to `adaptive_bridge_no_small`:

- full token-wise gate gain: `+0.018333`
- sequence-mean gate gain: `+0.008333`
- global-mean gate gain: `+0.008333`

Interpretation:

- sequence/global averaging preserve only about `45%` of the full token-wise PIQA gain over the no-small control
- both simpler granularities also fall below the stronger bridge-centered route restrictions
- this is not “most of the current gain”

## ARC-Easy

Same-run comparison only.

| Variant | Accuracy |
| --- | ---: |
| bridge + path B only | `0.808333` |
| global-mean gate | `0.806667` |
| full token-wise gate | `0.805000` |
| bridge + path A only | `0.803333` |
| sequence-mean gate | `0.800000` |
| bridge only forced | `0.800000` |
| adaptive no-small | `0.795000` |

Interpretation:

- a simpler granularity can be competitive on this bounded slice
- but this does not resolve `ARC-Easy`
- the task is still below the bridge baselines in the main eval artifacts

## What The Granularity Audit Says

1. Token-wise routing is still materially important for the current active reference.
2. Sequence-mean gating does not preserve most of the full token-wise gain on confirmation, `LAMBADA`, or `PIQA`.
3. Global-mean gating is competitive only on the already-unresolved `ARC-Easy` slice.
4. The simpler granularity variants look more like partial explanations than drop-in replacements.

## Practical Reading

The current result cannot be explained as “any low-capacity sequence-level conditioner would have done the same job.”

Most defensible reading:

- bridge-centered structure still looks directionally right
- delegated paths are not acting as pure noise
- but token-level routing is still doing real work on the preserved LM-style strengths and the bounded `PIQA` win

So a bridge-conditioned simplification is still a hypothesis, not yet the active replacement.
