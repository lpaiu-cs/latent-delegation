# Adaptive Bridge Route Ablation

Date: `2026-04-24`

Primary artifact:

- `outputs/adaptive_bridge/route_ablation/results.json`

## Setup

Inference-only counterfactual ablations were run on the trained `adaptive_bridge_moe` checkpoints for seeds `42/43/44`.

Evaluated tasks:

- confirmation holdout
- LAMBADA
- PIQA
- ARC-Easy

Route policies:

1. full adaptive MoE
2. bridge only forced
3. bridge + path A only
4. bridge + path B only
5. delegated paths only
6. adaptive no-small control

Important scope note:

- these are same-run route counterfactuals
- they are not frozen-reference comparisons
- they test contribution structure, not new trained variants

## Summary

### Confirmation Holdout

| Variant | KL | NLL |
| --- | ---: | ---: |
| full adaptive MoE | `0.244074` | `2.860653` |
| adaptive no-small | `0.276183` | `2.902047` |
| delegated paths only | `0.311770` | `2.912739` |
| bridge + path A only | `0.323754` | `2.956094` |
| bridge + path B only | `0.334605` | `2.957979` |
| bridge only forced | `0.398817` | `3.135439` |

### LAMBADA

| Variant | KL | NLL |
| --- | ---: | ---: |
| full adaptive MoE | `0.241733` | `3.381540` |
| adaptive no-small | `0.253045` | `3.389639` |
| bridge + path B only | `0.291546` | `3.433614` |
| delegated paths only | `0.300530` | `3.418282` |
| bridge only forced | `0.305291` | `3.506085` |
| bridge + path A only | `0.329694` | `3.488761` |

### PIQA

| Variant | Accuracy |
| --- | ---: |
| full adaptive MoE | `0.785000` |
| bridge + path A only | `0.783333` |
| bridge only forced | `0.780000` |
| bridge + path B only | `0.778333` |
| adaptive no-small | `0.766667` |
| delegated paths only | `0.758333` |

### ARC-Easy

| Variant | Accuracy |
| --- | ---: |
| bridge + path B only | `0.808333` |
| full adaptive MoE | `0.805000` |
| bridge + path A only | `0.803333` |
| bridge only forced | `0.800000` |
| adaptive no-small | `0.795000` |
| delegated paths only | `0.791667` |

## Interpretation

### Bridge Is Not Sufficient For LM-Style Preservation

Bridge-only forced is clearly weaker than the full MoE on:

- confirmation holdout:
  - KL `+0.154743`
  - NLL `+0.274786`
- LAMBADA:
  - KL `+0.063557`
  - NLL `+0.124546`

So the current internal and LAMBADA preservation is not just the bridge acting alone.

### Delegated-Only Mixing Is Also Not The Answer

Delegated-paths-only is weaker than the full MoE on every evaluated task:

- confirmation holdout:
  - KL `+0.067696`
  - NLL `+0.052086`
- LAMBADA:
  - KL `+0.058796`
  - NLL `+0.036742`
- PIQA:
  - accuracy `-0.026667`
- ARC-Easy:
  - accuracy `-0.013333`

This argues against the gain being mostly direct delegated production without a strong bridge contribution.

### PIQA Looks Bridge-Centered

The recovered task, `PIQA`, remains close to the full MoE under bridge-centered restrictions:

- bridge only forced:
  - accuracy `0.780000`
  - delta vs full `-0.005000`
- bridge + path A only:
  - accuracy `0.783333`
  - delta vs full `-0.001667`
- bridge + path B only:
  - accuracy `0.778333`
  - delta vs full `-0.006667`

Meanwhile:

- adaptive no-small: `0.766667`
- delegated paths only: `0.758333`

Interpretation:

- bridge is the main correction path on `PIQA`
- path A is the better specialized complement on `PIQA`
- the delegated paths alone do not explain the recovered gain

### Path Specialization Is Task-Dependent

Path A is stronger on:

- `PIQA`
- confirmation holdout

Path B is stronger on:

- `LAMBADA`
- `ARC-Easy`

This does not support a single delegated path dominating every external behavior.

## Main Diagnostic Answer

The current adaptive gain does not look like generic MoE capacity or unstable direct residual mixing.

Most plausible reading from the ablations:

- the bridge is a core correction path
- delegated paths act as specialized complements
- but in the current model they still matter materially for LM-style preservation

So the data support a bridge-centered story directionally, but they do not yet prove that delegated paths can be reduced to pure conditioning without loss.

## Unresolved Weakness

`ARC-Easy` remains unresolved.

- the best route ablation on `ARC-Easy` is `bridge + path B only` at `0.808333`
- that is still below the bridge baselines from the main eval:
  - `bridge_only_strong`: `0.813333`
  - `bridge_only_param_matched`: `0.818333`
