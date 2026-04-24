# Adaptive Bridge Phase 2 Decision

Date: `2026-04-24`

Inputs:

- `outputs/adaptive_bridge/real_seed42_43_44_warm_start/eval/paired_uncertainty.json`
- `outputs/adaptive_bridge/route_ablation/results.json`
- `notes/adaptive_bridge_eval_hardening.md`
- `notes/adaptive_bridge_route_ablation.md`

## 1. Is the current adaptive-bridge result robust under uncertainty, or only point-estimate positive?

Answer:

- internal holdout preservation is uncertainty-robust
- LAMBADA preservation is uncertainty-robust
- `PIQA` is uncertainty-robust versus frozen `v0.6.0` and the no-small control
- `PIQA` is still only point-estimate positive versus the bridge baselines

So the current result is not uniformly robust. The strongest external recovery claim is still only point-estimate positive against the actual bridge baselines.

## 2. Does `adaptive_bridge_moe` clearly beat `adaptive_bridge_no_small` on the key recovered task(s)?

Answer: `yes`

Evidence:

- `PIQA`: `+0.018333` accuracy, bootstrap CI `[+0.006667, +0.031667]`
- development holdout: better KL and NLL
- confirmation holdout: better KL and NLL
- LAMBADA: better KL and NLL

This is strong evidence that delegated small-model computation matters beyond the conditioning scaffold.

## 3. Do the route-ablation diagnostics support the hypothesis that bridge is the core correction path and delegated paths are acting as conditioners or specialized complements?

Answer: `directionally yes, but not as a completed proof`

Why:

- delegated-paths-only underperforms the full MoE on every ablated task
- bridge-only forced retains most of the `PIQA` gain
- `PIQA` remains near full performance under `bridge + path A only`
- `ARC-Easy` is strongest under `bridge + path B only`

But:

- bridge-only forced is clearly too weak on confirmation and LAMBADA
- current delegated paths still contribute real delta content, not only conditioning

So the diagnostics support a bridge-centered interpretation, but not a fully validated conditioner-only claim yet.

## 4. Is path A or path B carrying most of the external recovery?

Answer:

- path A carries more of the `PIQA` recovery
- path B carries more of the `LAMBADA` / `ARC-Easy` complement

There is no single delegated path that dominates every external behavior.

## 5. Does ARC-Easy remain the main unresolved task?

Answer: `yes`

Reason:

- main 3-seed eval:
  - `adaptive_bridge_moe`: `0.805000`
  - `bridge_only_strong`: `0.813333`
  - `bridge_only_param_matched`: `0.818333`
- route ablation:
  - even the best ablation, `bridge + path B only`, reaches only `0.808333`

`ARC-Easy` remains the clearest unresolved weakness after milestone 1.

## 6. Is it justified to move to a bridge-conditioned simplification?

Answer: `not yet`

Reason:

- the main external win over bridge baselines is still only point-estimate positive under uncertainty
- route ablations suggest the right simplification direction, but they do not yet show that delegated paths can be safely reduced to conditioning-only without harming the preserved LM-style strengths
- bridge-only forced degrades too much on confirmation and LAMBADA to justify an immediate architecture jump

## Recommendation

`Stay on current adaptive-bridge MoE and gather more evidence`

Rationale:

- milestone 1 is genuinely positive
- the mechanism diagnostics are informative
- but the evidence is still not strong enough to justify replacing direct delegated contribution with a bridge-conditioned simplification immediately

Most disciplined next move inside the current branch:

- keep the current adaptive-bridge MoE as the active reference
- if more work continues, gather one more bounded round of evidence on the bridge-baseline gap before changing the architecture
- keep `ARC-Easy` as the explicit unresolved target
