# Idea 4 Static Mixture Report

## Setup

- Idea 4 scope: static two-path mixture only
- fixed large removed window: `24..27`
- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`
- backbones: frozen real Gemma 2 9B / 2B
- training objective: same output-aware Stage B family as Phase 1
- warm start policy: each path reused the matching confirmed Phase 1 checkpoint, with the Phase 1 scalar gate absorbed into the return adapter before Idea 4 training
- new routing freedom: one global 2-logit softmax only

## Coarse Screen

- budget: 1 seed, 100 Stage B steps
- static mixture hidden diagnostics:
  - hidden MSE `8.941406`
  - cosine `0.862732`
  - delta norm mean `105.319206`
  - weights: path B `0.511719`, path A `0.488281`
- static mixture no-small hidden diagnostics:
  - hidden MSE `13.891602`
  - cosine `0.852371`
  - delta norm mean `65.427504`
  - weights: path B `0.494141`, path A `0.503906`
- bridge controls on the same coarse run were materially worse in hidden recovery:
  - `bridge_only`: hidden MSE `15.253906`, cosine `0.826324`
  - `bridge_only_param_matched`: hidden MSE `15.257812`, cosine `0.827118`
- coarse hidden result: the static mixture already dominated the no-small and bridge controls, so confirmation was justified on output grounds.

## Confirmation

- budget: 3 seeds, 200 Stage B steps
- static mixture hidden summary:
  - hidden MSE `7.148438 ± 0.079378`
  - cosine `0.868975 ± 0.000931`
  - delta norm mean `129.206298 ± 1.075897`
  - weight path B `0.519531 ± 0.000000`
  - weight path A `0.481120 ± 0.001128`
  - path B delta norm mean `154.442175 ± 5.640720`
  - path A delta norm mean `130.675042 ± 2.309828`
- static mixture no-small summary:
  - hidden MSE `12.299154 ± 0.106002`
  - cosine `0.861201 ± 0.000487`
  - delta norm mean `82.279780 ± 0.712389`
  - weights stayed near uniform: path B `0.500000`, path A `0.500651`
- bridge summaries:
  - `bridge_only`: hidden MSE `13.438151 ± 0.222119`, cosine `0.845490 ± 0.001836`
  - `bridge_only_param_matched`: hidden MSE `12.670898 ± 0.090082`, cosine `0.850215 ± 0.000848`

## Interpretation

- The hidden-space gain from the static mixture was large and stable across all 3 seeds.
- The no-small control also improved over the single-path no-small behavior, so mixing interface routes is not useless by itself.
- The actual delegated-computation mixture still recovered substantially more hidden structure than the no-small mixture, which is the important diagnostic for whether the two-path mixture is doing more than routing around the interface.
- The learned static weights stayed close to balanced rather than collapsing onto one path. The mixture leaned slightly toward path B, but not enough to collapse back into a hard single-window choice.
- Output-level metrics remain the primary evidence. This report should be read as support for the output-probe result, not as a replacement for it.
