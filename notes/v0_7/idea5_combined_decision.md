# Idea 5 Combined Decision

## Scope

- `v0.6.0` remains the frozen current best result.
- Idea 5 in this task was limited to local monotone-alignment discovery around large `22..30` and small `13..20`.
- The only empirical follow-up allowed here was one bounded candidate derived from the monotone solver.

## Discovery Result

- The monotone solver recovered a clear local corridor rather than a single hard aligned window.
- The top path was:
  - `22..22 -> 13..14`
  - `23..24 -> 15..16`
  - `25..27 -> 17..18`
  - `28..30 -> 19..20`
- Around the successful splice `24..27`, that path compressed to the small-layer union `15..18`.
- Under the same structural proxy, the direct window ranking was:
  - `24..27 -> 16..18` with proxy cost `0.237026`
  - `24..27 -> 14..19` with proxy cost `0.261984`
  - `24..27 -> 15..18` with proxy cost `0.270362`
  - `25..29 -> 15..19` with proxy cost `0.307316`
  - legacy `24..29 -> 14..19` with proxy cost `0.315038`

## Minimal Empirical Check

- One real Gemma pilot candidate was run with candidate-specific Stage A, output-aware Stage B, and the existing Phase 1A output probe:
  - derived candidate: `24..27 -> 15..18`
  - artifact root: `artifacts/v0_7/idea5_discovery/empirical_check/`
- Pilot output metrics for the derived candidate:
  - KL `0.424089`
  - NLL `2.944444`
  - PPL `19.000104`
  - top-1 `0.713889`
  - top-5 `0.712222`
- Versus the `v0.6.0` token-wise Idea 4 main-holdout result:
  - KL worsened by `+0.168349`
  - NLL improved by `-0.035738`
  - PPL improved by `-0.763656`
  - top-1 worsened by `-0.049462`
  - top-5 worsened by `-0.032046`
- Versus the confirmed static mixture:
  - KL worsened by `+0.156993`
  - NLL improved by `-0.055994`
  - top-1 worsened by `-0.048120`
- Versus `bridge_only` on the main holdout:
  - KL worsened by `+0.135641`
  - NLL improved by `-0.127606`
  - top-1 worsened by `-0.041600`

## Answers

1. Does monotone alignment discovery support the current theoretical story?
Yes. The solver recovered a low-cost monotone corridor around the validated shortlist and favored multi-segment asymmetric correspondence rather than a single hard local match.

2. Does it explain the success of the two-path Idea 4 shortlist better than a hard single-window hypothesis?
Yes. The best path places `24..27` inside a broader monotone corridor whose local small-side support spans `15..18`, with the two successful shortlisted windows appearing as adjacent coarse samples of that corridor.

3. Is there one minimal Idea 5 candidate worth testing?
Yes. The cleanest derived single-path compression is `24..27 -> 15..18`.

4. If tested, does it match or beat the `v0.6.0` token-wise Idea 4 model?
No. The derived single-path candidate improved NLL and PPL, but it lost badly on KL and on top-1/top-5 agreement relative to token-wise Idea 4. Under the project's KL-first output ranking, that is not competitive enough.

5. Should the project proceed with Idea 5 model-building, or stop and preserve Idea 4 as the best branch?
Stop here. The discovery evidence strengthens the theoretical monotone-alignment story, but the one bounded empirical check did not support expanding Idea 5 model-building beyond discovery in this task.

## Next Branch

- Idea 5 strengthened the monotone-corridor hypothesis.
- Its one bounded empirical candidate was not competitive with `v0.6.0`.
- The next branch is Idea 2, not more Idea 5 model-building.

Stop Idea 5 and preserve v0.6.0 as the current best result
