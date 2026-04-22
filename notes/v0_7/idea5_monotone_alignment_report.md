# Idea 5 Monotone Alignment Report

## Scope

- Local discovery region on the large side: `22..30`.
- Local discovery region on the small side: `13..20`.
- Cost components: stage-signature distance, hidden-alignment proxy, logit-disruption proxy, and a disabled-by-default output-anchor proxy.
- Solver moves: `1:1`, `2:1`, `1:2`, `2:2`, and `3:2`.

## Top Path

- Total cost: `1.128269`
- Segment count: `4`
- Best overlap with the confirmed Phase 1 shortlist around `24..27`: `24..27 -> 16..18` at overlap `0.750`.

### Segments

1. `22..22 -> 13..14` | combined=`0.286616` | stage=`3.006732` | hidden=`2.345761` | logit=`1.880915`
2. `23..24 -> 15..16` | combined=`0.273830` | stage=`3.034798` | hidden=`2.734476` | logit=`1.316298`
3. `25..27 -> 17..18` | combined=`0.288444` | stage=`2.994608` | hidden=`1.800267` | logit=`2.393056`
4. `28..30 -> 19..20` | combined=`0.279379` | stage=`2.951958` | hidden=`1.687134` | logit=`2.422320`

## Window Diagnostics

- `24..27 -> 16..18` | proxy=`0.237026` | stage=`2.730391` | hidden=`1.908594` | logit=`1.952512`
- `24..27 -> 14..19` | proxy=`0.261984` | stage=`2.858483` | hidden=`1.989775` | logit=`2.052247`
- `24..27 -> 15..18` | proxy=`0.270362` | stage=`2.917509` | hidden=`2.237339` | logit=`1.872477`
- `25..29 -> 15..19` | proxy=`0.307316` | stage=`3.099865` | hidden=`2.290119` | logit=`2.089143`
- `24..29 -> 14..19` | proxy=`0.315038` | stage=`3.139670` | hidden=`2.319163` | logit=`2.116368`

## Answers

1. Does the monotone solver naturally recover the successful two-path shortlist region? Yes.
2. Does it suggest that the two shortlisted windows are adjacent samples from a broader low-cost corridor? Yes.
3. Does it suggest asymmetric mapping pressure or multi-segment local correspondence? Yes.
4. Does it make the old `24..29 -> 14..19` split look structurally implausible in a more principled way? Yes.
5. Does it justify building an Idea 5 model at all? Yes, in a bounded follow-up.

## Minimal Derived Candidate

- Proposed single-path compression of the top monotone path around the successful splice: `24..27 -> 15..18`.
- This is a bounded follow-up candidate only. It was not promoted to a new architecture in this discovery run.