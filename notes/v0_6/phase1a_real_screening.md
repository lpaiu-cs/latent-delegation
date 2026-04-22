# Phase 1A Real Screening

## Protocol

- real path only: Gemma 2 9B teacher and Gemma 2 2B delegated block
- Windows-native execution only: `py -3.12 -m ...` and PowerShell runners
- candidate-specific Stage A rerun for every screened candidate; no Stage A checkpoint reuse across mismatched windows
- candidate-specific output-aware Stage B for every candidate
- output probe on the matching held-out split
- comparison set per candidate: `skip_only`, `hybrid_no_small`, `hybrid`, `bridge_only`, `bridge_only_param_matched`
- bridge controls were regenerated inside the matching candidate-specific Stage B run; no cross-window bridge reuse
- ranking order: KL to teacher, then NLL/PPL, then `hybrid > hybrid_no_small`, then gap to bridge controls

## Coarse Screening

One seed, lightweight Stage A and Stage B budgets, then output probe.

| candidate | mapping | KL | NLL | PPL | top-1 | top-5 | dKL vs `hybrid_no_small` | dNLL vs `hybrid_no_small` | dKL vs `bridge_only` | dNLL vs `bridge_only` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | `24..29 -> 14..19` | 0.725030 | 3.425046 | 30.724053 | 0.652812 | 0.630746 | -0.241896 | -0.324649 | -0.080399 | -0.090312 |
| candidate A | `24..27 -> 16..18` | 0.328102 | 3.032549 | 20.750055 | 0.761002 | 0.728790 | -0.081156 | -0.152124 | -0.031139 | -0.016886 |
| candidate B | `24..27 -> 14..19` | 0.327293 | 3.024373 | 20.581106 | 0.761614 | 0.728056 | -0.083045 | -0.165037 | -0.036994 | -0.036675 |
| candidate C | `25..29 -> 15..19` | 0.602056 | 3.300886 | 27.136680 | 0.686430 | 0.663203 | -0.149995 | -0.234337 | -0.054670 | -0.051956 |

Coarse-screen rank:

1. `24..27 -> 14..19`
2. `24..27 -> 16..18`
3. `25..29 -> 15..19`
4. legacy `24..29 -> 14..19`

The coarse screen was decisive enough to keep only candidates A and B for 3-seed confirmation. Both dominated the legacy split on every primary output metric.

## Confirmation

Top 2 only, 3 seeds, standard pilot budget, then output probe again.

| candidate | mapping | KL mean ± sd | NLL mean ± sd | PPL mean ± sd | top-1 mean | top-5 mean | dKL vs `hybrid_no_small` | dNLL vs `hybrid_no_small` | dKL vs `bridge_only` | dNLL vs `bridge_only` |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate A | `24..27 -> 16..18` | 0.282215 ± 0.016190 | 3.074461 ± 0.093456 | 21.700679 ± 1.998012 | 0.756979 | 0.739941 | -0.018538 | -0.069355 | 0.000773 | 0.013718 |
| candidate B | `24..27 -> 14..19` | 0.281641 ± 0.017965 | 3.078029 ± 0.095159 | 21.780681 ± 2.048019 | 0.755546 | 0.740074 | -0.017463 | -0.057073 | -0.000492 | 0.017439 |

Confirmation interpretation:

- Both confirmed candidates preserved `hybrid > hybrid_no_small` on KL and NLL in all 3 seeds.
- Both confirmed candidates preserved `hybrid > skip_only` in all 3 seeds.
- Neither confirmed candidate produced a full primary-metric win over `bridge_only` or `bridge_only_param_matched`.
- `24..27 -> 14..19` kept the best mean KL and the best mean top-5 overlap.
- `24..27 -> 16..18` kept the best mean NLL, best mean PPL, and best mean top-1 agreement.
- Under the stated screening rule, the nominal confirmation rank is still `24..27 -> 14..19` first and `24..27 -> 16..18` second, but the gap is effectively a tie at this budget.

## Secondary Diagnostics

- `24..27 -> 14..19` hybrid hidden diagnostics: hidden MSE `13.606445`, cosine `0.851440`, gate `0.069336`, delta norm `1086.875485`
- `24..27 -> 16..18` hybrid hidden diagnostics: hidden MSE `13.958008`, cosine `0.849726`, gate `0.069824`, delta norm `1066.878764`
- Both confirmed hybrids improved hidden MSE and cosine over `hybrid_no_small` and `skip_only`.
- Gates stayed near the intended small-value regime rather than running away.
- Delta norms were large but stayed on the same scale as the bridge controls, so the output gains do not appear to come from obviously pathological gate inflation.

## Phase 1A Result

- Real Gemma screening strongly rejects the frozen contiguous `24..29 -> 14..19` split as the best default.
- The top continuation shortlist is `{24..27 -> 14..19, 24..27 -> 16..18}`.
- `25..29 -> 15..19` improved on the legacy baseline but did not challenge the top pair.
