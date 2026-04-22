# Phase 1 Real Combined Decision

## Idea 4 Starting Point

- The legacy contiguous split `24..29 -> 14..19` is rejected as the best default.
- The real continuation shortlist is exactly `{24..27 -> 14..19, 24..27 -> 16..18}`.
- Both shortlisted candidates are scientifically close enough that forcing a single hard winner would overstate the evidence.
- Idea 4 is justified precisely because Phase 1 produced a near-tied two-window shortlist rather than a clean single-window winner.

1. Does real Gemma evidence still reject the frozen contiguous `6 -> 6` split as the best default?
Yes. Real stage signatures moved away from the frozen contiguous `24..29 -> 14..19` prior, and real Phase 1A screening showed both `24..27 -> 14..19` and `24..27 -> 16..18` beating the legacy split by a large margin on KL, NLL, PPL, and top-1 agreement.

2. Which candidate is best on output-level metrics?
`24..27 -> 14..19` is the nominal winner under the stated ranking rule because it kept the best confirmed mean KL to teacher (`0.281641`) and the best confirmed mean top-5 overlap (`0.740074`). `24..27 -> 16..18` remained extremely close and kept the slightly better confirmed mean NLL, PPL, and top-1 agreement.

3. Does any candidate materially reduce the gap to the strong bridge controls?
Yes. Both confirmed candidates collapsed the KL gap to near zero relative to `bridge_only` and `bridge_only_param_matched`. Neither produced a clean overall output-level win over the strong bridge controls, because mean NLL still remained slightly worse.

4. Does any candidate preserve or strengthen `hybrid > hybrid_no_small` at the output level?
Yes. Both confirmed candidates beat `hybrid_no_small` on KL and NLL in all 3 seeds, while also improving top-1 and top-5 output agreement.

5. Is the gain merely incremental, or strong enough to justify Idea 4 immediately?
The gain is incremental in absolute recovery and still below the strong bridge controls, but it is strong enough to justify Idea 4 immediately because Phase 1 clearly supports the narrower asymmetric-window hypothesis and rejects the frozen contiguous `6 -> 6` structural prior.

6. What single candidate set should be carried into Idea 4?
Carry only the two-window shortlist `{24..27 -> 14..19, 24..27 -> 16..18}`. Drop the legacy split and the swapped `25..29 -> 15..19` candidate from the next step.

Proceed to Idea 4 with shortlisted windows
