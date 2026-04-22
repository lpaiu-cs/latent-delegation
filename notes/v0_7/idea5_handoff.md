# Idea 5 Handoff

- `v0.5.1` remains the frozen prior feasibility result.
- `v0_6` completed real Phase 1, static Idea 4, a fresh untouched holdout recheck, and token-wise Idea 4.
- Static mixture is the first pilot result that beat both bridge controls on KL and NLL, and that bridge win survived a fresh untouched holdout.
- Token-wise Idea 4 improved further and beat static mixture plus both bridge controls on both the main holdout and the fresh untouched holdout.
- The fixed shortlist is exactly `{24..27 -> 14..19, 24..27 -> 16..18}`.
- The large removed window is fixed to `24..27`.
- The next branch is Idea 5, not Stage C.

## Rationale

- The successful two-path token-wise result may be a local approximation to a broader monotone, asymmetric cross-scale alignment.
- Idea 5 starts as a discovery track rather than a new architecture push.
- The immediate question is whether the near-tied shortlist windows sit inside a broader low-cost monotone corridor.

## Guardrails

- Keep the search local around large `22..30` and small `13..20`.
- Keep same-family Gemma 2B/9B only.
- Keep frozen-backbone policy.
- Do not start Stage C.
- Do not implement Idea 2 in this task.
- Do not rewrite `v0_6`.
- Only test one minimal derived Idea 5 candidate if the monotone evidence is strong enough.

