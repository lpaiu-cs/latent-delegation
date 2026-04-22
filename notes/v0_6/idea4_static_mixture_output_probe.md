# Idea 4 Static Mixture Output Probe

## Coarse Screen

- budget: 1 seed, 100 Stage B steps, same held-out Stage B validation slice
- static mixture:
  - KL `0.294867`
  - NLL `2.915419`
  - PPL `18.456539`
  - top-1 `0.768949`
  - top-5 `0.734291`
- key coarse comparisons:
  - vs best single-path candidate (`24..27 -> 16..18` on this seed): dKL `-0.005311`, dNLL `-0.057534`
  - vs single-path B: dKL `-0.006831`, dNLL `-0.060972`
  - vs no-small control: dKL `-0.005917`, dNLL `-0.066855`
  - vs `bridge_only`: dKL `-0.062659`, dNLL `-0.133023`
  - vs mixture-budget param-matched bridge: dKL `-0.059619`, dNLL `-0.127216`
- coarse decision: the static mixture was already better than both single-path references, the no-small control, and both bridge controls on the primary output metrics, so 3-seed confirmation was justified.

## Confirmation

- budget: 3 seeds, 200 Stage B steps, same held-out policy as Phase 1
- confirmed static mixture:
  - KL `0.267095 ± 0.016769`
  - NLL `3.000438 ± 0.096956`
  - PPL `20.156769 ± 1.924796`
  - top-1 `0.762009 ± 0.008599`
  - top-5 `0.741646 ± 0.004826`

Confirmed references:

- best single-path candidate by mean KL: `24..27 -> 14..19`
  - KL `0.281641 ± 0.017965`
  - NLL `3.078029 ± 0.095159`
- other shortlisted single-path candidate: `24..27 -> 16..18`
  - KL `0.282215 ± 0.016190`
  - NLL `3.074461 ± 0.093456`
- static mixture no-small:
  - KL `0.269326 ± 0.016630`
  - NLL `3.068990 ± 0.104198`
- `bridge_only`:
  - KL `0.288448 ± 0.013781`
  - NLL `3.072051 ± 0.110548`
- mixture-budget param-matched bridge:
  - KL `0.283258 ± 0.016296`
  - NLL `3.045527 ± 0.108679`

## Primary Deltas

- static mixture minus best single-path candidate:
  - dKL `-0.014546 ± 0.001862`
  - dNLL `-0.077591 ± 0.004188`
  - wins on KL and NLL: `3/3`
- static mixture minus other shortlisted single-path candidate:
  - dKL `-0.015120 ± 0.001143`
  - dNLL `-0.074023 ± 0.003522`
  - wins on KL and NLL: `3/3`
- static mixture minus no-small control:
  - dKL `-0.002231 ± 0.001662`
  - dNLL `-0.068552 ± 0.010723`
  - wins on KL and NLL: `3/3`
- static mixture minus `bridge_only`:
  - dKL `-0.021352 ± 0.003348`
  - dNLL `-0.071613 ± 0.013831`
  - wins on KL and NLL: `3/3`
- static mixture minus mixture-budget param-matched bridge:
  - dKL `-0.016163 ± 0.001825`
  - dNLL `-0.045089 ± 0.012672`
  - wins on KL and NLL: `3/3`

## Output-Level Reading

- The static mixture did not just stay competitive with the best single-path candidate; it beat both shortlisted single-path references in all 3 confirmation seeds.
- The static mixture also beat its no-small control in all 3 seeds, so the gain cannot be reduced to route mixing alone.
- Within this pilot, the static mixture did more than merely narrow the gap to the strong bridge controls; it beat both of them on the primary output metrics in all 3 seeds.
- Secondary agreement metrics were directionally positive overall, although top-5 changed only slightly against some comparators. That does not change the main decision because KL and NLL were the required ranking metrics.
