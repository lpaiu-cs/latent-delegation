# Stage B Output Probe Report

## setup

- Config: configs/gemma2_conservative_pilot_256.yaml
- seq_len: 256
- Seeds: 42, 43, 44
- Held-out policy: reused the seed-matched Stage B validation text slice. No training was run.
- Primary output-level decision metrics: logit KL to the full-large teacher and held-out next-token NLL.
- Supporting metrics: perplexity, teacher top-1 agreement, and teacher top-5 overlap.
- Metric note: perplexity is an exponential transform of NLL, so it should not be treated as independent evidence.

## aggregate summary

- skip_only: kl_mean=1.527346, nll_mean=4.468700, ppl_mean=87.588721, top1_mean=0.464642, top5_mean=0.503024
- hybrid_no_small: kl_mean=1.001221, nll_mean=3.862371, ppl_mean=47.707230, top1_mean=0.566062, top5_mean=0.577198
- bridge_only: kl_mean=1.058858, nll_mean=3.910191, ppl_mean=50.061429, top1_mean=0.559166, top5_mean=0.567113
- bridge_only_param_matched: kl_mean=1.040588, nll_mean=3.893158, ppl_mean=49.181381, top1_mean=0.561232, top5_mean=0.569972
- hybrid: kl_mean=1.257108, nll_mean=4.106072, ppl_mean=60.964145, top1_mean=0.514797, top5_mean=0.540144
- full_large: kl_mean=0.000000, nll_mean=2.949112, ppl_mean=19.191130, top1_mean=1.000000, top5_mean=1.000000

## paired hybrid deltas

- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.
- hybrid_minus_skip_only: kl_delta_mean=-0.270238, nll_delta_mean=-0.362628, ppl_delta_mean=-26.624576, top1_delta_mean=0.050155, top5_delta_mean=0.037120, primary_wins=3/3
- hybrid_minus_hybrid_no_small: kl_delta_mean=0.255887, nll_delta_mean=0.243702, ppl_delta_mean=13.256915, top1_delta_mean=-0.051265, top5_delta_mean=-0.037054, primary_wins=0/3
- hybrid_minus_bridge_only: kl_delta_mean=0.198249, nll_delta_mean=0.195881, ppl_delta_mean=10.902716, top1_delta_mean=-0.044369, top5_delta_mean=-0.026968, primary_wins=0/3
- hybrid_minus_bridge_only_param_matched: kl_delta_mean=0.216520, nll_delta_mean=0.212914, ppl_delta_mean=11.782764, top1_delta_mean=-0.046435, top5_delta_mean=-0.029827, primary_wins=0/3

## interpretation rule

- A seed-level output win means hybrid has lower KL and lower NLL than the comparator on the same held-out slice.
- A reproducible output-level win means hybrid wins on the primary metrics in at least 2 of 3 seeds and the aggregate KL/NLL deltas point in the same direction.

## answers

1. Does hybrid beat skip_only at the output level? Yes. Hybrid wins on the primary metrics in 3/3 seeds.
2. Does hybrid beat hybrid_no_small at the output level? No. Hybrid wins on the primary metrics in 0/3 seeds.
3. Does hybrid beat bridge_only at the output level? No. Hybrid wins on the primary metrics in 0/3 seeds.
4. Does hybrid beat bridge_only_param_matched at the output level? No. Hybrid wins on the primary metrics in 0/3 seeds.
5. Are any wins consistent across the 3 seeds? Yes. The reproducible output-level wins are: skip_only.
6. Which metric is most trustworthy here: KL, CE/NLL, or PPL? CE/NLL. It is the direct held-out text likelihood objective, while PPL is just its exponential transform and KL is primarily a teacher-matching measure.
7. What is the best current research claim after adding output-level evidence? The delegated small-model path is active and clearly better than skip-only, but the current Stage B checkpoints do not translate the hidden-space advantage over no-small or bridge controls into better output-level language-model behavior.

## decision

Do not proceed to Stage C

## framing

- Recommendation basis: the project remains a qualified positive result relative to skip-only, but it is a negative result on the stronger claims `hybrid > hybrid_no_small` and `hybrid > strong bridge` at this milestone.
