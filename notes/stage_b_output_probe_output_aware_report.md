# Stage B Output Probe Report

## setup

- Config: configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml
- seq_len: 256
- Seeds: 42, 43, 44
- Held-out policy: reused the seed-matched Stage B validation text slice. No training was run.
- Primary output-level decision metrics: logit KL to the full-large teacher and held-out next-token NLL.
- Supporting metrics: perplexity, teacher top-1 agreement, and teacher top-5 overlap.
- Metric note: perplexity is an exponential transform of NLL, so it should not be treated as independent evidence.

## aggregate summary

- skip_only: kl_mean=1.527346, nll_mean=4.468700, ppl_mean=87.588721, top1_mean=0.464642, top5_mean=0.503024
- hybrid_no_small: kl_mean=0.673030, nll_mean=3.501779, ppl_mean=33.231261, top1_mean=0.648189, top5_mean=0.636736
- bridge_only: kl_mean=0.646337, nll_mean=3.393890, ppl_mean=29.833916, top1_mean=0.654290, top5_mean=0.640657
- bridge_only_param_matched: kl_mean=0.647097, nll_mean=3.395382, ppl_mean=29.869270, top1_mean=0.654521, top5_mean=0.639915
- hybrid: kl_mean=0.655263, nll_mean=3.423486, ppl_mean=30.723442, top1_mean=0.649148, top5_mean=0.637986
- full_large: kl_mean=0.000000, nll_mean=2.949112, ppl_mean=19.191130, top1_mean=1.000000, top5_mean=1.000000

## paired hybrid deltas

- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.
- hybrid_minus_skip_only: kl_delta_mean=-0.872083, nll_delta_mean=-1.045214, ppl_delta_mean=-56.865279, top1_delta_mean=0.184506, top5_delta_mean=0.134962, primary_wins=3/3
- hybrid_minus_hybrid_no_small: kl_delta_mean=-0.017767, nll_delta_mean=-0.078293, ppl_delta_mean=-2.507819, top1_delta_mean=0.000959, top5_delta_mean=0.001250, primary_wins=3/3
- hybrid_minus_bridge_only: kl_delta_mean=0.008926, nll_delta_mean=0.029596, ppl_delta_mean=0.889526, top1_delta_mean=-0.005142, top5_delta_mean=-0.002672, primary_wins=0/3
- hybrid_minus_bridge_only_param_matched: kl_delta_mean=0.008166, nll_delta_mean=0.028104, ppl_delta_mean=0.854171, top1_delta_mean=-0.005373, top5_delta_mean=-0.001930, primary_wins=0/3

## interpretation rule

- A seed-level output win means hybrid has lower KL and lower NLL than the comparator on the same held-out slice.
- A reproducible output-level win means hybrid wins on the primary metrics in at least 2 of 3 seeds and the aggregate KL/NLL deltas point in the same direction.

## answers

1. Does hybrid beat skip_only at the output level? Yes. Hybrid wins on the primary metrics in 3/3 seeds.
2. Does hybrid beat hybrid_no_small at the output level? Yes. Hybrid wins on the primary metrics in 3/3 seeds.
3. Does hybrid beat bridge_only at the output level? No. Hybrid wins on the primary metrics in 0/3 seeds.
4. Does hybrid beat bridge_only_param_matched at the output level? No. Hybrid wins on the primary metrics in 0/3 seeds.
5. Are any wins consistent across the 3 seeds? Yes. The reproducible output-level wins are: skip_only, hybrid_no_small.
6. Which metric is most trustworthy here: KL, CE/NLL, or PPL? CE/NLL. It is the direct held-out text likelihood objective, while PPL is just its exponential transform and KL is primarily a teacher-matching measure.
7. What is the best current research claim after adding output-level evidence? The delegated small-model path is active and clearly better than skip-only, but the current Stage B checkpoints do not translate the hidden-space advantage over no-small or bridge controls into better output-level language-model behavior.

## decision

Do not proceed to Stage C

## framing

- Recommendation basis: the project remains a qualified positive result relative to skip-only, but it is a negative result on the stronger claims `hybrid > hybrid_no_small` and `hybrid > strong bridge` at this milestone.
