# Stage B Entry-Tune Output Probe Report

## setup

- Seeds: 42, 43, 44
- Train entry projector: True
- Stage B learning rates: base=0.0003, entry=0.00015, return=0.0003, gate=0.0003
- Bridge controls were reused from the frozen-entry output-aware Stage B reference run.
- Primary metrics: teacher-logit KL and held-out NLL.

## aggregate output summary

- skip_only_reference: kl_mean=1.527346, nll_mean=4.468700, ppl_mean=87.588721, top1_mean=0.464642, top5_mean=0.503024
- bridge_only_reference: kl_mean=0.646337, nll_mean=3.393890, ppl_mean=29.833916, top1_mean=0.654290, top5_mean=0.640657
- bridge_only_param_matched_reference: kl_mean=0.647097, nll_mean=3.395382, ppl_mean=29.869270, top1_mean=0.654521, top5_mean=0.639915
- hybrid_frozen_entry: kl_mean=0.655263, nll_mean=3.423486, ppl_mean=30.723442, top1_mean=0.649148, top5_mean=0.637986
- hybrid_train_entry: kl_mean=0.668607, nll_mean=3.451810, ppl_mean=31.607548, top1_mean=0.645438, top5_mean=0.638054
- hybrid_no_small_frozen_entry: kl_mean=0.673030, nll_mean=3.501779, ppl_mean=33.231261, top1_mean=0.648189, top5_mean=0.636736
- hybrid_no_small_train_entry: kl_mean=0.680983, nll_mean=3.486210, ppl_mean=32.738539, top1_mean=0.644719, top5_mean=0.634773
- full_large_reference: kl_mean=0.000000, nll_mean=2.949112, ppl_mean=19.191130, top1_mean=1.000000, top5_mean=1.000000

## paired deltas

- Delta sign convention: negative is better for KL/NLL/PPL, positive is better for top-1/top-5 agreement.
- hybrid_train_entry_minus_hybrid_frozen_entry: kl_delta_mean=0.013344, nll_delta_mean=0.028324, ppl_delta_mean=0.884106, top1_delta_mean=-0.003710, top5_delta_mean=0.000069, primary_wins=0/3
- hybrid_no_small_train_entry_minus_hybrid_no_small_frozen_entry: kl_delta_mean=0.007953, nll_delta_mean=-0.015569, ppl_delta_mean=-0.492722, top1_delta_mean=-0.003470, top5_delta_mean=-0.001963, primary_wins=1/3
- hybrid_train_entry_minus_hybrid_no_small_train_entry: kl_delta_mean=-0.012376, nll_delta_mean=-0.034400, ppl_delta_mean=-1.130992, top1_delta_mean=0.000719, top5_delta_mean=0.003282, primary_wins=2/3
- hybrid_train_entry_minus_bridge_only: kl_delta_mean=0.022270, nll_delta_mean=0.057920, ppl_delta_mean=1.773632, top1_delta_mean=-0.008852, top5_delta_mean=-0.002603, primary_wins=0/3
- hybrid_train_entry_minus_bridge_only_param_matched: kl_delta_mean=0.021509, nll_delta_mean=0.056428, ppl_delta_mean=1.738277, top1_delta_mean=-0.009083, top5_delta_mean=-0.001861, primary_wins=0/3
- hybrid_train_entry_minus_skip_only: kl_delta_mean=-0.858739, nll_delta_mean=-1.016890, ppl_delta_mean=-55.981173, top1_delta_mean=0.180796, top5_delta_mean=0.135031, primary_wins=3/3

## questions

1. Does training the entry projector improve hybrid output metrics? No.
2. Does training the entry projector improve hybrid_no_small output metrics? No.
3. When both are allowed to tune the entry projector, does hybrid still beat hybrid_no_small? Yes.
4. Does entry tuning materially reduce the gap to bridge_only or bridge_only_param_matched? No. bridge_only gap changed from kl=0.008926/nll=0.029596 to kl=0.022270/nll=0.057920; parameter-matched gap changed from kl=0.008166/nll=0.028104 to kl=0.021509/nll=0.056428.
5. Is the remaining gap now small enough to justify one tiny architecture sweep, or is the result already converging to a qualified negative against strong bridges? The result is still converging toward a qualified negative against strong bridges.

## recommendation

Do not proceed; write up the qualified result
