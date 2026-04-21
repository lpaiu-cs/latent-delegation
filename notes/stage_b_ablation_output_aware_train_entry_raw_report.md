# Stage B Ablation Report

## setup

- Config: configs/gemma2_conservative_pilot_256_stage_b_output_aware_train_entry.yaml
- seq_len: 256
- max_train_steps: 200
- Seeds: 42, 43, 44
- Trained variants: hybrid, hybrid_no_small
- Stage A checkpoint policy: reused one fixed Stage A checkpoint across all seeds so the Stage B comparison isolates the delegated-vs-bridge question instead of reintroducing Stage A variation.
- Stage A checkpoint path: .\artifacts\stage_a_pilot_ckpt\stage_a_checkpoint.pt
- Stage B loss weights: hidden_mse=1.0, hidden_cosine=1.0, kl=5.0, ce=1.0, delta_reg=0.0001
- Hybrid Stage B trainable params: 8638977
- Original bridge-only Stage B trainable params: 458753
- Parameter-matched bridge rank: 1205
- Parameter-matched bridge Stage B trainable params: 8637441

## aggregate summary

- skip_only: hidden_mse_mean=25.509766, hidden_mse_std=0.402841, cosine_mean=0.746104, cosine_std=0.003649, gate_value_mean=0.000000, delta_norm_mean=0.000000
- hybrid: hidden_mse_mean=21.162109, hidden_mse_std=0.406264, cosine_mean=0.795959, cosine_std=0.002069, gate_value_mean=0.057454, delta_norm_mean=1720.684966
- hybrid_no_small: hidden_mse_mean=23.944010, hidden_mse_std=0.467953, cosine_mean=0.776896, cosine_std=0.003313, gate_value_mean=0.061768, delta_norm_mean=1280.483346
- hybrid_gate_zero: hidden_mse_mean=25.509766, hidden_mse_std=0.402841, cosine_mean=0.746104, cosine_std=0.003649

## interpretation rule

- A per-seed win means hybrid has lower hidden-state MSE and higher cosine on the same seed.
- A result counts as clear and reproducible here only if hybrid wins on both metrics in at least 2 of 3 seeds and the aggregate means point in the same direction.

## answers

1. Does hybrid consistently beat skip-only? Yes. Hybrid wins on both metrics in 3/3 seeds.
2. Does hybrid beat hybrid_no_small? Yes. Hybrid wins on both metrics in 3/3 seeds.
3. Does hybrid beat the original bridge-only? Not evaluated here. Hybrid wins on both metrics in 0/3 seeds.
4. Does hybrid beat the parameter-matched bridge-only? Not evaluated here. Hybrid wins on both metrics in 0/3 seeds.
5. Are any wins consistent across seeds? Yes. The only controls that meet the 2-of-3 reproducibility rule are: skip_only, hybrid_no_small.
6. Is the delegated path actually used, based on gate and delta diagnostics? Yes. Hybrid gate mean=0.057454, hybrid delta_norm_mean=1720.684966, and hybrid vs gate-zero gains are mse=4.347656, cosine=0.049856.
7. What is the most defensible current claim? The delegated small-model path is active and helps relative to skip-only, but the current evidence does not show a reproducible advantage over the stronger bridge controls.

## recommendation

Do not proceed to Stage C yet

## next minimal action

- Recommendation basis: this was a focused partial run intended to feed a separate comparison report.
