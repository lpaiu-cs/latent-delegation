# Stage B Ablation Report

## setup

- Config: configs/gemma2_conservative_pilot_256_stage_b_output_aware.yaml
- seq_len: 256
- max_train_steps: 200
- Seeds: 42, 43, 44
- Stage A checkpoint policy: reused one fixed Stage A checkpoint across all seeds so the Stage B comparison isolates the delegated-vs-bridge question instead of reintroducing Stage A variation.
- Stage A checkpoint path: artifacts/stage_a_pilot_ckpt/stage_a_checkpoint.pt
- Stage B loss weights: hidden_mse=1.0, hidden_cosine=1.0, kl=5.0, ce=1.0, delta_reg=0.0001
- Hybrid Stage B trainable params: 376833
- Original bridge-only Stage B trainable params: 458753
- Parameter-matched bridge rank: 53
- Parameter-matched bridge Stage B trainable params: 379905

## aggregate summary

- skip_only: hidden_mse_mean=25.509766, hidden_mse_std=0.402841, cosine_mean=0.746104, cosine_std=0.003649, gate_value_mean=0.000000, delta_norm_mean=0.000000
- bridge_only: hidden_mse_mean=22.161458, hidden_mse_std=0.274672, cosine_mean=0.782511, cosine_std=0.001656, gate_value_mean=0.069661, delta_norm_mean=1206.653747
- bridge_only_param_matched: hidden_mse_mean=22.436198, hidden_mse_std=0.212678, cosine_mean=0.781138, cosine_std=0.001740, gate_value_mean=0.069661, delta_norm_mean=1180.169830
- hybrid_no_small: hidden_mse_mean=24.585286, hidden_mse_std=0.397063, cosine_mean=0.763743, cosine_std=0.002791, gate_value_mean=0.070150, delta_norm_mean=843.765400
- hybrid: hidden_mse_mean=22.847656, hidden_mse_std=0.335938, cosine_mean=0.786835, cosine_std=0.002097, gate_value_mean=0.068197, delta_norm_mean=1317.256423
- hybrid_gate_zero: hidden_mse_mean=25.509766, hidden_mse_std=0.402841, cosine_mean=0.746104, cosine_std=0.003649

## interpretation rule

- A per-seed win means hybrid has lower hidden-state MSE and higher cosine on the same seed.
- A result counts as clear and reproducible here only if hybrid wins on both metrics in at least 2 of 3 seeds and the aggregate means point in the same direction.

## answers

1. Does hybrid consistently beat skip-only? Yes. Hybrid wins on both metrics in 3/3 seeds.
2. Does hybrid beat hybrid_no_small? Yes. Hybrid wins on both metrics in 3/3 seeds.
3. Does hybrid beat the original bridge-only? No. Hybrid wins on both metrics in 0/3 seeds.
4. Does hybrid beat the parameter-matched bridge-only? No. Hybrid wins on both metrics in 0/3 seeds.
5. Are any wins consistent across seeds? Yes. The only controls that meet the 2-of-3 reproducibility rule are: skip_only, hybrid_no_small.
6. Is the delegated path actually used, based on gate and delta diagnostics? Yes. Hybrid gate mean=0.068197, hybrid delta_norm_mean=1317.256423, and hybrid vs gate-zero gains are mse=2.662109, cosine=0.040731.
7. What is the most defensible current claim? The delegated small-model path is active and helps relative to skip-only, but the current evidence does not show a reproducible advantage over the stronger bridge controls.

## recommendation

Do not proceed to Stage C yet

## next minimal action

- Recommendation basis: the stronger bridge control on this run is `bridge_only`, and hybrid does not yet show a clear reproducible win over the stronger controls under the stated rule.
