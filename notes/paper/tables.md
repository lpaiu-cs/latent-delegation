# Canonical Tables

These tables are generated directly from frozen artifacts. CSV and JSON copies live under `artifacts/paper_tables/`.

## Appendix Table A1. Adapter Training Protocol

- This table resolves the paper-method reproducibility detail that is not an output result table.
- It records Stage A/B corpus size, step budget, optimizer, LR, batch/accumulation, trainable modules, checkpoint selection, and source configs.
- Machine-readable files: `table_a1_adapter_training_protocol.csv`, `table_a1_adapter_training_protocol.json`.
- Source configs: `configs/v0_6/phase1_real/candidate_a_confirm_seed42.yaml`, `configs/v0_6/phase1_real/candidate_b_confirm_seed42.yaml`, `configs/v0_6/idea4/static_mixture_confirm_seed42.yaml`, `configs/v0_6/idea4/tokenwise_confirm_seed42.yaml`.

| run_family | training_pool | stage_a_steps | stage_b_steps | optimizer | base_lr | return_lr | gate_lr | weight_decay | max_grad_norm | micro_batch_size | grad_accum_steps | seq_len | checkpoint_selection | trainable_modules |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1_single_path_shortlist | Stage A: 144 examples; Stage B: 192 examples from 128 Wikitext-103-v1 train snippets plus 64 GSM8K train QA records | 200 | 200 | AdamW | 3.0e-4 |  |  | 0.0 | 1.0 | 1 | 8 | 256 | final fixed-budget checkpoint; no validation early stopping | Stage A: entry projector; Stage B: return adapter and scalar gate |
| static_two_path_mixture | Same Stage B pool; paths warm-started from confirmed Phase 1 checkpoints | 0 | 200 | AdamW | 3.0e-4 |  |  | 0.0 | 1.0 | 1 | 8 | 256 | final fixed-budget checkpoint; no validation early stopping | two return adapters plus global mixture logits; entry projectors loaded from Phase 1 and frozen |
| tokenwise_two_path_routing | Same Stage B pool; warm-started from static mixture; entry projectors frozen | 0 | 200 | AdamW | 1.5e-4 | 1.5e-4 | 3.0e-4 | 0.0 | 1.0 | 1 | 8 | 256 | final fixed-budget checkpoint; no validation early stopping | two return adapters plus token-wise gate network |

## Table 1. Bring-Up / Smoke Summary

- Exact metric names are preserved in the machine-readable CSV/JSON.
- The Stage A row is included here because it was part of the initial bring-up feasibility path.
- Machine-readable files: `table_01_bring_up_smoke_summary.csv`, `table_01_bring_up_smoke_summary.json`.
- Source artifacts: `artifacts/env_sanity.json`, `artifacts/real_gemma_smoke.json`, `artifacts/stage_a_pilot_metrics.json`.

| section | compared_model | seed_policy | holdout_policy | overall_pass | overall_success | completed_cases | expected_cases | success | largest_successful_seq_len | peak_vram_mb | wall_time_sec | cuda_available | hf_auth_token_present | gemma_access_success | bitsandbytes_available | device_name | total_vram_gb | python_version | torch_version | seq_len | train_loss_start | train_loss_end | heldout_mse_before | heldout_mse_after | heldout_cosine_before | heldout_cosine_after | trainable_parameters | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| environment | all | not_applicable | not_applicable | True |  |  |  |  |  |  |  | True | True | True | True | NVIDIA GeForce RTX 5090 | 31.842 | 3.12.9 | 2.10.0.dev20251104+cu128 |  |  |  |  |  |  |  |  |  |
| smoke_matrix | all | not_applicable | not_applicable |  | True | 14 | 14 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| stage_a_pilot | entry_projector | 42 | alignment_validation_subset |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 256 | 354 | 91 | 111.195312 | 99.949219 | 0.007812 | 0.844734 | 8262144 |  |
| smoke_case | small_only_load | not_applicable | synthetic_forward_smoke |  |  |  |  | True |  | 3376 | 7.891505 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |
| smoke_case | large_only_load | not_applicable | synthetic_forward_smoke |  |  |  |  | True |  | 8814 | 17.865116 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |
| smoke_case | full_large | not_applicable | synthetic_forward_smoke |  |  |  |  | True | 256 | 6493.492676 | 18.785292 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |
| smoke_case | skip_only | not_applicable | synthetic_forward_smoke |  |  |  |  | True | 256 | 6495.242676 | 20.131872 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |
| smoke_case | bridge_only | not_applicable | synthetic_forward_smoke |  |  |  |  | True | 256 | 6500.493164 | 20.383161 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |
| smoke_case | hybrid | not_applicable | synthetic_forward_smoke |  |  |  |  | True | 256 | 9616.736816 | 26.018009 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | cuda |

## Table 2. v0.5.x Key Ablation Summary

- This table keeps the v0.5.x pilot progression explicit rather than rewriting history around the later v0.6.0 branch.
- The output-aware Stage B rows are the last v0.5.x architecture before the Phase 1 continuation work.
- Machine-readable files: `table_02_v05_key_ablation_summary.csv`, `table_02_v05_key_ablation_summary.json`.
- Source artifacts: `artifacts/stage_b_ablation_results.json`, `artifacts/stage_b_output_probe_results.json`, `artifacts/stage_b_ablation_output_aware_results.json`, `artifacts/stage_b_output_probe_output_aware_results.json`, `artifacts/stage_b_entry_tune_results.json`, `artifacts/stage_b_entry_tune_output_probe_results.json`.

| subphase | compared_model | seed_policy | holdout_policy | hidden_mse_mean | hidden_cosine_mean | delta_norm_mean | gate_value_mean | logit_kl_to_teacher_mean | nll_mean | perplexity_mean | top1_agreement_mean | top5_overlap_mean | source_hidden | source_output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage_b_hidden_only | bridge_only | 42,43,44 | stage_b validation split reused for the matching seed | 20.535807 | 0.800323 | 1300.676524 | 0.071615 | 1.058858 | 3.910191 | 50.061429 | 0.559166 | 0.567113 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_hidden_only | bridge_only_param_matched | 42,43,44 | stage_b validation split reused for the matching seed | 20.880208 | 0.797765 | 1245.829941 | 0.07194 | 1.040588 | 3.893158 | 49.181381 | 0.561232 | 0.569972 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_hidden_only | full_large | 42,43,44 | stage_b validation split reused for the matching seed |  |  |  |  | 0 | 2.949112 | 19.19113 | 1 | 1 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_hidden_only | hybrid | 42,43,44 | stage_b validation split reused for the matching seed | 21.068359 | 0.809565 | 1485.074935 | 0.071289 | 1.257108 | 4.106072 | 60.964145 | 0.514797 | 0.540144 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_hidden_only | hybrid_no_small | 42,43,44 | stage_b validation split reused for the matching seed | 23.601562 | 0.781677 | 798.720241 | 0.07487 | 1.001221 | 3.862371 | 47.70723 | 0.566062 | 0.577198 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_hidden_only | skip_only | 42,43,44 | stage_b validation split reused for the matching seed | 25.509766 | 0.746104 | 0 | 0 | 1.527346 | 4.4687 | 87.588721 | 0.464642 | 0.503024 | artifacts/stage_b_ablation_results.json | artifacts/stage_b_output_probe_results.json |
| stage_b_output_aware | bridge_only | 42,43,44 | stage_b validation split reused for the matching seed | 22.161458 | 0.782511 | 1206.653747 | 0.069661 | 0.646337 | 3.39389 | 29.833916 | 0.65429 | 0.640657 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_output_aware | bridge_only_param_matched | 42,43,44 | stage_b validation split reused for the matching seed | 22.436198 | 0.781138 | 1180.16983 | 0.069661 | 0.647097 | 3.395382 | 29.86927 | 0.654521 | 0.639915 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_output_aware | full_large | 42,43,44 | stage_b validation split reused for the matching seed |  |  |  |  | 0 | 2.949112 | 19.19113 | 1 | 1 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_output_aware | hybrid | 42,43,44 | stage_b validation split reused for the matching seed | 22.847656 | 0.786835 | 1317.256423 | 0.068197 | 0.655263 | 3.423486 | 30.723442 | 0.649148 | 0.637986 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_output_aware | hybrid_no_small | 42,43,44 | stage_b validation split reused for the matching seed | 24.585286 | 0.763743 | 843.7654 | 0.07015 | 0.67303 | 3.501779 | 33.231261 | 0.648189 | 0.636736 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_output_aware | skip_only | 42,43,44 | stage_b validation split reused for the matching seed | 25.509766 | 0.746104 | 0 | 0 | 1.527346 | 4.4687 | 87.588721 | 0.464642 | 0.503024 | artifacts/stage_b_ablation_output_aware_results.json | artifacts/stage_b_output_probe_output_aware_results.json |
| stage_b_entry_tune_follow_up | bridge_only_param_matched_reference | 42,43,44 | teacher_logit_output_probe | 22.436198 | 0.781138 | 1180.16983 | 0.069661 | 0.647097 | 3.395382 | 29.86927 | 0.654521 | 0.639915 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | bridge_only_reference | 42,43,44 | teacher_logit_output_probe | 22.161458 | 0.782511 | 1206.653747 | 0.069661 | 0.646337 | 3.39389 | 29.833916 | 0.65429 | 0.640657 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | full_large_reference | 42,43,44 | teacher_logit_output_probe |  |  |  |  | 0 | 2.949112 | 19.19113 | 1 | 1 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | hybrid_frozen_entry | 42,43,44 | teacher_logit_output_probe | 22.847656 | 0.786835 | 1317.256423 | 0.068197 | 0.655263 | 3.423486 | 30.723442 | 0.649148 | 0.637986 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | hybrid_no_small_frozen_entry | 42,43,44 | teacher_logit_output_probe | 24.585286 | 0.763743 | 843.7654 | 0.07015 | 0.67303 | 3.501779 | 33.231261 | 0.648189 | 0.636736 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | hybrid_no_small_train_entry | 42,43,44 | teacher_logit_output_probe | 23.94401 | 0.776896 | 1280.483346 | 0.061768 | 0.680983 | 3.48621 | 32.738539 | 0.644719 | 0.634773 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | hybrid_train_entry | 42,43,44 | teacher_logit_output_probe | 21.162109 | 0.795959 | 1720.684966 | 0.057454 | 0.668607 | 3.45181 | 31.607548 | 0.645438 | 0.638054 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |
| stage_b_entry_tune_follow_up | skip_only_reference | 42,43,44 | teacher_logit_output_probe | 25.509766 | 0.746104 | 0 | 0 | 1.527346 | 4.4687 | 87.588721 | 0.464642 | 0.503024 | artifacts/stage_b_entry_tune_results.json | artifacts/stage_b_entry_tune_output_probe_results.json |

## Table 3. Phase 1 Shortlist Summary

- The real Gemma Phase 1 decision rejected the legacy `24..29 -> 14..19` split as the best default.
- The confirmed shortlist carried forward into Idea 4 was exactly `24..27 -> 14..19` and `24..27 -> 16..18`.
- Machine-readable files: `table_03_phase1_shortlist_summary.csv`, `table_03_phase1_shortlist_summary.json`.
- Source artifacts: `artifacts/v0_6/phase1_real/combined/ranking_summary.json`.

| screening_stage | candidate_id | mapping | seed_count | holdout_policy | logit_kl_to_teacher | nll | perplexity | top1_agreement | top5_overlap | delta_kl_vs_hybrid_no_small | delta_nll_vs_hybrid_no_small | delta_kl_vs_bridge_only | delta_nll_vs_bridge_only | wins_vs_hybrid_no_small_all_seeds | wins_vs_skip_only_all_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse | candidate_b | 24..27 -> 14..19 | 1 | phase1_real_coarse_output_probe | 0.327293 | 3.024373 | 20.581106 | 0.761614 | 0.728056 | -0.083045 | -0.165037 | -0.036994 | -0.036675 |  |  |
| coarse | candidate_a | 24..27 -> 16..18 | 1 | phase1_real_coarse_output_probe | 0.328102 | 3.032549 | 20.750055 | 0.761002 | 0.72879 | -0.081156 | -0.152124 | -0.031139 | -0.016886 |  |  |
| coarse | candidate_c | 25..29 -> 15..19 | 1 | phase1_real_coarse_output_probe | 0.602056 | 3.300886 | 27.13668 | 0.68643 | 0.663203 | -0.149995 | -0.234337 | -0.05467 | -0.051956 |  |  |
| coarse | legacy | 24..29 -> 14..19 | 1 | phase1_real_coarse_output_probe | 0.72503 | 3.425046 | 30.724053 | 0.652812 | 0.630746 | -0.241896 | -0.324649 | -0.080399 | -0.090312 |  |  |
| confirmation | candidate_b_confirm | 24..27 -> 14..19 | 3 | phase1_real_confirmation_output_probe | 0.281641 | 3.078029 | 21.780681 | 0.755546 | 0.740074 | -0.017463 | -0.057073 | -0.000492 | 0.017439 | True | True |
| confirmation | candidate_a_confirm | 24..27 -> 16..18 | 3 | phase1_real_confirmation_output_probe | 0.282215 | 3.074461 | 21.700679 | 0.756979 | 0.739941 | -0.018538 | -0.069355 | 0.000773 | 0.013718 | True | True |

## Table 4. Static Mixture Summary on Development and Untouched Confirmation Holdouts

- The fresh-holdout recheck is included because it was the required rigor step before token-wise gating.
- This table keeps the static-mixture main and fresh holdouts together so the bridge win can be read in one place.
- Machine-readable files: `table_04_static_mixture_summary.csv`, `table_04_static_mixture_summary.json`.
- Source artifacts: `artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json`, `artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json`.

| phase_name | holdout_policy | seed_count | compared_model | hidden_mse_mean | hidden_cosine_mean | logit_kl_to_teacher_mean | nll_mean | perplexity_mean | top1_agreement_mean | top5_overlap_mean | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | bridge_only |  |  | 0.288448 | 3.072051 | 21.673345 | 0.755489 | 0.737227 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | bridge_only_param_matched |  |  | 0.283258 | 3.045527 | 21.102889 | 0.755978 | 0.738655 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | full_large |  |  | 0 | 2.949112 | 19.19113 | 1 | 1 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | single_path_a |  |  | 0.282215 | 3.074461 | 21.700679 | 0.756979 | 0.739941 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | single_path_b |  |  | 0.281641 | 3.078029 | 21.780681 | 0.755546 | 0.740074 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | skip_only |  |  | 0.455535 | 3.380945 | 29.51741 | 0.698153 | 0.696677 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | static_mixture |  |  | 0.267095 | 3.000438 | 20.156769 | 0.762009 | 0.741646 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_confirmation_main | development holdout (reused during model selection) | 3 | static_mixture_no_small |  |  | 0.269326 | 3.06899 | 21.597682 | 0.760718 | 0.743137 | artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | bridge_only |  |  | 0.289564 | 3.295081 | 26.979976 | 0.745563 | 0.73838 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | bridge_only_param_matched |  |  | 0.284433 | 3.262601 | 26.117707 | 0.749861 | 0.739795 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | full_large |  |  | 0 | 3.235025 | 25.407006 | 1 | 1 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | single_path_a |  |  | 0.285229 | 3.306798 | 27.299031 | 0.74584 | 0.74193 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | single_path_b |  |  | 0.283206 | 3.307474 | 27.317435 | 0.750277 | 0.742568 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | skip_only |  |  | 0.45204 | 3.636127 | 37.944603 | 0.700915 | 0.699667 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | static_mixture |  |  | 0.267244 | 3.213048 | 24.854807 | 0.753466 | 0.740155 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |
| static_mixture_fresh_holdout_recheck | untouched confirmation holdout (fresh Wikitext test slice) | 3 | static_mixture_no_small |  |  | 0.267322 | 3.296346 | 27.014454 | 0.749029 | 0.745535 | artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json |

## Table 5. Token-Wise Summary on Development and Untouched Confirmation Holdouts

- This is the canonical `v0.6.0` table.
- The current best model/result claim remains tied to these token-wise rows, not to the later analysis branches.
- Machine-readable files: `table_05_tokenwise_summary.csv`, `table_05_tokenwise_summary.json`.
- Source artifacts: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json`, `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json`.

| phase_name | holdout_policy | seed_count | compared_model | hidden_mse_mean | hidden_cosine_mean | logit_kl_to_teacher_mean | nll_mean | perplexity_mean | top1_agreement_mean | top5_overlap_mean | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | bridge_only |  |  | 0.288448 | 3.072051 | 21.673345 | 0.755489 | 0.737227 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | bridge_only_param_matched |  |  | 0.302323 | 3.102081 | 22.330668 | 0.75062 | 0.731407 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | full_large |  |  | 0 | 2.949112 | 19.19113 | 1 | 1 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | skip_only |  |  | 0.455535 | 3.380945 | 29.51741 | 0.698153 | 0.696677 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | static_mixture |  |  | 0.267095 | 3.000438 | 20.156769 | 0.762009 | 0.741646 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | static_mixture_no_small |  |  | 0.269326 | 3.06899 | 21.597682 | 0.760718 | 0.743137 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | tokenwise_mixture |  |  | 0.255739 | 2.980182 | 19.76376 | 0.763351 | 0.744268 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_main | development holdout (reused during model selection) | 3 | tokenwise_mixture_no_small |  |  | 0.257501 | 3.038605 | 20.951639 | 0.762129 | 0.746334 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | bridge_only |  |  | 0.289564 | 3.295081 | 26.979976 | 0.745563 | 0.73838 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | bridge_only_param_matched |  |  | 0.301746 | 3.327024 | 27.855381 | 0.737105 | 0.734359 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | full_large |  |  | 0 | 3.235025 | 25.407006 | 1 | 1 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | skip_only |  |  | 0.45204 | 3.636127 | 37.944603 | 0.700915 | 0.699667 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | static_mixture |  |  | 0.267244 | 3.213048 | 24.854807 | 0.753466 | 0.740155 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | static_mixture_no_small |  |  | 0.267322 | 3.296346 | 27.014454 | 0.749029 | 0.745535 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | tokenwise_mixture |  |  | 0.248886 | 3.185004 | 24.167632 | 0.758319 | 0.742734 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |
| tokenwise_confirmation_fresh_holdout | untouched confirmation holdout (fresh Wikitext test slice) | 3 | tokenwise_mixture_no_small |  |  | 0.251294 | 3.261786 | 26.097261 | 0.759151 | 0.747393 | artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json |

## Table 6. Idea 5 Bounded Discovery Summary

- The discovery branch succeeded analytically by recovering a local corridor, but its bounded empirical candidate did not beat `v0.6.0`.
- This table keeps both the proxy ranking and the single bounded pilot in the same canonical view.
- Machine-readable files: `table_06_idea5_bounded_discovery_summary.csv`, `table_06_idea5_bounded_discovery_summary.json`.
- Source artifacts: `artifacts/v0_7/idea5_discovery/solver/top_paths.json`, `artifacts/v0_7/idea5_discovery/empirical_check/pilot_results.json`.

| row_group | name | mapping | seed_policy | holdout_policy | combined_proxy_cost | stage_signature_distance | hidden_alignment_proxy | logit_disruption_proxy | output_anchor_proxy | logit_kl_to_teacher | nll | perplexity | top1_agreement | top5_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proxy_window | shortlist_path_a | 24..27 -> 16..18 | not_applicable | local_monotone_proxy_region | 0.237026 | 2.730391 | 1.908594 | 1.952512 | 0 |  |  |  |  |  |
| proxy_window | shortlist_path_b | 24..27 -> 14..19 | not_applicable | local_monotone_proxy_region | 0.261984 | 2.858483 | 1.989775 | 2.052247 | 0 |  |  |  |  |  |
| proxy_window | derived_midpoint | 24..27 -> 15..18 | not_applicable | local_monotone_proxy_region | 0.270362 | 2.917509 | 2.237339 | 1.872477 | 0.125 |  |  |  |  |  |
| proxy_window | phase1_candidate_c | 25..29 -> 15..19 | not_applicable | local_monotone_proxy_region | 0.307316 | 3.099865 | 2.290119 | 2.089143 | 0.333333 |  |  |  |  |  |
| proxy_window | legacy_fixed_split | 24..29 -> 14..19 | not_applicable | local_monotone_proxy_region | 0.315038 | 3.13967 | 2.319163 | 2.116368 | 0.166667 |  |  |  |  |  |
| top_path_segment | path_1_segment_1 | 22..22 -> 13..14 | not_applicable | local_monotone_proxy_region | 0.286616 | 3.006732 | 2.345761 | 1.880915 | 0.928571 |  |  |  |  |  |
| top_path_segment | path_1_segment_2 | 23..24 -> 15..16 | not_applicable | local_monotone_proxy_region | 0.27383 | 3.034798 | 2.734476 | 1.316298 | 0.733333 |  |  |  |  |  |
| top_path_segment | path_1_segment_3 | 25..27 -> 17..18 | not_applicable | local_monotone_proxy_region | 0.288444 | 2.994608 | 1.800267 | 2.393056 | 0.291667 |  |  |  |  |  |
| top_path_segment | path_1_segment_4 | 28..30 -> 19..20 | not_applicable | local_monotone_proxy_region | 0.279379 | 2.951958 | 1.687134 | 2.42232 | 0.928571 |  |  |  |  |  |
| bounded_empirical_check | hybrid | L24-27__S15-18 | 42 | idea5_bounded_pilot |  |  |  |  |  | 0.424089 | 2.944444 | 19.000104 | 0.713889 | 0.712222 |
| bounded_empirical_check | hybrid_no_small | L24-27__S15-18 | 42 | idea5_bounded_pilot |  |  |  |  |  | 0.544705 | 3.157639 | 23.515009 | 0.693056 | 0.6775 |
| bounded_empirical_check | skip_only | L24-27__S15-18 | 42 | idea5_bounded_pilot |  |  |  |  |  | 0.561719 | 3.177083 | 23.976719 | 0.691667 | 0.674167 |

## Table 7. Idea 2 Attribution Summary

- All deltas are relative to the full `v0.6.0` token-wise model on the named holdout policy.
- The path-specific rows are included because the analysis branch found path A to be more sensitive than path B.
- Machine-readable files: `table_07_idea2_attribution_summary.csv`, `table_07_idea2_attribution_summary.json`.
- Source artifacts: `artifacts/v0_8/idea2_attribution/results.json`.

| row_group | holdout_policy | variant_name | seed_policy | logit_kl_to_teacher_delta | nll_delta | perplexity_delta | top1_agreement_delta | top5_overlap_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_suppression | untouched_confirmation_holdout | bridge_only | 42,43,44 | 0.040677 | 0.110077 | 2.812344 | -0.012757 | -0.004354 |
| overall_suppression | untouched_confirmation_holdout | bridge_only_param_matched | 42,43,44 | 0.052859 | 0.14202 | 3.687748 | -0.021215 | -0.008375 |
| overall_suppression | untouched_confirmation_holdout | tokenwise_attn_suppressed | 42,43,44 | 0.109496 | 0.224608 | 6.086972 | -0.033694 | -0.024293 |
| overall_suppression | untouched_confirmation_holdout | tokenwise_both_suppressed | 42,43,44 | 0.154198 | 0.379402 | 11.150949 | -0.044093 | -0.031004 |
| overall_suppression | untouched_confirmation_holdout | tokenwise_mlp_suppressed | 42,43,44 | 0.182872 | 0.37961 | 11.164554 | -0.053522 | -0.036661 |
| overall_suppression | untouched_confirmation_holdout | tokenwise_no_small | 42,43,44 | 0.002408 | 0.076782 | 1.929629 | 0.000832 | 0.004659 |
| path_specific_suppression | untouched_confirmation_holdout | tokenwise_attn_suppressed_path_a | 42,43,44 | 0.05877 | 0.107356 | 2.741662 | -0.023988 | -0.012895 |
| path_specific_suppression | untouched_confirmation_holdout | tokenwise_attn_suppressed_path_b | 42,43,44 | 0.05224 | 0.114202 | 2.925029 | -0.015391 | -0.008098 |
| path_specific_suppression | untouched_confirmation_holdout | tokenwise_mlp_suppressed_path_a | 42,43,44 | 0.089358 | 0.168088 | 4.438798 | -0.033694 | -0.02127 |
| path_specific_suppression | untouched_confirmation_holdout | tokenwise_mlp_suppressed_path_b | 42,43,44 | 0.068471 | 0.165835 | 4.365769 | -0.019689 | -0.011869 |
| overall_suppression | development_holdout | bridge_only | 42,43,44 | 0.032709 | 0.091869 | 1.909584 | -0.007862 | -0.007041 |
| overall_suppression | development_holdout | bridge_only_param_matched | 42,43,44 | 0.046584 | 0.121899 | 2.566907 | -0.012731 | -0.012861 |
| overall_suppression | development_holdout | tokenwise_attn_suppressed | 42,43,44 | 0.103797 | 0.21867 | 4.840058 | -0.029798 | -0.026582 |
| overall_suppression | development_holdout | tokenwise_both_suppressed | 42,43,44 | 0.147248 | 0.332872 | 7.813044 | -0.046221 | -0.035207 |
| overall_suppression | development_holdout | tokenwise_mlp_suppressed | 42,43,44 | 0.182743 | 0.350897 | 8.28429 | -0.050819 | -0.042359 |
| overall_suppression | development_holdout | tokenwise_no_small | 42,43,44 | 0.001761 | 0.058422 | 1.187879 | -0.001222 | 0.002067 |
| path_specific_suppression | development_holdout | tokenwise_attn_suppressed_path_a | 42,43,44 | 0.060497 | 0.118126 | 2.49188 | -0.016073 | -0.015143 |
| path_specific_suppression | development_holdout | tokenwise_attn_suppressed_path_b | 42,43,44 | 0.036265 | 0.091609 | 1.906898 | -0.010653 | -0.008452 |
| path_specific_suppression | development_holdout | tokenwise_mlp_suppressed_path_a | 42,43,44 | 0.100613 | 0.174154 | 3.760242 | -0.026991 | -0.023539 |
| path_specific_suppression | development_holdout | tokenwise_mlp_suppressed_path_b | 42,43,44 | 0.049808 | 0.130929 | 2.748704 | -0.013152 | -0.010777 |

## Table 8. v0_9 Generalization Summary

- The bounded generalization branch is evaluation-only; it does not alter the frozen `v0.6.0` best-model claim.
- Bootstrap rows are included only for token-wise versus static and bridge baselines, matching the saved analysis.
- Machine-readable files: `table_08_v09_generalization_summary.csv`, `table_08_v09_generalization_summary.json`.
- Source artifacts: `artifacts/v0_9/generalization/aggregated/summary.json`.

| task_name | task_type | holdout_policy | sampling_seed | seed_policy | compared_model | accuracy_mean | mean_choice_margin_mean | accuracy_delta_mean | accuracy_delta_ci_low | accuracy_delta_ci_high | mean_choice_margin_delta_mean | mean_choice_margin_delta_ci_low | mean_choice_margin_delta_ci_high | logit_kl_to_teacher_mean | nll_mean | perplexity_mean | logit_kl_delta_mean | logit_kl_delta_ci_low | logit_kl_delta_ci_high | nll_delta_mean | nll_delta_ci_low | nll_delta_ci_high | truncation_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | bridge_only | 0.432292 | 0.566671 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | bridge_only_param_matched | 0.4375 | 0.563944 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | skip_only | 0.4375 | 0.629761 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | static_mixture | 0.427083 | 0.58429 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | tokenwise_mixture | 0.442708 | 0.591288 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice | deterministic_validation_slice | 9005 | 42,43,44 | tokenwise_mixture_no_small | 0.411458 | 0.576253 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_challenge | multichoice_bootstrap | deterministic_validation_slice | 9005 | 42,43,44 | tokenwise_minus_bridge_only |  |  | 0.010417 | -0.046875 | 0.072917 | 0.024618 | -0.01825 | 0.070821 |  |  |  |  |  |  |  |  |  |  |
| arc_challenge | multichoice_bootstrap | deterministic_validation_slice | 9005 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  | 0.005208 | -0.052083 | 0.057292 | 0.027344 | -0.022868 | 0.078756 |  |  |  |  |  |  |  |  |  |  |
| arc_challenge | multichoice_bootstrap | deterministic_validation_slice | 9005 | 42,43,44 | tokenwise_minus_static_mixture |  |  | 0.015625 | -0.010417 | 0.046875 | 0.006999 | -0.012655 | 0.028605 |  |  |  |  |  |  |  |  |  |  |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | bridge_only | 0.828125 | 1.536992 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | bridge_only_param_matched | 0.828125 | 1.514064 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | skip_only | 0.828125 | 1.446747 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | static_mixture | 0.796875 | 1.478984 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | tokenwise_mixture | 0.791667 | 1.484111 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice | deterministic_validation_slice | 9004 | 42,43,44 | tokenwise_mixture_no_small | 0.796875 | 1.511017 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| arc_easy | multichoice_bootstrap | deterministic_validation_slice | 9004 | 42,43,44 | tokenwise_minus_bridge_only |  |  | -0.036458 | -0.083333 | 0 | -0.052882 | -0.120199 | 0.010661 |  |  |  |  |  |  |  |  |  |  |
| arc_easy | multichoice_bootstrap | deterministic_validation_slice | 9004 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  | -0.036458 | -0.083333 | 0 | -0.029953 | -0.114304 | 0.048335 |  |  |  |  |  |  |  |  |  |  |
| arc_easy | multichoice_bootstrap | deterministic_validation_slice | 9004 | 42,43,44 | tokenwise_minus_static_mixture |  |  | -0.005208 | -0.015625 | 0 | 0.005127 | -0.015381 | 0.02478 |  |  |  |  |  |  |  |  |  |  |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | bridge_only | 0.65625 | 0.516439 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | bridge_only_param_matched | 0.65625 | 0.513102 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | skip_only | 0.640625 | 0.511597 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | static_mixture | 0.671875 | 0.519775 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | tokenwise_mixture | 0.671875 | 0.516154 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice | deterministic_validation_slice | 9001 | 42,43,44 | tokenwise_mixture_no_small | 0.671875 | 0.516398 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| hellaswag | multichoice_bootstrap | deterministic_validation_slice | 9001 | 42,43,44 | tokenwise_minus_bridge_only |  |  | 0.015625 | -0.010417 | 0.046875 | -0.000285 | -0.013794 | 0.013835 |  |  |  |  |  |  |  |  |  |  |
| hellaswag | multichoice_bootstrap | deterministic_validation_slice | 9001 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  | 0.015625 | -0.010417 | 0.052083 | 0.003052 | -0.012288 | 0.01945 |  |  |  |  |  |  |  |  |  |  |
| hellaswag | multichoice_bootstrap | deterministic_validation_slice | 9001 | 42,43,44 | tokenwise_minus_static_mixture |  |  | -0 | -0.015625 | 0.015625 | -0.003621 | -0.00944 | 0.00236 |  |  |  |  |  |  |  |  |  |  |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | bridge_only | 0.734375 | 0.775146 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | bridge_only_param_matched | 0.744792 | 0.776326 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | skip_only | 0.734375 | 0.799438 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | static_mixture | 0.723958 | 0.759684 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | tokenwise_mixture | 0.723958 | 0.760132 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice | deterministic_validation_slice | 9002 | 42,43,44 | tokenwise_mixture_no_small | 0.729167 | 0.773153 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| piqa | multichoice_bootstrap | deterministic_validation_slice | 9002 | 42,43,44 | tokenwise_minus_bridge_only |  |  | -0.010417 | -0.046875 | 0.015625 | -0.015015 | -0.036336 | 0.007975 |  |  |  |  |  |  |  |  |  |  |
| piqa | multichoice_bootstrap | deterministic_validation_slice | 9002 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  | -0.020833 | -0.072917 | 0.010417 | -0.016195 | -0.040487 | 0.011271 |  |  |  |  |  |  |  |  |  |  |
| piqa | multichoice_bootstrap | deterministic_validation_slice | 9002 | 42,43,44 | tokenwise_minus_static_mixture |  |  | 0 | 0 | 0 | 0.000448 | -0.00826 | 0.008952 |  |  |  |  |  |  |  |  |  |  |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | bridge_only | 0.635417 | 0.743652 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | bridge_only_param_matched | 0.65625 | 0.743327 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | skip_only | 0.6875 | 0.80127 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | static_mixture | 0.65625 | 0.68929 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | tokenwise_mixture | 0.645833 | 0.707194 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice | deterministic_validation_slice | 9003 | 42,43,44 | tokenwise_mixture_no_small | 0.635417 | 0.719808 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0 |
| winogrande | multichoice_bootstrap | deterministic_validation_slice | 9003 | 42,43,44 | tokenwise_minus_bridge_only |  |  | 0.010417 | -0.067708 | 0.083333 | -0.036458 | -0.102783 | 0.021973 |  |  |  |  |  |  |  |  |  |  |
| winogrande | multichoice_bootstrap | deterministic_validation_slice | 9003 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  | -0.010417 | -0.088542 | 0.052083 | -0.036133 | -0.097412 | 0.022135 |  |  |  |  |  |  |  |  |  |  |
| winogrande | multichoice_bootstrap | deterministic_validation_slice | 9003 | 42,43,44 | tokenwise_minus_static_mixture |  |  | -0.010417 | -0.052083 | 0.020833 | 0.017904 | -0.024577 | 0.058757 |  |  |  |  |  |  |  |  |  |  |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | bridge_only |  |  |  |  |  |  |  |  | 0.254975 | 3.433371 | 30.98094 |  |  |  |  |  |  | 0 |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | bridge_only_param_matched |  |  |  |  |  |  |  |  | 0.266066 | 3.446407 | 31.387445 |  |  |  |  |  |  | 0 |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | skip_only |  |  |  |  |  |  |  |  | 0.34523 | 3.591141 | 36.275445 |  |  |  |  |  |  | 0 |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | static_mixture |  |  |  |  |  |  |  |  | 0.258699 | 3.419273 | 30.547385 |  |  |  |  |  |  | 0 |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | tokenwise_mixture |  |  |  |  |  |  |  |  | 0.251354 | 3.423984 | 30.691544 |  |  |  |  |  |  | 0 |
| lambada_openai | lm | deterministic_test_slice | 9010 | 42,43,44 | tokenwise_mixture_no_small |  |  |  |  |  |  |  |  | 0.245181 | 3.443454 | 31.294895 |  |  |  |  |  |  | 0 |
| lambada_openai | lm_bootstrap | deterministic_test_slice | 9010 | 42,43,44 | tokenwise_minus_bridge_only |  |  |  |  |  |  |  |  |  |  |  | -0.003499 | -0.006922 | -0.000219 | -0.009033 | -0.018473 | 0.00057 |  |
| lambada_openai | lm_bootstrap | deterministic_test_slice | 9010 | 42,43,44 | tokenwise_minus_bridge_only_param_matched |  |  |  |  |  |  |  |  |  |  |  | -0.014684 | -0.018865 | -0.010747 | -0.021973 | -0.033122 | -0.009684 |  |
| lambada_openai | lm_bootstrap | deterministic_test_slice | 9010 | 42,43,44 | tokenwise_minus_static_mixture |  |  |  |  |  |  |  |  |  |  |  | -0.007548 | -0.01 | -0.005519 | 0.004476 | 0.000244 | 0.009033 |  |
