# Figure Specs

These are figure-ready specs, not new results. The specs live under `artifacts/paper_figures/` and are derived only from frozen artifacts and notes.

## Figure 1. Timeline / Milestone Progression

- Spec-only figure. Use a milestone timeline or ladder chart.
- The figure should show why `v0.6.0` stays frozen as the best branch while later branches remain analytical or evaluative.
- Spec file: `figure_01_timeline_milestones.json`.
- Source artifacts: `notes/final_report.md`, `notes/v0_6/idea4_tokenwise_combined_decision.md`, `notes/v0_7/idea5_combined_decision.md`, `notes/v0_8/idea2_combined_decision.md`, `notes/v0_9/generalization_results.md`.

| milestone | status | claim_status |
| --- | --- | --- |
| v0.4.0 | bring_up_complete | same_family_path_runs_on_single_gpu |
| v0.5.0 | output_aware_stage_b_added | hybrid_beats_skip_and_no_small_but_not_bridges |
| v0.5.1 | entry_tune_follow_up_complete | qualified_feasibility_result |
| v0.6.0 | tokenwise_idea4_frozen_best | beats_bridges_on_original_and_fresh_holdouts |
| v0_7 | analysis_only | corridor_hypothesis_strengthened_not_best_model |
| v0_8 | analysis_only | attention_and_mlp_both_needed |
| v0_9 | bounded_generalization | mixed_external_validity |

## Figure 2. Legacy Split to Token-Wise Progression

- Spec-only figure. Plot lower-is-better KL and NLL across the structural progression.
- This figure is the clearest one-panel summary of why the repo stops at `v0.6.0` rather than at `v0.5.1`.
- Spec file: `figure_02_structural_progression.json`.
- Source artifacts: `artifacts/v0_6/phase1_real/combined/ranking_summary.json`, `artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json`, `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json`.

| stage_name | mapping | seed_count | holdout_policy | logit_kl_to_teacher | nll |
| --- | --- | --- | --- | --- | --- |
| legacy_split_coarse | 24..29 -> 14..19 | 1 | phase1_real_coarse_output_probe | 0.72503 | 3.425046 |
| best_single_path_confirm | 24..27 -> 14..19 | 3 | phase1_real_confirmation_output_probe | 0.281641 | 3.078029 |
| static_mixture_main | softmix(path_b,path_a) | 3 | stage_b validation split reused for the matching seed | 0.267095 | 3.000438 |
| tokenwise_main | tokenwise(path_b,path_a) | 3 | stage_b validation split reused for the matching seed | 0.255739 | 2.980182 |

## Figure 3. Development vs Untouched Confirmation Holdout KL/NLL

- Spec-only figure. The same model set should be plotted on both holdouts.
- The key reading is whether the token-wise bridge win survives the untouched confirmation slice.
- Spec file: `figure_03_original_vs_fresh_holdout.json`.
- Source artifacts: `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json`, `artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json`, `artifacts/v0_6/idea4_static_mixture/confirm/output_probe/results.json`, `artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/results.json`.

| holdout_name | model_name | seed_count | logit_kl_to_teacher_mean | nll_mean |
| --- | --- | --- | --- | --- |
| development_holdout | tokenwise_mixture | 3 | 0.255739 | 2.980182 |
| development_holdout | static_mixture | 3 | 0.267095 | 3.000438 |
| development_holdout | tokenwise_mixture_no_small | 3 | 0.257501 | 3.038605 |
| development_holdout | bridge_only | 3 | 0.288448 | 3.072051 |
| development_holdout | bridge_only_param_matched | 3 | 0.302323 | 3.102081 |
| untouched_confirmation_holdout | tokenwise_mixture | 3 | 0.248886 | 3.185004 |
| untouched_confirmation_holdout | static_mixture | 3 | 0.267244 | 3.213048 |
| untouched_confirmation_holdout | tokenwise_mixture_no_small | 3 | 0.251294 | 3.261786 |
| untouched_confirmation_holdout | bridge_only | 3 | 0.289564 | 3.295081 |
| untouched_confirmation_holdout | bridge_only_param_matched | 3 | 0.301746 | 3.327024 |

## Figure 4. Idea 5 Corridor Visualization Summary

- Spec-only figure. A corridor heatmap or ranked strip chart both fit.
- The figure should make it visually obvious that the successful Idea 4 shortlist lives in a broader low-cost corridor.
- Spec file: `figure_04_idea5_corridor.json`.
- Source artifacts: `artifacts/v0_7/idea5_discovery/solver/top_paths.json`.

| name | mapping | combined_proxy_cost |
| --- | --- | --- |
| shortlist_path_a | 24..27 -> 16..18 | 0.237026 |
| shortlist_path_b | 24..27 -> 14..19 | 0.261984 |
| derived_midpoint | 24..27 -> 15..18 | 0.270362 |
| phase1_candidate_c | 25..29 -> 15..19 | 0.307316 |
| legacy_fixed_split | 24..29 -> 14..19 | 0.315038 |

## Figure 5. Idea 2 Attribution Deltas

- Spec-only figure. Lower deltas are better because all rows are degradations relative to the full token-wise baseline.
- The intended reading is that both attention and MLP matter, with larger degradation when MLP is suppressed.
- Spec file: `figure_05_idea2_attribution.json`.
- Source artifacts: `artifacts/v0_8/idea2_attribution/results.json`.

| holdout_name | variant_name | logit_kl_to_teacher_delta | nll_delta |
| --- | --- | --- | --- |
| untouched_confirmation_holdout | tokenwise_attn_suppressed | 0.109496 | 0.224608 |
| untouched_confirmation_holdout | tokenwise_mlp_suppressed | 0.182872 | 0.37961 |
| untouched_confirmation_holdout | tokenwise_both_suppressed | 0.154198 | 0.379402 |
| development_holdout | tokenwise_attn_suppressed | 0.103797 | 0.21867 |
| development_holdout | tokenwise_mlp_suppressed | 0.182743 | 0.350897 |
| development_holdout | tokenwise_both_suppressed | 0.147248 | 0.332872 |

## Figure 6. Generalization Summary Across Benchmarks

- Spec-only figure. Use separate facets for multiple-choice accuracy and LM NLL.
- This is the figure that should visually enforce the mixed-generalization claim boundary.
- Spec file: `figure_06_generalization_summary.json`.
- Source artifacts: `artifacts/v0_9/generalization/aggregated/summary.json`.

| task_name | task_type | model_name | primary_metric_name | primary_metric_value |
| --- | --- | --- | --- | --- |
| arc_challenge | multichoice | tokenwise_mixture | accuracy_mean | 0.442708 |
| arc_challenge | multichoice | static_mixture | accuracy_mean | 0.427083 |
| arc_challenge | multichoice | bridge_only | accuracy_mean | 0.432292 |
| arc_challenge | multichoice | bridge_only_param_matched | accuracy_mean | 0.4375 |
| arc_easy | multichoice | tokenwise_mixture | accuracy_mean | 0.791667 |
| arc_easy | multichoice | static_mixture | accuracy_mean | 0.796875 |
| arc_easy | multichoice | bridge_only | accuracy_mean | 0.828125 |
| arc_easy | multichoice | bridge_only_param_matched | accuracy_mean | 0.828125 |
| hellaswag | multichoice | tokenwise_mixture | accuracy_mean | 0.671875 |
| hellaswag | multichoice | static_mixture | accuracy_mean | 0.671875 |
| hellaswag | multichoice | bridge_only | accuracy_mean | 0.65625 |
| hellaswag | multichoice | bridge_only_param_matched | accuracy_mean | 0.65625 |
| piqa | multichoice | tokenwise_mixture | accuracy_mean | 0.723958 |
| piqa | multichoice | static_mixture | accuracy_mean | 0.723958 |
| piqa | multichoice | bridge_only | accuracy_mean | 0.734375 |
| piqa | multichoice | bridge_only_param_matched | accuracy_mean | 0.744792 |
| winogrande | multichoice | tokenwise_mixture | accuracy_mean | 0.645833 |
| winogrande | multichoice | static_mixture | accuracy_mean | 0.65625 |
| winogrande | multichoice | bridge_only | accuracy_mean | 0.635417 |
| winogrande | multichoice | bridge_only_param_matched | accuracy_mean | 0.65625 |
| lambada_openai | lm | tokenwise_mixture | nll_mean | 3.423984 |
| lambada_openai | lm | static_mixture | nll_mean | 3.419273 |
| lambada_openai | lm | bridge_only | nll_mean | 3.433371 |
| lambada_openai | lm | bridge_only_param_matched | nll_mean | 3.446407 |
