# Stage B Entry-Tune Report

## setup

- Seeds: 42, 43, 44
- Tuned variants: hybrid, hybrid_no_small
- Train entry projector: True
- Stage B learning rates: base=0.0003, entry=0.00015, return=0.0003, gate=0.0003
- Reference bridge results were reused from the frozen-entry output-aware Stage B run.

## aggregate hidden summary

- skip_only_reference: hidden_mse_mean=25.509766, cosine_mean=0.746104, gate_value_mean=0.000000, delta_norm_mean=0.000000
- bridge_only_reference: hidden_mse_mean=22.161458, cosine_mean=0.782511, gate_value_mean=0.069661, delta_norm_mean=1206.653747
- bridge_only_param_matched_reference: hidden_mse_mean=22.436198, cosine_mean=0.781138, gate_value_mean=0.069661, delta_norm_mean=1180.169830
- hybrid_frozen_entry: hidden_mse_mean=22.847656, cosine_mean=0.786835, gate_value_mean=0.068197, delta_norm_mean=1317.256423
- hybrid_train_entry: hidden_mse_mean=21.162109, cosine_mean=0.795959, gate_value_mean=0.057454, delta_norm_mean=1720.684966
- hybrid_no_small_frozen_entry: hidden_mse_mean=24.585286, cosine_mean=0.763743, gate_value_mean=0.070150, delta_norm_mean=843.765400
- hybrid_no_small_train_entry: hidden_mse_mean=23.944010, cosine_mean=0.776896, gate_value_mean=0.061768, delta_norm_mean=1280.483346

## entry diagnostics

- hybrid: entry_grad_norm_mean=0.229204, entry_grad_norm_std=0.016586, final_entry_update_norm_mean=7.616570, final_entry_update_norm_std=0.149043
- hybrid_no_small: entry_grad_norm_mean=0.027438, entry_grad_norm_std=0.002578, final_entry_update_norm_mean=11.701016, final_entry_update_norm_std=0.075224

## interpretation

- hybrid train-entry vs frozen-entry: mse_delta_mean=-1.685547, cosine_delta_mean=0.009125, wins_on_both=3/3
- hybrid_no_small train-entry vs frozen-entry: mse_delta_mean=-0.641276, cosine_delta_mean=0.013153, wins_on_both=3/3
- hybrid gap to bridge_only: frozen_gap_mse_mean=0.686198, tuned_gap_mse_mean=-0.999349, frozen_gap_cosine_mean=-0.004323, tuned_gap_cosine_mean=-0.013448
- hybrid gap to bridge_only_param_matched: frozen_gap_mse_mean=0.411458, tuned_gap_mse_mean=-1.274089, frozen_gap_cosine_mean=-0.005697, tuned_gap_cosine_mean=-0.014821
- Output-level interpretation is deferred to the dedicated output-probe comparison report.
