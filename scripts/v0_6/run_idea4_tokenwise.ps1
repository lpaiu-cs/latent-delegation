param(
    [ValidateSet("coarse", "confirm", "probe_main", "probe_fresh")]
    [string]$Mode = "coarse"
)

$ErrorActionPreference = "Stop"
Set-Location "E:\lab\latent-delegation"

switch ($Mode) {
    "coarse" {
        py -3.12 -m src.v0_6.idea4_tokenwise `
            --config configs/v0_6/idea4/tokenwise_coarse_seed42.yaml `
            --static-stage-dir artifacts/v0_6/idea4_static_mixture/confirm/stage_b `
            --output-dir artifacts/v0_6/idea4_tokenwise/coarse/stage_b `
            --results-path artifacts/v0_6/idea4_tokenwise/coarse/stage_b/results.json `
            --summary-path artifacts/v0_6/idea4_tokenwise/coarse/stage_b/summary.csv `
            --diagnostics-path artifacts/v0_6/idea4_tokenwise/coarse/stage_b/diagnostics.json `
            --report-path notes/v0_6/_scratch_idea4_tokenwise_coarse_stage_b.md `
            --seeds 42
    }
    "confirm" {
        py -3.12 -m src.v0_6.idea4_tokenwise `
            --config configs/v0_6/idea4/tokenwise_confirm_seed42.yaml `
            --static-stage-dir artifacts/v0_6/idea4_static_mixture/confirm/stage_b `
            --output-dir artifacts/v0_6/idea4_tokenwise/confirm/stage_b `
            --results-path artifacts/v0_6/idea4_tokenwise/confirm/stage_b/results.json `
            --summary-path artifacts/v0_6/idea4_tokenwise/confirm/stage_b/summary.csv `
            --diagnostics-path artifacts/v0_6/idea4_tokenwise/confirm/stage_b/diagnostics.json `
            --report-path notes/v0_6/_scratch_idea4_tokenwise_confirm_stage_b.md `
            --seeds 42 43 44
    }
    "probe_main" {
        py -3.12 -m src.v0_6.idea4_tokenwise_output_probe `
            --config configs/v0_6/idea4/tokenwise_confirm_seed42.yaml `
            --static-stage-dir artifacts/v0_6/idea4_static_mixture/confirm/stage_b `
            --tokenwise-stage-dir artifacts/v0_6/idea4_tokenwise/confirm/stage_b `
            --output-dir artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main `
            --results-path artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/results.json `
            --summary-path artifacts/v0_6/idea4_tokenwise/confirm/output_probe_main/summary.csv `
            --report-path notes/v0_6/_scratch_idea4_tokenwise_confirm_output_probe_main.md `
            --holdout-policy main_validation `
            --seeds 42 43 44
    }
    "probe_fresh" {
        py -3.12 -m src.v0_6.idea4_tokenwise_output_probe `
            --config configs/v0_6/idea4/tokenwise_confirm_seed42.yaml `
            --static-stage-dir artifacts/v0_6/idea4_static_mixture/confirm/stage_b `
            --tokenwise-stage-dir artifacts/v0_6/idea4_tokenwise/confirm/stage_b `
            --output-dir artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout `
            --results-path artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/results.json `
            --summary-path artifacts/v0_6/idea4_tokenwise/confirm/output_probe_fresh_holdout/summary.csv `
            --report-path notes/v0_6/_scratch_idea4_tokenwise_confirm_output_probe_fresh.md `
            --holdout-policy fresh_untouched `
            --seeds 42 43 44
    }
}
