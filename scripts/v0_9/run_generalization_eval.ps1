$ErrorActionPreference = "Stop"

$config = "configs/v0_9/generalization_eval.yaml"
$multichoiceDir = "artifacts/v0_9/generalization/raw/multichoice"
$lmDir = "artifacts/v0_9/generalization/raw/lm"
$aggregatedDir = "artifacts/v0_9/generalization/aggregated"
$resultsNote = "notes/v0_9/generalization_results.md"
$paperNote = "notes/v0_9/generalization_summary_for_paper.md"

py -3.12 -m src.v0_9.eval_multichoice `
  --config $config `
  --output-dir $multichoiceDir `
  --results-path "$multichoiceDir/results.json" `
  --summary-path "$multichoiceDir/summary.csv"

py -3.12 -m src.v0_9.eval_lm_generalization `
  --config $config `
  --output-dir $lmDir `
  --results-path "$lmDir/results.json" `
  --summary-path "$lmDir/summary.csv"

py -3.12 -m src.v0_9.aggregate_generalization `
  --config $config `
  --multichoice-dir $multichoiceDir `
  --lm-dir $lmDir `
  --output-dir $aggregatedDir `
  --results-note $resultsNote `
  --paper-summary-note $paperNote
