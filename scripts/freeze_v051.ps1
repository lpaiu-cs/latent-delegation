param()

$ErrorActionPreference = "Stop"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

python -m src.tools.freeze_v051
