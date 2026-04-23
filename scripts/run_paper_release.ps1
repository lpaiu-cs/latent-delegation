$ErrorActionPreference = "Stop"

py -3.12 -m src.tools.paper_release `
  --repo-root . `
  --output-dir artifacts/paper_release `
  --note-path notes/paper/reproducibility.md
