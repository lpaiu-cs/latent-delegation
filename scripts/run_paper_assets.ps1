$ErrorActionPreference = "Stop"

py -3.12 -m src.tools.paper_assets `
  --repo-root . `
  --tables-dir artifacts/paper_tables `
  --figures-dir artifacts/paper_figures `
  --tables-note notes/paper/tables.md `
  --figures-note notes/paper/figures.md
