# Paper Draft Scaffold

This directory now contains both:

- a markdown-first draft built from the paper-facing sources in [notes/paper](</E:/lab/latent-delegation/notes/paper>)
- a LaTeX submission scaffold for the same frozen `v0.6.0` paper narrative

Current source bundle:

- [abstract.md](</E:/lab/latent-delegation/notes/paper/abstract.md>)
- [introduction.md](</E:/lab/latent-delegation/notes/paper/introduction.md>)
- [related_work.md](</E:/lab/latent-delegation/notes/paper/related_work.md>)
- [method.md](</E:/lab/latent-delegation/notes/paper/method.md>)
- [experiments.md](</E:/lab/latent-delegation/notes/paper/experiments.md>)
- [results.md](</E:/lab/latent-delegation/notes/paper/results.md>)
- [limitations.md](</E:/lab/latent-delegation/notes/paper/limitations.md>)
- [conclusion.md](</E:/lab/latent-delegation/notes/paper/conclusion.md>)
- [claim_boundary.md](</E:/lab/latent-delegation/notes/paper/claim_boundary.md>)
- [references.md](</E:/lab/latent-delegation/notes/paper/references.md>)
- [tables.md](</E:/lab/latent-delegation/notes/paper/tables.md>)
- [figures.md](</E:/lab/latent-delegation/notes/paper/figures.md>)

The scaffold exists to keep a paper draft organized without changing any experimental artifact numbers.

Current draft files:

- [manuscript.md](</E:/lab/latent-delegation/paper_draft/manuscript.md>)
- [appendix.md](</E:/lab/latent-delegation/paper_draft/appendix.md>)
- [main.tex](</E:/lab/latent-delegation/paper_draft/main.tex>)
- [appendix.tex](</E:/lab/latent-delegation/paper_draft/appendix.tex>)
- [references.bib](</E:/lab/latent-delegation/paper_draft/references.bib>)
- [build.ps1](</E:/lab/latent-delegation/paper_draft/build.ps1>)

## LaTeX Build

If `latexmk` is available on `PATH`, build with:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_draft\build.ps1
```

The current workspace does not bundle a TeX toolchain, so the build script expects a local TeX installation.
