# Related Work

This project is positioned between representation-alignment work and negative results on cross-scale transfer.

Recent representation-transfer work argues that related models can admit simple affine maps between hidden spaces. The Linear Representation Transferability Hypothesis studies affine maps between models of different scales and shows that transferred directions can preserve useful behavior. Recent language-model stitching work similarly shows that residual-stream features can be transferred across models with lightweight mappings. These results motivate the entry and return adapters used here, but they do not by themselves show that a smaller model can replace a missing block of a larger model's forward computation.

This paper also takes explicit guidance from negative results. Neural Incompatibility argues that cross-scale parametric knowledge transfer faces a real structural barrier even with alignment. We treat that as a design warning. The system therefore stays in a same-family setting, keeps both backbones frozen, and evaluates against strong large-space bridge controls rather than assuming that layer windows are interchangeable by default.

The same-family choice is grounded in the Gemma 2 technical report. Gemma-2 2B and 9B share architecture family structure, including RMSNorm, RoPE, grouped-query attention, GeGLU MLP blocks, and the alternating local/global attention pattern. That makes same-family delegation plausible enough to test, while still leaving room for genuine cross-scale mismatch.

For broader evaluation, the paper uses standard log-likelihood-compatible benchmarks: HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, and LAMBADA. In paper citation form these correspond to HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC (Clark et al., 2018), and LAMBADA (Paperno et al., 2016). The exact source URLs used for repo preparation are collected in [references.md](</E:/lab/latent-delegation/notes/references.md>).
