# References

This note uses concise, implementation-oriented takeaways from primary sources available on 2026-04-20.

## 1. Gemma 2 technical report / model docs

Sources:
- https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf
- https://huggingface.co/google/gemma-2-9b

Key takeaways:
- Gemma 2 is a decoder-only transformer family with same-family scaling from 2B to 27B.
- The 2B model uses `d_model=2304` and `26` layers; the 9B model uses `d_model=3584` and `42` layers.
- Gemma 2 alternates local sliding-window attention and global attention, so delegated windows should preserve native block ordering rather than invent a custom subnetwork.
- Gemma 2 uses RMSNorm, RoPE, GeGLU, grouped-query attention, and final logit soft-capping; any hybrid execution path must preserve these frozen backbone details.
- The 2B and 9B models were trained with knowledge distillation, which makes same-family latent compatibility more plausible than a random cross-family pairing.
- Hugging Face model docs show standard `transformers` loading paths, including 4-bit `bitsandbytes` quantization for single-accelerator use.

Supports this project assumption:
- Same-family Gemma 2 backbones likely have partially aligned residual spaces, making affine entry/return adapters plausible for a conservative feasibility test.

Risk / limitation implied:
- Similar family membership does not guarantee layerwise interchangeability; Gemma 2 architectural details like alternating attention types and soft-capping may make some layer windows harder to splice cleanly.

## 2. LRT / Linear Representation Transferability

Source:
- https://arxiv.org/abs/2506.00653

Key takeaways:
- The paper proposes the Linear Representation Transferability hypothesis: an affine transformation may exist between representation spaces of related models.
- It tests learned affine mappings across different model sizes, not just same-size checkpoints.
- Steering vectors transferred through learned mappings can preserve semantic effects in larger models.
- The central positive claim is representational transferability, not full state equivalence.

Supports this project assumption:
- A simple affine entry projector from large residual space to small residual space is a defensible first v1 interface.

Risk / limitation implied:
- The paper demonstrates transfer of behaviors and directions, not guaranteed faithful substitution of a missing computation block inside a forward pass.

## 3. Model stitching for language models

Source:
- https://arxiv.org/abs/2506.06609

Key takeaways:
- Affine mappings between residual streams can transfer features between small and large language models at low cost.
- The paper finds that small and large models learn similar representation spaces, which is strong motivation for same-family stitching experiments.
- Transferred probes and steering vectors can recover useful performance signals after mapping.
- Feature transfer is uneven: semantic, structural, and functional features do not all transfer equally well.

Supports this project assumption:
- A learned interface around a delegated small-model block can be treated as a model-stitching problem in residual space.

Risk / limitation implied:
- Feature overlap is partial, so a delegated window may recover some functions of the removed large block while still failing on others; failure analysis must stay feature-local and task-local.

## 4. Neural Incompatibility

Source:
- https://arxiv.org/abs/2505.14436

Key takeaways:
- Cross-scale parametric transfer requires alignment; raw parameter reuse is not enough.
- Even with explicit pre-alignment or post-alignment, transfer remains unstable across benchmarks.
- The paper argues there are structural differences across model scales that create a real incompatibility barrier.
- The negative result is about reliable transfer, not just one-off qualitative success.

Supports this project assumption:
- Alignment modules are necessary; a direct layer swap without learned adapters is not a credible baseline.

Risk / limitation implied:
- This project may fail even with same-family models, and a negative result would still be consistent with current evidence; the codebase must surface failures clearly instead of overclaiming.

## 5. COCONUT

Source:
- https://arxiv.org/abs/2412.06769

Key takeaways:
- COCONUT uses the last hidden state as a continuous reasoning state and feeds it back as the next input embedding.
- The paper treats latent states as useful computational objects, not only as byproducts of token decoding.
- Continuous latent reasoning can support search-like behavior without forcing every reasoning step through text.
- The setup remains within one model’s learned latent manifold.

Supports this project assumption:
- It is reasonable to treat hidden states as interfaces for computation, so a delegated latent block is a meaningful object to optimize.

Risk / limitation implied:
- COCONUT does not show cross-model latent compatibility; it should not be used to justify any “thought transfer” claim for this project.

## 6. QLoRA / bitsandbytes quantization docs

Sources:
- https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf
- https://huggingface.co/docs/transformers/en/quantization/bitsandbytes

Key takeaways:
- QLoRA shows that 4-bit NF4 quantization can make very large models trainable or runnable on a single GPU by keeping the base model frozen and only training extra parameters.
- HF `bitsandbytes` docs recommend `bnb_4bit_compute_dtype=torch.bfloat16` and `bnb_4bit_quant_type="nf4"` for 4-bit setups.
- HF docs explicitly note that 8-bit and 4-bit training are only supported for training extra parameters, which matches this project’s frozen-backbone requirement.
- Nested quantization can reduce memory further, but it is optional and should not complicate v1.
- QLoRA improves memory feasibility, not necessarily speed.

Supports this project assumption:
- Frozen 4-bit Gemma backbones plus small trainable adapters are the right default for a one-GPU feasibility project.

Risk / limitation implied:
- Quantized partial-layer execution can be awkward in practice, and 4-bit runtime overhead may reduce speed gains from delegation; the code should support falling back to bf16 debug execution when needed.
