# Introduction

This paper asks a narrow question: can a large same-family model keep the master residual stream and final logits while delegating part of its middle computation to a smaller same-family model through latent-space transfer? The framing is intentionally bounded. We do not claim thought transfer, model equivalence, or broad downstream superiority. We ask whether delegated latent computation can recover useful work under strict practical constraints.

Those constraints are deliberate. The target setting is a single RTX 5090-class GPU, same-family open models only, frozen backbones, and lightweight learned interface modules. The default pair is Gemma-2 9B and Gemma-2 2B. This keeps the experiment interpretable: if the hybrid fails, it fails in a realistic frozen-backbone regime; if it succeeds, the success can be tied to a concrete latent substitution mechanism rather than to large-scale finetuning or distributed training.

The fixed-window baseline gives only a qualified feasibility result. Delegated small-model computation improves over `skip_only` and no-small controls, but not over strong bridge baselines that remain entirely in large hidden space. That leaves a structural question unresolved: does delegation fail in principle, or is the original fixed contiguous layer match simply the wrong prior?

The answer is the main result of the paper. A real-model asymmetric window search rejects the original fixed contiguous substitution and identifies a near-tied two-window shortlist. A static two-path mixture over that shortlist is the first model to beat both bridge controls on KL and NLL, and a low-capacity token-wise gate over the same two paths improves further. The resulting final model beats both bridge controls on the development holdout and again on an untouched confirmation holdout.

This paper is closest in spirit to recent representation-transfer results such as the Linear Representation Transferability Hypothesis and language-model model stitching, but asks a harder operational question: not whether features or steering directions can be mapped between models, but whether a small model can replace a missing block of a larger model's forward computation under frozen same-family constraints. At the same time, recent neural incompatibility results motivate strong bridge controls and a conservative claim boundary rather than an assumption of easy cross-scale interchangeability.

The contributions are:

1. We show that the original fixed contiguous substitution is not the right structural prior; a bounded asymmetric window search identifies a near-tied local shortlist instead.
2. We show that a static two-path mixture and then a low-capacity token-wise two-path routing model can beat strong bridge baselines on held-out LM-style probes in the frozen same-family Gemma-2 9B -> 2B setting.
3. We strengthen the explanation of the result with monotone-corridor analysis and sublayer attribution, and we show that broader external generalization is mixed rather than broad.
