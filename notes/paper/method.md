# Method

## Base Interface

Let `h_t^L` denote the large-model hidden state at token position `t` after the frozen large prefix, and let `N(.)` denote RMSNorm. For each delegated path `p`, let `E_p` be the entry projector into small latent space, `S_p` the frozen delegated small-model window, and `R_p` the return adapter back into large hidden space. The path-specific returned delta is

`Delta_{p,t} = R_p(S_p(E_p(N(h_t^L))))`.

In the single-path case, the hybrid hidden state after the removed large block is

`h_t^H = h_t^L + Delta_t`,

after which the frozen large suffix produces the final logits.

## Static Two-Path Mixture

The first successful continuation model activates two delegated paths in parallel: path B, which maps `24..27 -> 14..19`, and path A, which maps `24..27 -> 16..18`. In the static two-path mixture, the returned deltas are combined by a learned global two-logit softmax,

`w = softmax(alpha)`,

`Delta_t = w_B Delta_{B,t} + w_A Delta_{A,t}`.

The matched no-small control keeps the same entry projectors, return adapters, and global mixture weights but removes actual delegated small-model computation.

## Token-Wise Two-Path Routing

The final model replaces the global mixture with a low-capacity token-wise router. The gate reads only the large-prefix hidden state at the splice boundary. In the confirmed final configuration, the gate is a direct linear head over RMS-normalized prefix states:

`g_t = softmax(W_g N(h_t^L) + b_g)`,

`Delta_t = g_{t,B} Delta_{B,t} + g_{t,A} Delta_{A,t}`.

The token-wise no-small control uses the same gate family and the same interface routes but removes delegated small-model computation.

## Training Objectives

Stage A trains only the entry projector and aligns the large-model splice state to the frozen small-model reference state:

`L_A = MSE(E(h^L), h_ref^S) + CosineLoss(E(h^L), h_ref^S)`.

The decisive training regime is output-aware Stage B. Its implemented objective is

`L_B = MSE(h^H, h^T) + CosineLoss(h^H, h^T) + lambda_KL L_KL + lambda_CE L_CE + lambda_D ||Delta||_2^2`,

where `h^T` is the frozen large-model hidden state after the removed large block, `L_KL` is teacher-logit KL, and `L_CE` is shifted next-token cross-entropy. In the confirmed runs, `lambda_KL = 5.0`, `lambda_CE = 1.0`, and `lambda_D = 1e-4`. The token-wise gate adds only a small stability package: a weak entropy term, a weak KL-to-static-prior term, and no temporal smoothness term in the confirmed final configuration. Stage C is intentionally not used in this paper.

## Fairness And Parameter Budgets

Parameter matching is explicit because routing adds trainable capacity beyond a plain large-space bridge. The static two-path mixture uses `753,666` trainable parameters, the token-wise two-path router uses `764,418`, and the matched token-wise no-small control uses the same budget. The plain bridge baseline uses `458,753` trainable parameters, so we also include an updated parameter-matched bridge with `766,977` trainable parameters. This fairness audit matters because the core scientific comparison is not whether the hybrid works at all, but whether delegated small-model computation adds value beyond a strong use of comparable trainable capacity in the large model's own latent space.
