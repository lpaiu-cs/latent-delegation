# Method

## Base Framing

The large model owns the input path, the master residual stream, the suffix, and the final logits. The small model owns delegated latent computation only. All backbone weights remain frozen. The trainable modules are limited to entry projectors, return adapters, and a low-capacity routing mechanism.

The original fixed-window baseline removes large layers `24..29` and replaces them with delegated small layers `14..19` entered from the hidden state after small layer `13`. The final successful model fixes the removed large window to `24..27` and uses two delegated paths:

- path B: `24..27 -> 14..19`
- path A: `24..27 -> 16..18`

## Notation And Forward Definitions

Let `h_t^L` denote the large-model hidden state at token position `t` after the frozen large prefix. Let `N(.)` denote RMSNorm, `E_p` the entry projector for path `p`, `S_p` the frozen delegated small-model window, and `R_p` the return adapter back into large hidden space.

For each delegated path `p in {A, B}`, the path delta is:

`Delta_{p,t} = R_p(S_p(E_p(N(h_t^L))))`

The hybrid hidden state after the removed large window is:

`h_t^H = h_t^L + Delta_t`

and the frozen large suffix then produces the final logits.

## Static Two-Path Mixture

The static mixture keeps both delegated paths active and combines their returned deltas with a learnable global two-logit softmax:

`w = softmax(alpha)`

`Delta_t = w_B * Delta_{B,t} + w_A * Delta_{A,t}`

The matched no-small control keeps the same entry projectors, return adapters, and global mixture weights, but removes the actual delegated small-model computation.

## Token-Wise Two-Path Routing

The final model replaces the global mixture with a low-capacity per-token gate. The gate input is only the large-prefix hidden state at the splice boundary. In the final confirmed model, the gate is a direct linear head over RMS-normalized prefix states, because the configured gate hidden size is `0`:

`g_t = softmax(W_g N(h_t^L) + b_g)`

`Delta_t = g_{t,B} * Delta_{B,t} + g_{t,A} * Delta_{A,t}`

The gate bias is initialized from the learned static-mixture prior. The token-wise no-small control uses the same gate family and interface routes but removes delegated small-model computation.

## Training Objectives

Stage A trains only the entry projector. Its exact objective in code is:

`L_A = MSE(E(h^L), h^S_ref) + CosineLoss(E(h^L), h^S_ref)`

where `h^S_ref` is the frozen small-model reference hidden state before the delegated small block.

The decisive training regime is output-aware Stage B. Its implemented objective is:

`L_B = MSE(h^H, h^T) + CosineLoss(h^H, h^T) + 5.0 * L_KL + 1.0 * L_CE + 1e-4 * ||Delta||_2^2`

Here `h^T` is the frozen large-model hidden state after the removed large block, `L_KL` is the teacher-logit KL term, and `L_CE` is shifted next-token cross-entropy. The token-wise gate adds a small prior/stability package:

- entropy penalty weight: `1e-4`
- KL-to-static-prior weight: `1e-3`
- smoothness penalty weight: `0.0`

Stage C is not used in this paper.

## Fairness And Parameter Budgets

The paper keeps parameter matching explicit because the routing models add capacity beyond a plain bridge:

| model | trainable parameters |
| --- | ---: |
| static two-path mixture | `753666` |
| static two-path mixture no-small | `753666` |
| token-wise two-path routing | `764418` |
| token-wise no-small | `764418` |
| `bridge_only` | `458753` |
| updated parameter-matched bridge | `766977` |

The updated parameter-matched bridge uses rank `107`, which is the closest saved match to the token-wise routing budget.
