# Idea 4 Token-Wise Report

## setup

- Model family and split remained fixed: Gemma 2 9B / 2B, large removed window `24..27`, shortlisted small paths `{14..19, 16..18}` only.
- Warm start policy: token-wise hybrid initialized from the confirmed static-mixture checkpoints; token-wise no-small initialized from the confirmed static-mixture no-small checkpoints.
- Gate design: RMSNorm over the splice-boundary large hidden state, then a minimal per-token 2-logit linear head (`hidden_dim = 0` in config). The gate never inspects teacher outputs.
- Gate regularization: entropy `1e-4`, KL-to-static-prior `1e-3`, smoothness `0.0`.

## fairness audit

- static_mixture trainable params: `753666`
- static_mixture_no_small trainable params: `753666`
- tokenwise_mixture trainable params: `764418`
- tokenwise_mixture_no_small trainable params: `764418`
- bridge_only trainable params: `458753`
- updated parameter-matched bridge rank: `107`
- updated parameter-matched bridge trainable params: `766977`

## hidden diagnostics

- tokenwise_mixture: hidden MSE `6.535482 ± 0.035669`, cosine `0.876766 ± 0.000712`.
- tokenwise_mixture_no_small: hidden MSE `10.207031 ± 0.242004`, cosine `0.868963 ± 0.000108`.
- static_mixture reference: hidden MSE `7.150391 ± 0.078781`, cosine `0.868993 ± 0.000838`.
- bridge_only reference: hidden MSE `13.438151 ± 0.222119`, cosine `0.845490 ± 0.001836`.
- updated parameter-matched bridge: hidden MSE `14.205078 ± 0.091547`, cosine `0.838460 ± 0.000429`.

## gate behavior

- tokenwise_mixture mean weights: path B `0.411437 ± 0.022216`, path A `0.588602 ± 0.022183`.
- tokenwise_mixture entropy: `0.602399 ± 0.007528`.
- tokenwise_mixture collapse score: `0.054150 ± 0.041328`.
- tokenwise_mixture per-token usage variance: path B `0.027443 ± 0.004226`, path A `0.027432 ± 0.004217`.
- tokenwise_mixture weighted delta norms: path B `75.874719 ± 3.319377`, path A `74.828257 ± 3.662180`.
- tokenwise_mixture_no_small is sharper and less stable: entropy `0.497769 ± 0.008935`, collapse score `0.218956 ± 0.012151`.

## interpretation

- The token-wise gate does not collapse to one path almost everywhere.
- The gate also does not stay near a trivial uniform split; it moves toward the shorter `16..18` path on average while retaining nontrivial per-token variance.
- The no-small control uses the same gate family but shows higher collapse and materially weaker hidden recovery, which argues against a pure route-capacity explanation.
