# Idea 4 Token-Wise Output Probe

## original holdout

- Hold-out policy: reused seed-matched validation slices, exactly as in the prior confirmation path.
- tokenwise_mixture: KL `0.255739 ± 0.016955`, NLL `2.980182 ± 0.105470`, PPL `19.763760 ± 2.050592`, top-1 `0.763351 ± 0.008271`, top-5 `0.744268 ± 0.004733`.
- vs static_mixture: dKL `-0.011356`, dNLL `-0.020256`, primary wins `3/3`.
- vs tokenwise_mixture_no_small: dKL `-0.001761`, dNLL `-0.058422`, primary wins `3/3`.
- vs bridge_only: dKL `-0.032709`, dNLL `-0.091869`, primary wins `3/3`.
- vs updated parameter-matched bridge: dKL `-0.046584`, dNLL `-0.121899`, primary wins `3/3`.

## fresh untouched holdout

- Hold-out policy: shared untouched `wikitext-103-v1` test slice, sampling seed `7606`, sample count `32`.
- tokenwise_mixture: KL `0.248886 ± 0.003762`, NLL `3.185004 ± 0.005495`, PPL `24.167632 ± 0.132829`, top-1 `0.758319 ± 0.001101`, top-5 `0.742734 ± 0.001985`.
- vs static_mixture: dKL `-0.018358`, dNLL `-0.028044`, primary wins `3/3`.
- vs tokenwise_mixture_no_small: dKL `-0.002408`, dNLL `-0.076782`, primary wins `2/3`.
- vs bridge_only: dKL `-0.040677`, dNLL `-0.110077`, primary wins `3/3`.
- vs updated parameter-matched bridge: dKL `-0.052859`, dNLL `-0.142020`, primary wins `3/3`.

## readout

- The static-mixture gain over bridges survives the untouched holdout.
- The token-wise gate then improves further over static mixture on both the original holdout and the untouched holdout.
- The no-small control remains weaker overall, but the fresh holdout comparison is less clean than the bridge/static comparisons because one seed misses the joint KL/NLL win condition.
