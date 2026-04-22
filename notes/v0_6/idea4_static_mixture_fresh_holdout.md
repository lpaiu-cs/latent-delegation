# Idea 4 Static Mixture Fresh Holdout

## slice definition

- Held-out slice: `wikitext-103-v1` test split.
- Sampling seed: `7606`.
- Sample count: `32` sequences at `seq_len = 256`.
- Exact IDs: `artifacts/v0_6/idea4_static_mixture/fresh_holdout_probe/sample_ids.json`.
- Rationale: prior v0.5.x and v0_6 model-selection probes reused seed-matched validation slices; this confirmation slice comes from the untouched test split.

## output summary

- static_mixture: KL `0.267244 ± 0.001267`, NLL `3.213048 ± 0.003226`, PPL `24.854807 ± 0.080249`, top-1 `0.753466 ± 0.002291`, top-5 `0.740155 ± 0.001408`.
- best single-path reference (`24..27 -> 14..19`): dKL `-0.015962`, dNLL `-0.094426`, primary wins `3/3`.
- `bridge_only`: dKL `-0.022320`, dNLL `-0.082033`, primary wins `3/3`.
- mixture-budget parameter-matched bridge: dKL `-0.017189`, dNLL `-0.049553`, primary wins `3/3`.
- static_mixture_no_small: dKL `-0.000078`, dNLL `-0.083299`, primary wins `1/3`.

## decision

- The static-mixture bridge win survives on the untouched holdout and is now externally stronger than the original reused-validation claim.
- The static-mixture gain over the no-small control remains mixed on KL on this slice, even though NLL still improves materially.
- The fresh holdout therefore strengthens the structural claim needed for token-wise Idea 4 continuation without pretending that every secondary comparison is equally stable.
