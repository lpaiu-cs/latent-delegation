# Latent Delegation v0.6+ Research Plan

## 0. Scope and intent

This document is the handoff plan for the post-`v0.5.1` research track in the same repository.

`v0.5.1` should be treated as a **frozen qualified result**:
- same-family Gemma-2 9B -> 2B one-way latent delegation is real and runnable on a single RTX 5090-class GPU,
- the delegated path is active,
- output-aware Stage B makes `hybrid` beat `skip_only` and `hybrid_no_small` at the output level,
- but the current hybrid does **not** beat the strong large-space bridge baselines.

This next track does **not** revise those claims. It starts from them.

The central diagnosis going forward is:

> The current 6-large <-> 6-small contiguous block replacement is likely structurally misaligned.  
> More importantly, forcing discrete layer-to-layer substitution is itself a bottleneck, because the true correspondence may be asymmetric, distributed, or functionally staged rather than one-to-one.

## 1. Terminology for this roadmap

### Current frozen baseline architecture
Default conservative split:
- large prefix: `0..23`
- removed large block: `24..29`
- large suffix: `30..41`
- small reference hidden: after layer `13`
- delegated small block: `14..19`

### Key baselines already established
- `FullLargeModel`
- `SkipOnlyLargeModel`
- `BridgeOnlyLargeModel`
- `BridgeOnlyParamMatched`
- `HybridNoSmallModel`
- `HybridDelegationModel`

### Stage definitions
- **Stage A**: align large hidden state into small latent space.
- **Stage B**: replace / approximate the removed large middle computation.
- **Stage C**: final output-level refinement of the whole hybrid; currently deferred.

### Working diagnosis
The current bottleneck is no longer "does delegation work at all?"  
The bottleneck is "why does a strong large-space bridge still outperform the current delegated replacement?"

## 2. Core hypotheses for the new track

### H1. Asymmetric correspondence hypothesis
A contiguous 6-layer large block probably does **not** correspond cleanly to a contiguous 6-layer small block.  
A better match may look like:
- 6 large -> 3 or 4 small,
- or one large window matched to multiple nearby small windows,
- or stage-level rather than raw depth-level correspondence.

### H2. Functional-stage mismatch hypothesis
Relative depth is only a rough prior.  
Layers should be matched using functional signatures, not only index position.

### H3. Discrete replacement leakage hypothesis
Even with better window choice, hard discrete substitution leaks information:
- some computation may be distributed across unequal numbers of layers,
- some functionality may not be representable as one contiguous replacement span,
- this leakage may cap performance before the bridge baselines are reached.

### H4. Output-level scoring should dominate hidden-only scoring
Future candidate selection must be ranked by output metrics first:
- teacher-logit KL,
- NLL / CE,
- PPL,
and only secondarily by hidden-space metrics.

## 3. Research strategy summary

The plan intentionally proceeds in increasing complexity:

1. **Idea 1 + Idea 3 first**  
   Re-test the current direction with better candidate search and better functional matching.  
   Expected outcome: some improvement, but likely still insufficient because hard discrete substitution remains a bottleneck.

2. **Idea 4 next**  
   Use the evidence from Idea 1 + 3 to relax the hard one-window choice into a soft mixture over a few candidate windows.

3. **Idea 5 after that**  
   Move from hand-picked windows to monotone / DTW-like alignment over layers or windows.

4. **Idea 2 last in this track**  
   Test whether the real correspondence is sublayer-specific (attention-only, MLP-only, or mixed).

Alternative paradigms (sidecar expert, small-conditioned bridge, activation-statistics-corrected hybrid, multi-anchor delegation, token-adaptive delegation) are explicitly deferred to follow-up research.

## 4. Immediate experimental phases

## Phase 1 — Idea 1 + Idea 3

### Phase 1A. Idea 1: Output-aware asymmetric window search

#### Goal
Replace the fixed 6-large <-> 6-small assumption with a constrained asymmetric search.

#### Rationale
If the true mapping is not one-to-one, a narrow search over:
- different large window lengths,
- different small window lengths,
- nearby relative-depth positions,
may produce a better delegated replacement than the frozen `24..29` -> `14..19` split.

#### Search space
Use a narrow, compute-respectful search around the current relative-depth prior.

Recommended first-pass search:
- large window lengths: `4, 6, 8`
- small window lengths: `2, 3, 4, 5, 6`
- large start positions near the current middle region
- small start positions near the current middle region
- preserve order and contiguity within each model

#### Ranking criteria
Primary:
- output KL to full large teacher
- NLL / CE
- PPL

Secondary:
- hidden MSE
- hidden cosine

#### Procedure
1. cheap 1-seed pilot screen,
2. shortlist top candidate windows,
3. 3-seed confirmation on the best few candidates.

#### Expected result
Likely modest but real gains over the `v0.5.1` frozen split, but not necessarily enough to beat strong bridge baselines.

### Phase 1B. Idea 3: Functional-stage matching

#### Goal
Stop treating layer indices as the main notion of correspondence.

#### Rationale
Two layers at similar relative depth may still sit in different functional stages.  
We need a stage signature for each layer or window.

#### Candidate stage signatures
Per layer / window:
- logit-lens entropy,
- KL to final logits,
- activation norm statistics,
- change in hidden cosine to neighboring layers,
- optional token-level entropy drift.

#### Usage
Use these signals to:
- characterize large and small model stages,
- identify which small windows are functionally closest to the removed large window,
- constrain or prioritize the Idea 1 search.

#### Expected result
Better interpretability and slightly better candidate choice, but probably still bounded by discrete replacement leakage.

### Phase 1 decision rule
If Idea 1 + 3:
- improve over the frozen `v0.5.1` hybrid split, keep the best candidates and proceed to Phase 2.
- do not improve meaningfully, still proceed to Phase 2, but treat that as evidence that hard one-window replacement is the main issue.

## Phase 2 — Idea 4: Soft mixture over candidate windows

### Goal
Remove the forced single-window choice.

### Rationale
The likely problem is not only "wrong window chosen" but "no single window is sufficient."

### Method
Use a small set of candidate small windows, selected from Phase 1, and learn a lightweight gate / mixture over them.

Possible forms:
- scalar gate per candidate,
- token-shared softmax over candidate windows,
- low-compute mixture with 2-3 candidate windows.

### Constraints
- single GPU,
- same-family only,
- frozen backbones,
- no broad architecture rewrite.

### Success criterion
The mixed-window hybrid should:
- preserve the `hybrid > hybrid_no_small` output-level result,
- materially shrink the gap to at least one strong bridge baseline.

### Failure interpretation
If it still loses clearly to strong bridges, the problem is probably deeper than window choice alone.

## Phase 3 — Idea 5: Monotone alignment / DTW-like matching

### Goal
Move beyond discrete hand-chosen replacement windows and learn an order-preserving asymmetric alignment.

### Rationale
If:
- 3 large layers correspond to 2 small layers,
- or one large computation is distributed across several small segments,
then a monotone alignment is a better abstraction than a single window swap.

### Method options
- dynamic-programming alignment over window costs,
- differentiable monotone alignment,
- piecewise monotone map from large-depth to small-depth.

### Scoring
Still output-aware first, hidden-aware second.

### Deliverable
A learned or inferred correspondence table such as:
- large range -> small range(s)
- confidence / cost per mapping

### Success criterion
A monotone alignment should outperform the best Phase 1 fixed-window candidate, or else it is not earning its complexity cost.

## Phase 4 — Idea 2: Sublayer-level substitution

### Goal
Test whether the mismatch is block-internal rather than whole-block structural.

### Rationale
A full transformer block may be the wrong unit of exchange.
Possible realities:
- attention transfers better than MLP,
- MLP transfers better than attention,
- hybrid large/small sublayer compositions are better than full-block substitution.

### Candidate variants
- attention-only replacement,
- MLP-only replacement,
- small attention + large-space MLP bridge,
- large-space attention bridge + small MLP.

### When to run
After window / alignment work, not before.  
Otherwise we change too many variables at once.

### Success criterion
A sublayer variant should either:
- reduce the bridge gap,
- or reveal which internal function is actually transferable.

## 5. Expected storyline and what counts as progress

### Expected near-term outcome
A likely outcome is:
- Idea 1 + 3 improve the current hybrid somewhat,
- but still fail to beat strong bridge baselines cleanly,
- which supports the discrete-replacement bottleneck diagnosis.

That is still useful, because it motivates Idea 4 and Idea 5 with concrete evidence rather than guesswork.

### What would count as a strong positive result
Any one of the following would be meaningful:
- a best fixed asymmetric window beats a strong bridge on output metrics,
- a soft-mixture window hybrid beats at least one strong bridge reproducibly,
- a monotone alignment materially reduces the bridge gap and preserves `hybrid > hybrid_no_small`.

### What would count as a useful negative result
Any one of the following:
- better candidate search does little,
- stage signatures improve interpretability but not output metrics,
- soft mixtures still fail to beat strong bridges,
- monotone alignment is too complex for too little gain.

That would strengthen the conclusion that the next step should be a different paradigm rather than more block matching.

## 6. Metrics and evaluation policy

### Primary metrics
- KL to full large teacher
- NLL / CE
- PPL

### Secondary metrics
- hidden MSE
- hidden cosine
- gate usage statistics
- delta norm statistics

### Baseline comparison policy
Every new candidate must be compared against:
- `skip_only`
- `hybrid_no_small`
- `bridge_only`
- `bridge_only_param_matched`
- current best `hybrid` from `v0.5.1`

### Seed policy
- 1-seed pilot screen for broad search
- 3-seed confirmation before any claim

### Compute policy
- single RTX 5090-class GPU only
- preserve frozen backbone policy unless explicitly changed in a later track
- do not start Stage C during this roadmap unless a later explicit decision justifies it

## 7. Repository and artifact policy

### Freeze rule
Do not overwrite `v0.5.1` artifacts or reports.

### New namespace suggestion
Use a new research namespace, for example:
- `configs/v0_6/`
- `scripts/v0_6/`
- `artifacts/v0_6/`
- `notes/v0_6/`

### Reporting rule
Each phase should produce:
- one summary report,
- one machine-readable result artifact,
- one short decision note stating whether to proceed.

## 8. Deferred follow-up paradigms (not for the current track)

These are worth keeping on the roadmap, but **not** in the immediate next sequence.

### A. Sidecar expert
Keep the large path intact and use the small model only as a parallel delta proposer.

### B. Small-conditioned bridge
Use the small path to control or condition a large-space bridge rather than replace the large block directly.

### C. Activation-statistics-corrected hybrid
Add explicit mean/variance or norm calibration on the returned large-space delta.

### D. Multi-anchor delegation
Use several small intervention points instead of one contiguous replacement span.

### E. Token-adaptive delegation
Only invoke the delegated path when token-level uncertainty justifies it.

These are not rejected; they are deferred because the current track first needs to settle whether better correspondence search is enough.

## 9. Practical next action

The immediate next action is:

1. freeze `v0.5.1` cleanly,
2. create the new `v0.6` research namespace inside the same repo,
3. implement Phase 1A and Phase 1B,
4. run a narrow output-aware asymmetric search plus stage-signature analysis,
5. use those results to decide the candidate set for Phase 2.

## 10. One-sentence thesis of the new track

> The next research track tests whether the current failure against strong bridge baselines is mainly caused by coarse, discrete, misaligned block substitution; if so, better asymmetric and stage-aware correspondence search should help, and if not, the evidence will justify moving to softer or entirely different delegation paradigms.
