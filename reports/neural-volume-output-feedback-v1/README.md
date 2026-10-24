# Output-to-volume feedback development smoke

CPU, seed 42, two threads, twelve online updates per known synthetic pair:
Hello world→Welcome, Goodbye world→Farewell. Activity perturbation seed 93.
The feedback-on and feedback-off variants share their existing initial weights and
update counts; feedback adds eight million parameters. This is not held-out testing.

Both conditions retain only **1/2 exact pairs** after teaching. This limited result
shows no task benefit, and exposes unresolved retention/interference. No extra
updates or checkpoint selection were used to improve the result.

In the feedback condition, 4,107,565 weights became nonzero: 1,891,111 positive and
2,216,454 negative. Temporarily zeroing feedback changed the same input's final
core state by a maximum 0.592931 and logits by 0.047389. This demonstrates an effect
on internal computation, not improved language ability. Connectivity has eight
million trainable entries even though some remain zero.

Separate regression tests set controlled positive/negative weights and reproduce
excitation/inhibition, zero feedback influence on the first driven tick, learning
through a two-token sequence, and decay after deliberately oversized weights are
projected to the permitted bounds. Gating avoids feeding full normalized output
back into a tiny residual state. Undriven activity peak is bounded by 0.915 times
the preceding peak; exact zero remains zero. This is a norm-to-zero bound, not a
pairwise contraction claim through layer normalization. It applies to nonnegative
core states under the supplied constraints, not arbitrary unconstrained weights.

The layer feeds previous continuous output activations back into the next core
update. It does not feed sampled words/professor text back or implement automatic
stopping when a reply is delivered. Successful response stopping and a benefit
for retained learning remain untested. All raw checkpoints remain private under
ignored runs; controls.json contains only sanitized synthetic measurements/settings.
