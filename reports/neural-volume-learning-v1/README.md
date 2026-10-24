# Spatial response learner: development smoke

This is a software and memorization check, not a reasoning/generalization benchmark.
Two public synthetic associations were taught sequentially: `Hello → Welcome`, then
`Goodbye → Farewell`. Neither is a copy task. The full 25³ network, real pinned
static embeddings and learned token readout were used. No professor generated these
labels or participated in numerical predictions.

Seed 42; CPU, two Torch threads; 12 incoming edges per unit, 96 readout hidden units,
16 output slots plus EOS, 128 readout classes, eight replay pairs. Adam lr 0.003,
gradient clip 1, 100 epochs after each admission. Cross entropy is scored through
first EOS. Row absolute recurrent weight sums are constrained to 0.7; leak 0.15,
ReLU drive factor 0.85, activity cap 4. Each token has a driven and undriven tick.
[controls.json](controls.json) records exact outputs, losses, versions and source
digests. Embedding metadata describes the static source, not the later runtime.
Each condition receives the same admitted target vocabulary as readout support.

| Condition | Exact taught responses after both admissions |
|---|---:|
| Learned recurrent core and readout, replay | 2/2 |
| Frozen random recurrent core, trained readout, replay | 2/2 |
| No replay of the earlier pair | 1/2 |
| Zero input during training and prediction | 1/2 |
| Untrained network with admitted output vocabulary | 0/2 |
| Seeded uniform random token/EOS readout | 0/2 |

No replay means clearing stored examples before the next admission, while retaining
trained weights. Zero-input retains identical output slots and optimizer access.
The frozen-core control changes no recurrent parameters; its success means this
task does not require learning recurrent dynamics. Nonzero recurrent changes in
the full learner establish backpropagation, not functional necessity or reasoning.
Two pairs give no population estimate. No significance, transfer, calibration or
teacher-quality claim is made. There is no held-out set or checkpoint selection.

Separate native source/package verification uses actual offline professor generation
and widgets, then overrides its proposal with the public targets above. It checks
vertex stimulation, silence preserving parameters, exact-zero invariance, changed
targets excluded from prediction, and perturbed-state rest with task replay.
Initial rest loss was 0.01847858 and final sampled loss 0.00783895 after ten updates;
Hello still predicted Welcome. These losses use different seeded perturbations:
this is not a paired rest-generalization estimate. Contraction already ensures decay.
Private interactions, weights, images and raw process output are not published.

Reproduce with prepared offline assets and a fresh destination:

```powershell
.\.venv\Scripts\python.exe neural-volume-lab/check_learning.py --output runs/neural-volume-lab/new-controls.json
```

Next: fresh sequence/order and spatial-collision tasks with frozen-core and
readout-only controls before attributing a benefit to recurrent learning.
