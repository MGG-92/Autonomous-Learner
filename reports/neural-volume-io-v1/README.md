# Learned input/output development diagnostic

CPU, seed 42, two threads, twenty supervised updates after each of two known
synthetic pairs: Hello→Welcome and Goodbye→Farewell. See controls.json for source
hashes, runtime, exact settings, losses and predictions. These are development
examples, not held-out evaluation.

| Condition | Exact pairs after teaching |
| --- | ---: |
| Learned input/output and recurrent core | 2/2 |
| Frozen recurrent edges | 2/2 |
| Fixed features, trained readout only | 2/2 |
| Earlier spatial baseline | 2/2 |

There is no evidence here that learned recurrence or input features are necessary
or better. The spatial control uses its original 128-class budget; the new model
uses 512 classes, so this is a functional smoke comparison, not matched-capacity
architecture evaluation.

Saving/restoring the learned model preserved predictions, updates and vocabulary;
the private trained file measured 241.177 MiB. A constructed reversal of coordinate
groups produced an identical old spatial stimulus but distinct learned core states;
7,871 core units were active on the first dense-input tick. This is an encoding
counterexample, not semantic generalization. The 576→512 adapter remains lossy.

Only this sanitized synthetic report is published. Checkpoints, numeric replay,
GUI artifacts and teacher conversations remain ignored locally.
