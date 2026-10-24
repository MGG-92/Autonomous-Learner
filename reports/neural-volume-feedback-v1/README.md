# Immediate reply feedback: development diagnostic

This checks update mechanics, not dialogue accuracy or general reasoning. A full
25³ network with the existing 1,898,572 parameters samples a reply after receiving
synthetic input ID 1; target IDs 8 and 9 supply the same output vocabulary to each
condition. No supervised optimization precedes measurement. Each condition starts
from the same seed-specific initialization, so its sampled reply is matched.
Seeds are 7, 42 and 97; all outcomes are retained. No final test set or checkpoint
selection is involved. [controls.json](controls.json) records exact log probabilities,
actions including EOS, parameters, source digests and runtime versions.

| Condition | Mean change in sampled tokens' mean log probability |
|---|---:|
| Human +1, default impact 5 | +0.088467 |
| Human −1, default impact 5 | −0.094361 |
| Human +1, impact 1 | +0.018182 |
| Professor +1, impact 1 and reward weight 0.25 | +0.004568 |
| Neutral reward / untrained parameters retained | 0 |
| Human +1 after clearing replay | +0.088467 |
| Fixed uniform random readout, no optimizer | 0 by construction |

All three seeds changed in the intended direction. Clearing replay has the same
effect as retaining it because scalar feedback does not look up or fabricate a
target response. The fixed random control does not learn from ratings. These are
mechanism controls, not measures of task success, memory benefit or RL generalization.

The feedback surrogate clips per-token new/behavior probability ratios at [0.8, 1.2]
in its objective, uses gradient norm cap 1 and SGD lr 0.05 × impact. Human impact
defaults to 5; professor impact is 1 with reward weight 0.25. One update per source
per sampled reply is allowed. This is immediate-reward contextual-bandit learning,
not full PPO or long-horizon RL. Objective clipping is not a strict probability/KL
bound. A live chat snapshot credits one actual transition; earlier state is detached.

Separate real-model native verification checks a real five-second timer, sequential
Hello/world input, no vertex introduction before a tick, live-state output, double-click
editing with pause, a fitted Welcome correction, scalar reward and a real offline
professor vote. The professor's Yes/No support is not calibrated or validated for
correctness. Chat coaching is a hybrid: examples bootstrap new words, while known-word
responses can receive reward only. No teacher-quality or useful-conversation claim
is established. All interactions in the verification are explicit synthetic text.

Reproduce after bootstrap:

```powershell
.\.venv\Scripts\python.exe neural-volume-lab/check_feedback.py --output runs/neural-volume-lab/new-feedback-controls.json
```

Next compare reward-only, examples-only and hybrid teaching on independently labeled
fresh conversations, tracking retention/interference and teacher mistakes. The earlier
frozen-core result remains: simple two-pair fitting did not require learning recurrence.
