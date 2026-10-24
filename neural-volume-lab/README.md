# Neural Volume Lab

An experimental native Qt desktop app for watching and teaching a small recurrent network. This does not demonstrate general reasoning or useful open-ended conversation.

Double-click `Launch Neural Volume Lab.cmd` after building locally. Source launch: `.venv/Scripts/python.exe neural-volume-lab/app.py`. Build: `.venv/Scripts/python.exe neural-volume-lab/build_desktop.py`. The generated launcher selects a versioned executable, preserving any already open application.

## Talking and teaching

1. Click **Start brain**. Adjust seconds per tick (default five).
2. Send a short message. One input token enters on each tick; the interface shows receiving progress.
3. After the last token, one stable reply appears. Idle activity does not rewrite it. An initially untrained network may produce an empty reply.
4. Use **Good reply** or **Needs work** to reward the exact sampled reply. Human impact defaults to five and ranges from one to ten.
5. Double-click a reply to pause and correct that particular message. Teaching performs twelve bounded supervised updates. The original reply stays visible.
6. **Try again** samples another message from the current state without advancing a hidden tick. Reward or correct it to teach; the network does not treat its own guesses as correct labels.
7. Optional **Professor coach** uses the existing frozen local model to review each delivered/practice reply and offer short examples. Its judgments are unverified. Retained human corrections take priority; professor examples receive five updates and lower reward weight.

## Actual architecture

Frozen SmolLM2 token embeddings have **576 components**, not 512. A learned 576-to-512 adapter feeds a learned dense 512-to-15,625 input projection. Every recurrent neuron connects to all 512 output units through another learned dense layer. The recurrent core has 12 directed inputs per neuron, excluding self and directly reciprocal edges. There are 20,947,564 trainable parameters. Neuron positions only control drawing; embedding coordinates no longer choose neurons.

The output readout predicts up to sixteen token slots plus an end marker from an admitted vocabulary of at most 511 target tokens. This supplied, bounded readout is not an unrestricted language decoder. Layer normalization and the 576-to-512 bottleneck still discard information. Inspect network retains the coordinate-polyline viewer and manual experiment controls. The volume shows actual input, core and output activity.

Scalar feedback is immediate contextual-bandit policy optimization with a clipped probability-ratio surrogate; it is not full PPO or long-horizon reinforcement learning. Corrections and teacher examples are supervised optimization. Frozen embeddings and professor language ability come from pretraining. The network has not learned those capabilities independently.

## Persistence and privacy

The app automatically saves weights, optimizer, sampler state, vocabulary and up to eight numeric training contexts in ignored `runs/neural-volume-lab/learning/dense-io-v1/brain.pt`. No raw transcript is saved, but token IDs and numeric replay remain private training data. The brain restores with silent activity; chat bubbles and transient reply credit are not restored. The primary trained checkpoint is approximately 241 MiB, with one previous backup and temporary atomic writes. Only one learning window can own that brain. Restore mismatches preserve files and fail explicitly; save failures pause the UI and warn.

All models, checkpoints, package builds and caches remain on the project drive. No model download or paid API is required with the existing cache. The GUI package reuses this checkout's Python worker; it is launchable but not a portable standalone distribution. Verification modes use isolated synthetic checkpoints.

## Evidence and limits

The [learned-I/O diagnostic](../reports/neural-volume-io-v1/README.md) verifies fitting, restoration and removal of one coordinate alias. All four controls fit both known pairs, including fixed features with only the readout trained. This does not establish a benefit from learned recurrence or semantic generalization. Next research should measure retained learning, interference and unfamiliar inputs under matched controls before increasing model complexity.

[Historical spatial design and previous launch notes](HISTORY.md) preserve the earlier experiment; its results are not relabeled as results of this architecture.
