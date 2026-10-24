# Neural Volume Lab

Work in progress: a new experiment in visualizing token embeddings and training
a spatial neural network from user-provided input/response examples.

The first stage is now a native desktop embedding viewer built with PySide6
QtWidgets and a QPainter drawing surface. No browser, web server or WebView is used.
It now includes a sparse recurrent learner and an offline local professor. Existing
research experiments and their frozen results remain separate. This is a supervised
association-fitting experiment, not a demonstration of independent general reasoning.

## Launch the desktop app

On this Windows checkout, double-click `Launch Neural Volume Lab.cmd`. It uses the
packaged `NeuralVolumeLab.exe` when present, otherwise the project's Python runtime.
The packaged application lives under ignored
`runs/neural-volume-lab/desktop-learning/NeuralVolumeLab/`. Keep that entire directory
together. The GUI bundles its Qt/embedding assets; neural/professor computation reuses
this checkout's `.venv` and cached model. It is a local deployment, not a portable
standalone compute distribution. No duplicate Torch or language-model bundle is built.

For source development, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[desktop,language]'
.\.venv\Scripts\python.exe neural-volume-lab/prepare_model.py
.\.venv\Scripts\python.exe neural-volume-lab/app.py
```

Preparation exports only the pinned embedding table and tokenizer from the existing
local model cache. It refuses to overwrite a prepared export. If the model is not
yet cached, explicitly download it once before preparation:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct', revision='12fd25f77366fa6b3b4b768ec3050bf629380bac', cache_dir='runs/models')"
```

The app never downloads a model or calls a service. The native renderer uses
checksummed arrays, NumPy, tokenizers and Qt. An on-demand local subprocess runs
CPU Torch and, only when requested, Transformers. Missing assets fail explicitly.
All messages, proposals, replay examples and learned weights are RAM-only: closing
the app loses learning. No private conversations or checkpoints are published.

## Teach and inspect the network

1. Click **Start network**. The 25³ grid starts silent with random trainable weights.
2. Enter a message, then **Visualize** and select a token. **Inject token** stimulates
   its vertices' nearest grid sites; **Step** advances one synchronous neural tick.
3. **Run** advances without further input; the firing-rate slider controls display
   tick pacing. It adds no artificial noise. **Silence** zeroes activity and pauses
   without erasing weights. Zero activity stays zero in this deterministic system.
4. Write an expected response, or **Ask professor** for an editable, unverified
   proposal. The frozen local SmolLM2-135M generates at most 16 tokens, greedily on CPU.
   It is a small, fallible model; output can be wrong or truncated. It never trains
   the learner automatically. The first professor request also loads the cached model.
5. **Train pair** performs 100 Adam updates in ten cancellable chunks. **Stop training**
   finishes the current chunk and stops. The last eight training pairs are replayed.
   A repeated click admits another replay entry; frequencies therefore affect training.
6. **Predict** runs a fresh recurrent pass using only the message, then its learned
   token readout. Network output appears separately from the expected response.
   Try `Hello → Welcome`, then `Goodbye → Farewell`, and retest the first message.
7. **Train rest** optimizes ten perturbed-state rest steps alongside task replay.
   Existing contraction already guarantees decay; this is an optimization diagnostic,
   not evidence of spontaneous activity. **Reset learning** erases the session after
   a confirmation and is distinct from Silence. **Clear space** only clears the view.

Network inputs and targets are limited to 16 tokens. The learned readout supports
127 distinct admitted target tokens plus EOS, with 16 ordered output slots. It cannot
invent vocabulary beyond these targets or generate arbitrary conversation. It may
return incorrect output for unseen inputs; there is no learned abstention guarantee.
Visualization alone still supports 64 tokens. A response reference is visualization
only until **Train pair** is explicitly invoked.

### What is learned and what is supplied

The grid has 15,625 units and 187,500 trainable directed edges (12 inputs per unit),
plus a learned 96-unit readout projection and fixed-slot token classifier: 1,898,572
parameters in total. Initialization seed is 42, recurrent biases are absent, initial
activity is zero. Local flat-grid offsets and seeded long offsets define a supplied
cyclic topology; boundary wrapping is an artificial convention. No self or reciprocal
edges exist; longer cycles remain possible. This is not a biological brain simulator.

For state `s` and vertex pulse `p`, one synchronous tick is
`s_next = clamp(0.15*s + 0.85*ReLU(sum(w*s_source) + p), max=4)`.
Each token supplies a binary nearest-site pulse, followed by one undriven tick.
Tokens enter in order; prediction starts from zero independently of live exploration.
Rounded indices use `(coordinate+1)*24/2` on each axis. Colliding vertices merge;
within-token vertex order and magnitudes inside a quantization cell are lost.
Segments do not stimulate. Display coordinates use the original fixed scale.

Backpropagation learns edge and readout parameters from token cross entropy through
the first EOS, with Adam lr 0.003 and gradient-norm clipping at 1. Row absolute weight
sums are constrained to 0.7 after each update, making undriven dynamics contractive.
The readout has its own learned biases. Targets are token labels; their embedding
polylines are inspectable references, **not** clamped recurrent predictions or an
implemented spatial reconstruction loss. This initial text readout replaces the
ambiguous reverse mapping from a neuron pattern to an ordered sentence.

Geometry, topology, recurrence, replay admission, optimizer, slots and decoder support
are programmed. The embeddings and professor contain pretrained information.
Learned weights fit examples; memory is the eight-pair replay buffer. Fitting is not
rule discovery. Our two-pair control also succeeds with a frozen recurrent core, so
it does not establish a benefit from learning recurrent dynamics. See the sanitized
development report in `reports/neural-volume-learning-v1/`.

### Disk usage and verification

This deployment reuses the existing model under `runs/models`. No model or package
download was needed. Child-process temporary/model caches, PyInstaller cache/work,
exports, binaries and captures are redirected into ignored `runs/` on the checkout's
drive. The GUI does not duplicate Torch/Transformers into its package. Operating-system
and unrelated application writes are outside this application's disk control.

```powershell
.\.venv\Scripts\python.exe neural-volume-lab/check_learning.py --output runs/neural-volume-lab/new-controls.json
.\.venv\Scripts\python.exe neural-volume-lab/app.py --verify-learning-output runs/neural-volume-lab/new-source-check
& .\runs\neural-volume-lab\desktop-learning\NeuralVolumeLab\NeuralVolumeLab.exe --verify-learning-output runs/neural-volume-lab/new-package-check
```

Verification uses only explicit synthetic text, checks real native controls and
worker/model execution, and saves local screenshots. Ordinary interaction is not
logged. Missing compute/model dependencies report errors; there is no scripted fallback.

## What the first stage does

- Accept a message and optional expected-response reference, both visualized from
  actual static token vectors. References and free predictions are distinguished.
- Display each selected token's complete polyline instantly. Play/pause, previous/next,
  token chips, playback pace and all-token overlay are available.
- Orbit with drag or arrow keys, zoom with the wheel or +/−, and reset the camera.
- Inspect token IDs, raw tokenizer pieces, vector dimensions and raw XYZ components.
- Use the frozen SmolLM2-135M-Instruct input embedding table: 49152 rows, 576 components,
  192 vertices and 191 segments per token. No contextual model forward pass occurs.
- Divide every component by the same table-wide maximum absolute value, 1.9609375.
  Do not normalize tokens separately. Camera fitting is shared across both submitted
  sequences and does not change their coordinates. Zero-pad incomplete triples;
  discard no dimensions. Coordinate values are not interpreted as concept axes.
- Respect a 64-token / 4096-character limit per field, rejecting rather than silently
  truncating input. Text that looks like markup is rendered as plain text.

The pretrained table carries pretrained information. This is an embedding visualization
experiment, not evidence of independent reasoning, neural firing or learned behavior.
The guides describe the current coordinate view; they are not a neuron lattice.

## Package and verify

```powershell
.\.venv\Scripts\python.exe -m pip install 'pyinstaller>=6.11,<7'
.\.venv\Scripts\python.exe neural-volume-lab/build_desktop.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe neural-volume-lab/app.py --verify-output runs/neural-volume-lab/source-check
& .\runs\neural-volume-lab\desktop-learning\NeuralVolumeLab\NeuralVolumeLab.exe --verify-output runs/neural-volume-lab/package-check
```

Explicit verification mode uses the real local model and native controls, tests
geometry and playback/selection/reference/reset behavior, captures a native window
image and writes a compact result to the requested local directory. It then exits.
It does not train or publish a model. The ordinary launch opens an empty volume.
Builds, model assets, test captures and verification output stay in ignored `runs/`.
Windows packaging is currently local; CI tests geometry and native widget behavior
without downloading model weights. The Windows build is not a signed installer.


## Proposed experience

- A minimal interface centered on a dark, navigable three-dimensional volume.
- Text fields for an input message and its expected response.
- Tokenize messages and obtain one embedding vector per token.
- Group successive embedding components into XYZ coordinates and connect the
  resulting points with straight segments. Display each token's complete polyline
  at once, advancing through the tokens in order.
- Place a 25-by-25-by-25 grid of neurons in the same volume: 15,625 neurons.
- Map embedding coordinates and neuron positions onto a shared, documented scale.
  Embedding vertices supply spatial stimulation; the drawn connecting segments
  are visual aids unless a later explicit design also gives them an input role.
- Evolve neural activity in discrete steps and visualize actual activations.
- Use an expected response's encoded pattern as a supervised training target.
- Begin by testing whether individual message/response associations can be fitted.
- Provide pause, single-step, activity-speed and silence/reset controls, plus an
  explicit training action intended to suppress unwanted residual activity.
- Consider a local teacher model and self-sustained activity in later experiments.

## Original network proposal

Start with dense connectivity, ReLU activations, zero weights and zero biases.
Avoid immediate feedback into the neurons responsible for an activation; one
candidate is directing each pair's connection only one way, with approximately
half the other neurons supplying inputs and half receiving outputs.

These are proposed experimental choices, not validated architecture decisions.

## Questions to resolve before implementation

1. **Trainability.** An ordinary all-zero linear/ReLU layer has zero gradients
   under the tested PyTorch backpropagation path, even with a nonzero target.
   A local four-unit check produced loss 1.0 and zero weight/bias gradients.
   Distinguish initially silent activity from zero trainable parameters; evaluate
   an initialization or learning rule that can actually leave the silent state.
2. **Embedding meaning.** Coordinate triples provide a reproducible drawing of
   vector components, not a validated three-dimensional map of concepts. Choose
   static token embeddings or contextual representations explicitly. Tokens may
   be word fragments. Pretrained embeddings introduce pretrained information.
3. **Spatial encoding.** Define fixed scaling, out-of-range handling, vectors
   whose length is not divisible by three, and stimulation between grid points.
   Quantization and combining points can lose information. Preserve token order
   and determine how coordinate-group order is represented during training.
4. **Connectivity and timing.** No direct reciprocal connections still permits
   longer cycles. Specify whether the restriction means a fixed directed graph,
   temporary refractoriness or a permanently acyclic network. Discrete simulation
   steps and playback speed must be distinct from neural dynamics parameters.
5. **Response generation.** A target activation pattern alone does not define
   how to recover ordered output tokens or stop a response. Define a measurable
   pattern loss and a text readout; keep targets separate from free prediction.
6. **Resources.** Dense directed connectivity without self-connections gives
   244,125,000 weights: about 976.5 MB in float32. Weights, gradients and two Adam
   moment arrays alone total about 3.906 GB, before simulation history and other
   overhead. Decide how to retain the full visual grid while keeping training usable.
7. **Resting activity.** Resetting activations, erasing learned weights and training
   for stable rest are different operations. Deterministic ReLU dynamics with
   zero biases and no external drive stay exactly at zero when initialized at zero.
   Later spontaneous activity needs an explicit source or persistence mechanism.
8. **Learning evidence.** Fitting one example tests memorization. Adding a second
   can interfere with the first. Track recall of earlier examples and distinguish
   memorization, generalization and any later autonomous behavior.

Technical references: [PyTorch initialization](https://docs.pytorch.org/docs/stable/nn.init.html),
[PyTorch ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html),
and [Hugging Face tokenization](https://huggingface.co/docs/transformers/tokenizer_summary).

Keep model downloads, checkpoints, raw interactions and training artifacts out of
versioned source. Any future visualization must show recorded or current numerical
activity rather than an animation presented as measured cognition.
