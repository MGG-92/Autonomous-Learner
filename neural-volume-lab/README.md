# Neural Volume Lab

Work in progress: a new experiment in visualizing token embeddings and training
a spatial neural network from user-provided input/response examples.

The first stage is now a native desktop embedding viewer built with PySide6
QtWidgets and a QPainter drawing surface. No browser, web server or WebView is used.
The neural network, training and teacher are still later stages. Existing research
experiments and their frozen results remain separate.

## Launch the desktop app

On this Windows checkout, double-click `Launch Neural Volume Lab.cmd`. It uses the
packaged `NeuralVolumeLab.exe` when present, otherwise the project's Python runtime.
The packaged application lives under ignored
`runs/neural-volume-lab/desktop/NeuralVolumeLab/`. Keep that entire directory together;
the executable uses the neighboring runtime/assets. No Python installation is needed
when launching that packaged build.

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

The app itself never downloads a model, calls a service, generates language or saves
messages. It reads checksummed, read-only local arrays and tokenizer assets. The
normal desktop runtime needs NumPy, tokenizers and PySide6 Essentials; it does not
load Torch or Transformers. Those libraries are needed only for export from the
original cached bfloat16 model. Missing/corrupt assets produce an explicit error.

## What the first stage does

- Accept a message and optional expected-response reference, both visualized from
  actual static token vectors. The reference is view-only, not a prediction or target
  used by any training routine yet.
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
& .\runs\neural-volume-lab\desktop\NeuralVolumeLab\NeuralVolumeLab.exe --verify-output runs/neural-volume-lab/package-check
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
