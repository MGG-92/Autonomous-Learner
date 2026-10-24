"""Export the pinned cached embedding table for an offline, Torch-free desktop runtime."""
import hashlib
import json
from pathlib import Path
import shutil


def prepare():
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from embeddings import MODEL_ID, REVISION

    root = Path(__file__).resolve().parent.parent
    source = Path(snapshot_download(MODEL_ID, revision=REVISION,
                                   cache_dir=str(root / "runs" / "models"), local_files_only=True))
    destination = root / "runs" / "neural-volume-lab" / "model"
    if destination.exists():
        raise FileExistsError("Prepared model exists; preserve it or choose an explicit new export")
    # Create only after validating the cached source. No network or model generation.
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as handle:
        table = handle.get_tensor("model.embed_tokens.weight").float().numpy()
    if table.shape != (49152, 576) or not np.isfinite(table).all():
        raise ValueError("Unexpected or nonfinite source table")
    destination.mkdir(parents=True)
    np.save(destination / "embeddings.npy", table, allow_pickle=False)
    shutil.copyfile(source / "tokenizer.json", destination / "tokenizer.json")
    digests = {}
    for name in ("embeddings.npy", "tokenizer.json"):
        with (destination / name).open("rb") as handle:
            digests[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"model": MODEL_ID, "revision": REVISION, "shape": list(table.shape),
                "scale": float(np.abs(table).max()), "sha256": digests,
                "kind": "static_input_embeddings", "conversion": "source bfloat16 to exact float32 values"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared local embedding assets in ignored runs/neural-volume-lab/model")


if __name__ == "__main__":
    prepare()
