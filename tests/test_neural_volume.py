"""Geometry, encoding and offline asset boundaries; no model download required by CI."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch


LAB = Path(__file__).resolve().parents[1] / "neural-volume-lab"
spec = importlib.util.spec_from_file_location("volume_embeddings", LAB / "embeddings.py")
embeddings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embeddings)


def test_triples_keep_coordinates_order_scale_and_all_dimensions():
    assert embeddings.coordinate_triples([-1, -2, 0, 6, 1, -3], 1) == [[-1, -2, 0], [6, 1, -3]]
    assert embeddings.coordinate_triples([2, -4, 6, 8], 2) == [[1, -2, 3], [4, 0, 0]]
    with pytest.raises(ValueError):
        embeddings.coordinate_triples([float("nan")], 1)
    with pytest.raises(ValueError):
        embeddings.coordinate_triples([1], 0)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(word) for word in text.split()]

    def convert_ids_to_tokens(self, token_id):
        return f"piece-{token_id}"

    def decode(self, ids, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        return str(ids[0])


def store():
    result = embeddings.EmbeddingStore()
    result.weights = torch.tensor([[1., 2., -3., 4.], [2., 4., -6., 8.]])
    result.tokenizer, result.scale = Tokenizer(), 8.
    return result


def test_static_vectors_and_fixed_scale_do_not_depend_on_neighboring_tokens():
    encoder = store()
    alone, sequence = encoder.encode("0"), encoder.encode("1 0 0")
    assert sequence["tokens"][1] == sequence["tokens"][2] == alone["tokens"][0]
    assert sequence["tokens"][0]["points"][0] == [0.25, 0.5, -0.75]
    assert alone["tokens"][0]["points"][0] == [0.125, 0.25, -0.375]
    assert alone["tokens"][0]["padded_components"] == 2
    assert encoder.metadata()["teacher"] is False
    assert encoder.metadata()["neural_simulation"] is False


@pytest.mark.parametrize("text", [" ", None, "x" * 4097, "0 " * 65])
def test_empty_or_oversized_input_is_rejected_not_silently_truncated(text):
    with pytest.raises(ValueError):
        store().encode(text)


def test_missing_assets_fail_explicitly_without_network_or_synthetic_fallback(tmp_path):
    encoder = embeddings.EmbeddingStore(tmp_path)
    with pytest.raises(RuntimeError, match="Local embedding model unavailable"):
        encoder.metadata()
    assert encoder.weights is None


def test_corrupt_local_assets_are_refused_before_loading(tmp_path):
    manifest = {"model": embeddings.MODEL_ID, "revision": embeddings.REVISION,
                "sha256": {"embeddings.npy": "invalid", "tokenizer.json": "invalid"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "embeddings.npy").write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="Local embedding model unavailable"):
        embeddings.EmbeddingStore(tmp_path).load()
