"""Private, bounded, atomic brain checkpoints. Never load executable pickle objects."""
from dataclasses import asdict
import math
import os
from pathlib import Path
import shutil

import torch


class BrainStore:
    SCHEMA = 1

    def __init__(self, path, identity):
        self.path = Path(path).resolve()
        self.identity = identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous = self.path.with_suffix(".previous.pt")
        self.lock = self.path.with_suffix(".lock").open("a+b")
        self.lock.seek(0, 2)
        if self.lock.tell() == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock.close()
            raise RuntimeError("Another app owns this saved brain. Keep one learning window running at a time.") from error
        self.restored = False
        self.recovered = False

    @staticmethod
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(BrainStore.finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(BrainStore.finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)

    def validate(self, data, learner):
        if data.get("schema") != self.SCHEMA or data.get("config") != asdict(learner.config):
            raise ValueError("Saved brain architecture does not match; original checkpoint preserved")
        if data.get("embedding_identity") != self.identity:
            raise ValueError("Saved brain embedding source does not match; original checkpoint preserved")
        expected = learner.network.state_dict()
        state = data["network"]
        if set(state) != set(expected) or any(state[k].shape != v.shape or state[k].dtype != v.dtype for k, v in expected.items()):
            raise ValueError("Saved brain parameter shapes do not match")
        if not torch.equal(state["sources"], expected["sources"]):
            raise ValueError("Saved brain topology does not match")
        vocab = data["vocabulary"]
        vocabulary_size = self.identity.get("vocabulary_size", 49152)
        if len(vocab) >= learner.config.vocabulary or len(set(vocab)) != len(vocab) or any(type(i) is not int or not 0 <= i < vocabulary_size for i in vocab):
            raise ValueError("Invalid saved output vocabulary")
        examples, metadata = data["examples"], data["example_metadata"]
        if len(examples) > learner.config.replay or len(metadata) != len(examples):
            raise ValueError("Invalid saved replay size")
        for (inputs, labels), meta in zip(examples, metadata):
            if labels.shape != (learner.config.slots + 1,) or labels.dtype != torch.long or int(labels.min()) < 0 or int(labels.max()) > len(vocab):
                raise ValueError("Invalid saved target labels")
            if isinstance(inputs, dict):
                if "readout_state" in inputs:
                    valid = set(inputs) <= {"readout_state", "sequence"} and inputs["readout_state"].shape == learner.network.activity.shape
                    if valid and "sequence" in inputs:
                        valid = (inputs["sequence"].ndim == 2 and 1 <= inputs["sequence"].shape[0] <= learner.config.slots and
                            inputs["sequence"].shape[1] == (learner.config.input_dimensions if learner.config.encoding == "learned" else learner.config.side ** 3))
                else:
                    valid = set(inputs) == {"state_before", "pulse"} and inputs["state_before"].shape == learner.network.activity.shape
                    if valid and inputs["pulse"] is not None:
                        valid = inputs["pulse"].shape == (learner.config.input_dimensions if learner.config.encoding == "learned" else learner.config.side ** 3,)
            else:
                valid = inputs.ndim == 2 and 1 <= inputs.shape[0] <= learner.config.slots and inputs.shape[1] == (learner.config.input_dimensions if learner.config.encoding == "learned" else learner.config.side ** 3)
            if not valid or (meta is not None and (len(meta) != 2 or meta[1] not in ("user", "human", "professor"))):
                raise ValueError("Invalid saved training context")
        if type(data["updates"]) is not int or data["updates"] < 0 or not self.finite(data):
            raise ValueError("Invalid or nonfinite saved learning state")

    def restore(self, learner):
        failures = []
        for candidate in (self.path, self.previous):
            if not candidate.exists():
                continue
            try:
                if candidate.stat().st_size > 512 * 1024 ** 2:
                    raise ValueError("Saved brain exceeds the local size bound")
                data = torch.load(candidate, map_location="cpu", weights_only=True)
                self.validate(data, learner)
                learner.network.load_state_dict(data["network"])
                learner.optimizer.load_state_dict(data["optimizer"])
                learner.feedback_optimizer.load_state_dict(data["feedback_optimizer"])
                learner.vocabulary = data["vocabulary"]
                learner.examples = data["examples"]
                learner.example_metadata = data["example_metadata"]
                learner.updates = data["updates"]
                learner.sampling.set_state(data["sampling_state"])
                learner.network.activity.zero_()
                self.restored = True
                self.recovered = candidate == self.previous
                return True
            except Exception as error:
                failures.append(f"{candidate.name}: {type(error).__name__}: {error}")
        if failures:
            raise RuntimeError("Saved brain could not be restored; files preserved. " + "; ".join(failures))
        return False

    @staticmethod
    def copy_atomic(source, destination):
        pending = destination.with_suffix(destination.suffix + ".pending")
        shutil.copyfile(source, pending)
        os.replace(pending, destination)

    def save(self, learner, reset=False):
        data = {"schema": self.SCHEMA, "config": asdict(learner.config), "embedding_identity": self.identity,
            "network": learner.network.state_dict(), "optimizer": learner.optimizer.state_dict(),
            "feedback_optimizer": learner.feedback_optimizer.state_dict(), "vocabulary": learner.vocabulary,
            "examples": learner.examples, "example_metadata": learner.example_metadata,
            "updates": learner.updates, "sampling_state": learner.sampling.get_state()}
        self.validate(data, learner)
        pending = self.path.with_suffix(".pending.pt")
        with pending.open("wb") as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if self.path.exists() and not reset and not self.recovered:
            self.copy_atomic(self.path, self.previous)
        os.replace(pending, self.path)
        if reset or self.recovered:
            self.copy_atomic(self.path, self.previous)
            self.recovered = False

    def close(self):
        self.lock.close()
