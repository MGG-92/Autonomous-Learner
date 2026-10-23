"""Stable session identities. Digests detect mismatch; they are not authentication."""
import hashlib
import json

import torch

from .policy import Heuristic


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def controller_identity(controller) -> dict:
    if isinstance(controller, str):
        return {"builtin": controller}
    name = f"{type(controller).__module__}.{type(controller).__qualname__}"
    if isinstance(controller, torch.nn.Module):
        hasher = hashlib.sha256()
        for key, value in sorted(controller.state_dict().items()):
            value = value.detach().cpu().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError("Controller contains nonfinite parameters")
            hasher.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode("utf-8"))
            hasher.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
        return {"type": name, "parameters_sha256": hasher.hexdigest()}
    if type(controller) is Heuristic:
        return {"type": name}
    raise ValueError("Persistent sessions require a supported deterministic controller")
