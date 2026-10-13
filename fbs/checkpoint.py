"""Architecture-aware checkpoints; no pickled model instances."""
from pathlib import Path
import torch
from .model import FBSModel


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "config": model.config, "model_state": model.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in data:
        raise ValueError("Legacy checkpoint lacks architecture config; reconstruct explicitly before migrating")
    config = dict(data["config"])
    if config.pop("kind", None) == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from .hf import QwenFBSModel
        backbone = Qwen3ForCausalLM(Qwen3Config(**config.pop("backbone_config")))
        model = QwenFBSModel(backbone, **config)
    else:
        model = FBSModel(**config)
    model.to(dtype=data["model_state"]["token_emb.weight"].dtype)
    model.load_state_dict(data["model_state"], strict=True)
    return model.to(device), data
