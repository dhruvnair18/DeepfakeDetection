# backend/inference/loader.py
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v0_1")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", f"backend/models_store/{DEFAULT_MODEL_VERSION}/mobilenet_v2.pt")
NUM_CLASSES = 2  # real/fake

class ModelLoader:
    _instance = None

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else "cpu")
        self.version = DEFAULT_MODEL_VERSION
        self.model = mobilenet_v2(weights=None)  # will set classifier next
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

        if os.path.exists(DEFAULT_MODEL_PATH):
            state = torch.load(DEFAULT_MODEL_PATH, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"[ModelLoader] Loaded weights from {DEFAULT_MODEL_PATH}")
        else:
            print("[ModelLoader] WARNING: model weights not found. Using randomly initialized head.")

        self.model.eval().to(self.device)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def predict_logits(self, batch_tensor):
        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            logits = self.model(batch_tensor)
        return logits.detach().cpu()
