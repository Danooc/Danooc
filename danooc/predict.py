"""
Inference utilities — load a trained model and classify a single image.
"""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from danooc.config import Config
from danooc.model import DanoocNet
from danooc.dataset import CIFAR10_MEAN, CIFAR10_STD


_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def predict_image(
    image_path: str,
    checkpoint: str = "best.pt",
    config: Config | None = None,
) -> Tuple[str, float]:
    """
    Classify a single image file.

    Returns ``(class_name, confidence%)``.
    """
    config = config or Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DanoocNet(num_classes=config.num_classes, dropout=0.0)
    ckpt_path = Path(config.checkpoint_dir) / checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    tensor = _INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, idx = probs.max(1)

    return config.class_names[idx.item()], confidence.item() * 100
