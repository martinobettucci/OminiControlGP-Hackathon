from typing import Optional
from PIL import Image
import torch
from transformers import pipeline

_nsfw_pipeline: Optional[object] = None


def load_nsfw_pipeline() -> object:
    """Lazy load the NSFW detection pipeline."""
    global _nsfw_pipeline
    if _nsfw_pipeline is None:
        _nsfw_pipeline = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            use_fast=True,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nsfw_pipeline


def is_nsfw_image(image: Image.Image) -> bool:
    """Return True if the image is detected as NSFW."""
    classifier = load_nsfw_pipeline()
    result = classifier(image, top_k=5)
    for entry in result:
        label = entry["label"].lower()
        score = entry["score"]
        if label in {"nsfw", "porn", "sexy", "hentai"} and score > 0.4:
            return True
    return False


def zero_if_nsfw(image: Image.Image) -> Image.Image:
    """Return a black image if the given image is NSFW."""
    if is_nsfw_image(image):
        print("NSFW content detected, zeroing output")
        return Image.new("RGB", image.size, (0, 0, 0))
    return image
