from sal.saliency_model import SaliencyModel
from sal.utils.resnet_encoder import resnet50encoder
from saliency_eval import load_image_as_variable, to_batch_variable
import os
import torchviz

saliency = SaliencyModel(
    resnet50encoder(pretrained=True),
    5,
    64,
    3,
    64,
    fix_encoder=True,
    use_simple_activation=False,
    allow_selector=True,
)
saliency.minimialistic_restore(os.path.join(os.path.dirname(__file__), "minsaliency"))
saliency.train(False)

# simply load an image
"""Loads an image and returns a pytorch Variable of shape (1, 3, H, W). Image will be normalised between -1 and 1."""
images = load_image_as_variable(os.path.join(os.path.dirname(__file__), "sal/utils/test2.jpg"), cuda=False)
print(f"images: {images.shape}")
selectors = 340  # 340 is a zebra
model_confidence = 6

_images, _selectors = (
    to_batch_variable(images, 4, cuda=False).float(),
    to_batch_variable(selectors, 1, cuda=False).long(),
)
print(f"_images: {_images.shape}, _selectors: {_selectors.shape}")

# Image will be normalized between -2 and 2
_images_2 = _images * 2
print(f"_images_2: {_images_2.shape}")

masks, exists_logits, cls_logits = saliency(_images * 2, _selectors, model_confidence=model_confidence)
print(f"masks: {masks.shape}, exists_logits: {exists_logits}, cls_logits: {cls_logits.shape}")

torchviz.make_dot(masks, params=dict(saliency.named_parameters()), show_attrs=True, show_saved=True).render(
    "saliency_model", format="pdf"
)
