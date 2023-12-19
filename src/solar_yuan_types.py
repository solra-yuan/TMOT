from typing import TypedDict, Any, Tuple, NotRequired
from torch import Tensor
from PIL import Image

from pycocotools import coco

class RandomState(TypedDict):
    random: Tuple[Any, ...]
    torch: Tensor

class CocoAnnotationExtensions(coco._Annotation):
    prev_image: NotRequired[Image.Image]
    prev_target: NotRequired[coco._Annotation]
    prev_prev_image: NotRequired[Image.Image]
    prev_prev_target: NotRequired[coco._Annotation]