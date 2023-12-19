from typing import TypedDict, Any, Tuple, NotRequired, TypeAlias
from torch import Tensor
from PIL import Image

TPolygonSegmentation: TypeAlias = list[list[float]]

# Unused in this module, but imported in multiple submodules.
class EncodedRLE(TypedDict):  # noqa: Y049
    size: list[int]
    counts: str | bytes

class RLE(TypedDict):
    size: list[int]
    counts: list[int]

class Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: TPolygonSegmentation | RLE | EncodedRLE
    area: float
    bbox: list[float]
    iscrowd: int

class RandomState(TypedDict):
    random: Tuple[Any, ...]
    torch: Tensor

class CocoAnnotationExtensions(Annotation):
    prev_image: NotRequired[Image.Image]
    prev_target: NotRequired[Annotation]
    prev_prev_image: NotRequired[Image.Image]
    prev_prev_target: NotRequired[Annotation]