import torchvision
from .backbone import Backbone

def clone_torchvision_models():
    """torchvision.models의 callable 속성들을 딕셔너리로 복제합니다."""
    models_dict = {}
    # dir()를 통해 속성들을 순회하며, callable한 객체만 추가합니다.
    for attr in dir(torchvision.models):
        if not attr.startswith("_"):
            candidate = getattr(torchvision.models, attr)
            if callable(candidate):
                models_dict[attr] = candidate
    return models_dict


class BackboneProvider:
    def __init__(self):
        # 백본 스키마에 따른 provider 함수들을 등록하는 딕셔너리
        self._providers = clone_torchvision_models()

    def register(self, schema: str, provider_func):
        """특정 스키마에 해당하는 provider 함수를 등록합니다."""
        self._providers[schema] = provider_func

    def get(
        self, 
        name,
        properties,
        options
    ):
        """
        요청받은 name에 해당하는 provider 함수가 있으면 호출하고,
        없으면 기본 provider 로직을 사용합니다.
        """
        if name in self._providers:
            model = self._providers[options.name]
        else:
            raise ValueError(
                f"No provider found for the given name: {name}")

        return Backbone(
            model,
            properties,
            options
        )