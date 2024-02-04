
import json
from typing import TypedDict
from FlirCocoGenerator import FlirCocoGenerator
from FlirThermalCocoGenerator import FlirThermalCocoGenerator


class CustomDatasetDict(TypedDict):
    """
    CustomDatasetDict에 대한 설명을 작성해주세요.
    """

    # CUSTOM_SEQS_INFO_DICT에 정의되어있는 데이터 셋의 이름입니다.
    name: str
    # 해당 데이터 셋이 열화상인지를 나타냅니다.
    is_thermal: bool
    # 데이터 셋의 최상위 경로입니다.
    data_root: str


class FrameRrangeDict(TypedDict):
    """
    FrameRrangeDict 대한 설명을 작성해주세요.
    """

    # start에 대한 설명을 작성해주세요.
    start: float
    # end에 대한 설명을 작성해주세요.
    end: float


DATA_PARSE_LIST = [
    'flir_adas_v2',
    'flir_adas_v2_thermal',
    'flir_adas_v2_small',
    'flir_adas_v2_thermal_small'
]

# 기존 data가 저장되어 있는 위치
FLIR_DATA_ROOT = '/mnt/y/Datasets/flir_adas_v2/'
# 파싱 후 data가 저장될 루트 위치
FLIR_SAVE_ROOT = '/mnt/y/Datasets/flir_adas_v2/'
dataset_base_path = '/mnt/y/Datasets/flir_adas_v2'
VIS_THRESHOLD = 0.25  # 데이터 추가 정제가 필요할 경우 vis threshold 이하의 아이템은 버리는 방향으로 구현

# add custom sequence info here
CUSTOM_SEQS_INFO_DICT = {}
for dataitem in DATA_PARSE_LIST:
    if dataitem == 'flir_adas_v2' or dataitem == 'flir_adas_v2_small':
        CUSTOM_SEQS_INFO_DICT[dataitem] = {
            'train_sequences': {
                'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338},
                'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226},
                'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208},
                'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033}
            },
            'val_sequences': {
                'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540},
                'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}
            },
            'test_sequences': {
                'video-hnbGXq3nNPjBbc7CL': {'img_width': 1024, 'img_height': 1224, 'seq_length': 411},
                'video-msNEBxJE5PPDqenBM': {'img_width': 1024, 'img_height': 1224, 'seq_length': 428}
            }
        }
    elif dataitem == 'flir_adas_v2_thermal' or dataitem == 'flir_adas_v2_thermal_small':
        CUSTOM_SEQS_INFO_DICT[dataitem] = {
            'train_sequences': {
                'video-4FRnNpmSmwktFJKjg': {'img_width': 512, 'img_height': 640, 'seq_length': 338},
                'video-6tLtjdkv5K5BuhB37': {'img_width': 512, 'img_height': 640, 'seq_length': 226},
                'video-vbrSzr4vFTm5QwuGH': {'img_width': 512, 'img_height': 640, 'seq_length': 208},
                'video-ZAtDSNuZZjkZFvMAo': {'img_width': 512, 'img_height': 640, 'seq_length': 1033},
            },
            'val_sequences': {
                'video-ePoikf5LyTTfqchga': {'img_width': 512, 'img_height': 640, 'seq_length': 540},
                'video-t3f7QC8hZr6zYXpEZ': {'img_width': 512, 'img_height': 640, 'seq_length': 565},
            },
            'test_sequences': {
                'video-5RSrbWYu9eokv5bvR': {'img_width': 512, 'img_height': 640, 'seq_length': 411},
                'video-SCiKdG3MqZfiE292B': {'img_width': 512, 'img_height': 640, 'seq_length': 428},
            }
        }

rgb_seq_to_thermal_seq = {
    'video-BzZspxAweF8AnKhWK': 'video-4FRnNpmSmwktFJKjg',
    'video-FkqCGijjAKpABetZZ': 'video-6tLtjdkv5K5BuhB37',
    'video-PGdt7pJChnKoJDt35': 'video-vbrSzr4vFTm5QwuGH',
    'video-RMxN6a4CcCeLGu4tA': 'video-ZAtDSNuZZjkZFvMAo',
    'video-YnfPeH8i2uBWmsSd2': 'video-ePoikf5LyTTfqchga',
    'video-dvZBYnphN2BwdMKBc': 'video-t3f7QC8hZr6zYXpEZ',
    'video-hnbGXq3nNPjBbc7CL': 'video-5RSrbWYu9eokv5bvR',
    'video-msNEBxJE5PPDqenBM': 'video-SCiKdG3MqZfiE292B'
}


if __name__ == '__main__':
    def generate_coco_flir(
        name: str,
        is_thermal: bool = False,
        data_root: str = FLIR_DATA_ROOT,
        frame_range: FrameRrangeDict = {'start': 0.0, 'end': 1.0}
    ):

        if is_thermal:
            split_names_list = ['train_coco_t', 'val_coco_t', 'test_coco_t']
            root_split = 'train_t'
        else:
            split_names_list = ['train_coco', 'val_coco', 'test_coco']
            root_split = 'train'

        for splited_name in split_names_list:
            seqs_names_from_splited_name = "".join(
                [splited_name.split('_')[0], '_sequences'])

            if is_thermal:
                print('is_thermal')
                print(name, seqs_names_from_splited_name)
                print(CUSTOM_SEQS_INFO_DICT[name]
                      [seqs_names_from_splited_name])
                generator = FlirThermalCocoGenerator(split_name=splited_name,
                                                     seqs_names=CUSTOM_SEQS_INFO_DICT[name][seqs_names_from_splited_name],
                                                     root_split=root_split,
                                                     frame_range=frame_range,
                                                     data_root=data_root,
                                                     rgb_seq_to_thermal_seq=rgb_seq_to_thermal_seq,
                                                     save_root=FLIR_SAVE_ROOT,
                                                     dataset_base_path=None)
            else:
                generator = FlirCocoGenerator(split_name=splited_name,
                                              seqs_names=CUSTOM_SEQS_INFO_DICT[name][seqs_names_from_splited_name],
                                              root_split=root_split,
                                              frame_range=frame_range,
                                              data_root=data_root,
                                              save_root=FLIR_SAVE_ROOT,
                                              dataset_base_path=None)

            generator.generate()

    with open('./src/coco_parser_custom.json') as f:
        parse_list: list[CustomDatasetDict] = json.load(f)

    for payload in parse_list:
        generate_coco_flir(name=payload['name'],
                           is_thermal=payload['is_thermal'],
                           data_root=payload['data_root'])
