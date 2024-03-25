import argparse
import json
from os import environ
from typing import TypedDict
from FlirCocoGenerator import FlirCocoGenerator
from FlirThermalCocoGenerator import FlirThermalCocoGenerator
from abstract_coco_generator import FrameRangeDict
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv('.env.local'))
except ImportError:
    pass

parser = argparse.ArgumentParser(description='Argparse Example')

parser.add_argument(
    '--path',
    type=str,
    default=environ.get("DATA_PATH") or './src/coco_parser_custom.json',
    help='Dataset path'
)


args = parser.parse_args()

#---
def set_split_frame_range(
    frame_range_for_split,
    train_and_val_share_sequence,
    split_frame_ratio=0.8
):
    if train_and_val_share_sequence:
        frame_range_for_split['train']['end'] = split_frame_ratio
        frame_range_for_split['val']['start'] = split_frame_ratio
#----

class CustomDatasetDict(TypedDict):
    """
    Defines the properties of the datasets that will be parsed.
    """

    name: str
    '''
        The name of dataset which is defined in coco_parser_custom.json.
        Also, the name must be included in DATA_PARSE_LIST.
    '''

    is_thermal: bool
    '''
        Whether the dataset is thermal dataset or not. 
        if is_thermal is true, the dataset is thermal image.
        if is_thermal is false, the dataset is RGB image.

    '''

    data_root: str
    '''
        The root directory of the dataset.
    '''


DATA_PARSE_LIST = [
    'flir_adas_v2',
    'flir_adas_v2_thermal',
    'flir_adas_v2_small',
    'flir_adas_v2_thermal_small'
]

dataset_base_path = '/mnt/y/Datasets/flir_adas_v2'
VIS_THRESHOLD = 0.25 ## <<<<
TRAIN_AND_VAL_SHARE_SEQUENCE = True ## >>>>
SPLIT_FRAME_RATIO = 0.8 ## >>>>

# add custom sequence info here
CUSTOM_SEQS_INFO_DICT = {}
for dataitem in DATA_PARSE_LIST:
    if dataitem == 'flir_adas_v2' or dataitem == 'flir_adas_v2_small':
        CUSTOM_SEQS_INFO_DICT[dataitem] = {
            'train_sequences': {
                'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338},
                'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226},
                'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208},
                'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033},
                'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540},
                'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}
            }, ## >>>>
            'val_sequences': {
                'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338},
                'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226},
                'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208},
                'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033},
                'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540},
                'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}
            }, ## >>>>
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
                'video-ePoikf5LyTTfqchga': {'img_width': 512, 'img_height': 640, 'seq_length': 540},
                'video-t3f7QC8hZr6zYXpEZ': {'img_width': 512, 'img_height': 640, 'seq_length': 565},
            }, ## >>>>
            'val_sequences': {
                'video-4FRnNpmSmwktFJKjg': {'img_width': 512, 'img_height': 640, 'seq_length': 338},
                'video-6tLtjdkv5K5BuhB37': {'img_width': 512, 'img_height': 640, 'seq_length': 226},
                'video-vbrSzr4vFTm5QwuGH': {'img_width': 512, 'img_height': 640, 'seq_length': 208},
                'video-ZAtDSNuZZjkZFvMAo': {'img_width': 512, 'img_height': 640, 'seq_length': 1033},
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
        data_root: str, ## <<<<
        save_root: str, ## <<<<
        is_thermal: bool = False,
        train_and_val_share_sequence=False
    ):
        # ---- <<<<
        if is_thermal:
            split_names_list = ['train_coco_t', 'val_coco_t', 'test_coco_t']
            root_split = 'train_t'
        else:
            split_names_list = ['train_coco', 'val_coco', 'test_coco']
            root_split = 'train'
        # ---- <<<<
        # ---- >>>>
        frame_range_for_split = {
            'train': {'start': 0.0, 'end': 1.0},
            'val': {'start': 0.0, 'end': 1.0},
            'test': {'start': 0.0, 'end': 1.0}
        }
        set_split_frame_range(
            frame_range_for_split,
            train_and_val_share_sequence,
            split_frame_ratio=SPLIT_FRAME_RATIO
        )
        # ---- >>>>
        for splited_name in split_names_list:
            splited_name_phase = splited_name.split('_')[0]
            seqs_names_from_splited_name = "".join(
                [splited_name_phase, '_sequences'])

            frame_range = frame_range_for_split[splited_name_phase] # >>>>

            if is_thermal:
                print('is_thermal')
                print(name, seqs_names_from_splited_name)
                print(CUSTOM_SEQS_INFO_DICT[name]
                      [seqs_names_from_splited_name])
                generator = FlirThermalCocoGenerator(
                    split_name=splited_name,
                    seqs_names=CUSTOM_SEQS_INFO_DICT[name][seqs_names_from_splited_name],
                    root_split=root_split,
                    frame_range=frame_range,
                    data_root=data_root,
                    rgb_seq_to_thermal_seq=rgb_seq_to_thermal_seq,
                    save_root=save_root
                )
            else:
                generator = FlirCocoGenerator(
                    split_name=splited_name,
                    seqs_names=CUSTOM_SEQS_INFO_DICT[name][seqs_names_from_splited_name],
                    root_split=root_split,
                    frame_range=frame_range,
                    data_root=data_root,
                    save_root=save_root,
                )

            generator.generate()

    with open(args.path) as f:
        parse_list: list[CustomDatasetDict] = json.load(f)

    for payload in parse_list: # >>>>
        generate_coco_flir(name=payload['name'],
                           is_thermal=payload['is_thermal'],
                           data_root=payload['data_root'],
                           train_and_val_share_sequence=TRAIN_AND_VAL_SHARE_SEQUENCE)
