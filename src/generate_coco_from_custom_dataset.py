"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

from trackformer.datasets.tracking.mots20_sequence import load_mots_gt

CUSTOM_ROOT = '/app/TMOT/data/flir_adas_v2/'
VIS_THRESHOLD = 0.25

FLIR_ADAS_V2 = True
FLIR_ADAS_V2_thermal = False

# add custom sequence info here
if FLIR_ADAS_V2:
    CUSTOM_SEQS_INFO = {
        'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338}, 
        'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226}, 
        'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208}, 
        'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033}, 
        'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540}, 
        'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}, 
        'video-hnbGXq3nNPjBbc7CL': {'img_width': 1024, 'img_height': 1224, 'seq_length': 411}, 
        'video-msNEBxJE5PPDqenBM': {'img_width': 1024, 'img_height': 1224, 'seq_length': 428}
        }
elif FLIR_ADAS_V2_thermal:
    CUSTOM_SEQS_INFO = {
        'video-4FRnNpmSmwktFJKjg': {'img_width': 512, 'img_height': 640, 'seq_length': 338}, 
        'video-5RSrbWYu9eokv5bvR': {'img_width': 512, 'img_height': 640, 'seq_length': 411}, 
        'video-6tLtjdkv5K5BuhB37': {'img_width': 512, 'img_height': 640, 'seq_length': 226}, 
        'video-SCiKdG3MqZfiE292B': {'img_width': 512, 'img_height': 640, 'seq_length': 428}, 
        'video-ZAtDSNuZZjkZFvMAo': {'img_width': 512, 'img_height': 640, 'seq_length': 1033}, 
        'video-ePoikf5LyTTfqchga': {'img_width': 512, 'img_height': 640, 'seq_length': 540}, 
        'video-t3f7QC8hZr6zYXpEZ': {'img_width': 512, 'img_height': 640, 'seq_length': 565}, 
        'video-vbrSzr4vFTm5QwuGH': {'img_width': 512, 'img_height': 640, 'seq_length': 208}
    }
    # CUSTOM_SEQS_INFO = {
    #     'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338}, 
    #     'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226}, 
    #     'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208}, 
    #     'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033}, 
    #     'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540}, 
    #     'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}, 
    #     'video-hnbGXq3nNPjBbc7CL': {'img_width': 1024, 'img_height': 1224, 'seq_length': 411}, 
    #     'video-msNEBxJE5PPDqenBM': {'img_width': 1024, 'img_height': 1224, 'seq_length': 428}
    #     }

    # thermal_seq_to_rgb_seq = {}
    rgb_seq_to_thermal_seq = {'video-BzZspxAweF8AnKhWK': 'video-4FRnNpmSmwktFJKjg',
                              'video-FkqCGijjAKpABetZZ': 'video-6tLtjdkv5K5BuhB37',
                              'video-PGdt7pJChnKoJDt35': 'video-vbrSzr4vFTm5QwuGH',
                              'video-RMxN6a4CcCeLGu4tA': 'video-ZAtDSNuZZjkZFvMAo',
                              'video-YnfPeH8i2uBWmsSd2': 'video-ePoikf5LyTTfqchga',
                              'video-dvZBYnphN2BwdMKBc': 'video-t3f7QC8hZr6zYXpEZ',
                              'video-hnbGXq3nNPjBbc7CL': 'video-5RSrbWYu9eokv5bvR',
                              'video-msNEBxJE5PPDqenBM': 'video-SCiKdG3MqZfiE292B'}



def generate_coco_from_custom(split_name='train', seqs_names=None,
                           root_split='train', flir_adas_v2=False, flir_adas_v2_thermal=False,
                           frame_range=None, data_root='data/flir_adas_v2'):
    """
    Generates COCO data from CUSTOM DATA.
    """

    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    if flir_adas_v2:
        data_root = CUSTOM_ROOT
    elif flir_adas_v2_thermal:
        data_root = CUSTOM_ROOT

    root_split_path = os.path.join(data_root, root_split)
    if flir_adas_v2_thermal:
        root_split_path = os.path.join(data_root, root_split.split('_')[0])
    coco_dir = os.path.join(data_root, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [ {'id': 1, 'name': 'person', 'supercategory': 'unknown'},
                                  {'id': 2, 'name': 'bike', 'supercategory': 'unknown'},
                                  {'id': 3, 'name': 'car', 'supercategory': 'unknown'},
                                  {'id': 4, 'name': 'motor', 'supercategory': 'unknown'},
                                  {'id': 5, 'name': 'truck', 'supercategory': 'unknown'},
                                  {'id': 6, 'name': 'light', 'supercategory': 'unknown'},
                                  {'id': 7, 'name': 'hydrant', 'supercategory': 'unknown'},
                                  {'id': 8, 'name': 'sign', 'supercategory': 'unknown'},
                                  {'id': 9, 'name': 'other vehicle', 'supercategory': 'unknown'},
                                  {'id': 10, 'name': 'dog', 'supercategory': 'unknown'}]
    # {'id': 1, 'name': 'person', 'supercategory': 'unknown'}
    # {'id': 2, 'name': 'bike', 'supercategory': 'unknown'}
    # {'id': 3, 'name': 'car', 'supercategory': 'unknown'}
    # {'id': 4, 'name': 'motor', 'supercategory': 'unknown'}
    # {'id': 8, 'name': 'truck', 'supercategory': 'unknown'}
    # {'id': 10, 'name': 'light', 'supercategory': 'unknown'}
    # {'id': 11, 'name': 'hydrant', 'supercategory': 'unknown'}
    # {'id': 12, 'name': 'sign', 'supercategory': 'unknown'}
    # {'id': 79, 'name': 'other vehicle', 'supercategory': 'unknown'}
    # {'id': 17, 'name': 'dog', 'supercategory': 'unknown'}
    coco_orig_category_id_to_sorted_order_dict = {1:1, 
                                                  2:2, 
                                                  3:3, 
                                                  4:4, 
                                                  8:5, 
                                                  10:6, 
                                                  11:7, 
                                                  12:8, 
                                                  79:9, 
                                                  17:10}


    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0  # 모든 시퀀스 대해 통합된 image id임

    print("root_split_path", root_split_path)
    # seqs : root_split_path에서 정렬된 이름.
    seqs = sorted(os.listdir(root_split_path))
    if flir_adas_v2_thermal:
        seqs = [rgb_seq_to_thermal_seq[se] for se in seqs]
        root_split_path = os.path.join(data_root, root_split)
        print("root_split_path, thermal:", root_split_path)
    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)

    orig_image_name_to_parsed_image_id = {}

    for seq in seqs:
        # CONFIG FILE
        config = configparser.ConfigParser()
        config_file = os.path.join(root_split_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])

        seg_list_dir = os.listdir(os.path.join(root_split_path, seq, 'img1'))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):
            
            if i == 0:
                first_frame_image_id = img_id
                print("seq first file name ", img)

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            orig_image_name_to_parsed_image_id[img] = img_id
            if i == 0 :
                print(annotations['images'][-1]) # peek last parsed image_annot
                print("orig_img_name",img, "img_id:", img_id)
            img_id += 1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img),
                       os.path.join(coco_dir, f"{seq}_{img}"))
    # GT
    annotation_id = 0

    # GT FILE
    gt_file_path = os.path.join(Path(root_split_path).parents[0], 'coco_gt', 'coco.json')
    print("gt_file_path", gt_file_path)
    if flir_adas_v2:
        gt_file_path = os.path.join(
            Path(root_split_path).parents[0], 
            'coco_gt', 
            'coco.json')
    if flir_adas_v2_thermal:
        gt_file_path = os.path.join(
            Path(root_split_path).parents[0], 
            'coco_gt_t', 
            'coco.json')
    nan_track_id_count = 0
    for seq in seqs:       
        if not os.path.isfile(gt_file_path):
            continue
        
        seq_annotations = []

        if flir_adas_v2 or flir_adas_v2_thermal:
            with open(gt_file_path, "r") as gt_file:
                annot_json_data = json.load(gt_file)

            # coco from mot할 땐 mot annotation이 이게 seq의 어느 frame_id에서 온지 정보가 있음
            # sorting된 시퀀스 내에서 sorting된 frame_id는 이름의 인덱스가 됨
            # -> (frame_id와 seq)를 key로 image name to image_id를 추출할 수 있음
            # flir_adas_v2의 annotation에서는 img_id가 존재
            # 오리지널 images annotation을 참조, 
            # img_id로부터 img 이름을 얻고 seq에 포함되지 않으면 filter out하고 나머지만 parse
                
            orig_image_id_to_image_name = {
                            img_dict['id']: img_dict['file_name']
                            for img_dict in annot_json_data['images']}
            # change annotation id to 0-based index.
            # don't ignore items
            # visibility set to 1
            # add sequence name field
            print_first = True
            for annot in annot_json_data['annotations']:
                image_name = orig_image_id_to_image_name.get(annot['image_id'], None)
                if image_name:
                    image_seq = image_name[5:28]
                if seq != image_seq:
                    continue
                if print_first:
                    print("first annot", annot)
                    print("first annotation of the seq:", image_seq)
                    print("annot['image_id']: ",annot['image_id'])
                    print("orig_image_id_to_image_name[annot['image_id]]: ",orig_image_id_to_image_name[annot['image_id']] )
                    print_first = False
                
                image_id = orig_image_name_to_parsed_image_id.get(image_name[5:], None)
                if image_id is None:
                    continue

                if 'track_id' in annot:
                    annotation = {
                        "id" : annotation_id,
                        "bbox":annot['bbox'],
                        "image_id": image_id,
                        "segmentation": annot['segmentation'],
                        "ignore": 0 if annot['category_id'] else 1,
                        "visibility": 1.0,
                        "area": annot['area'],
                        "iscrowd": 1 if annot['iscrowd'] else 0,
                        "seq": image_seq,
                        "category_id": coco_orig_category_id_to_sorted_order_dict[annot['category_id']],
                        "track_id": annot['track_id']+1} # track_id는 1부터 시작함
                else:     
                    nan_track_id_count += 1
                    print("track id is nan!", nan_track_id_count)
                    print("track id == nan annotation", annotation)
                    continue
                
                seq_annotations.append(annotation)
                annotation_id += 1
                # if frame_id not in seq_annoataions_per_frame: # needed when mots_vis = true, ignore this for now

        annotations['annotations'].extend(seq_annotations)

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)

if __name__ == '__main__':
    # generate_coco_from_custom(split_name='train_custom_testing', seqs_names=None,
    #                       root_split='train', flir_adas_v2=True,
    #                       data_root='data/flir_adas_v2')
    if FLIR_ADAS_V2:
        """
        {'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338}, # train
        'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226}, # train
        'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208}, # train
        'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033}, # val
        'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540}, # val
        'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}, # val
        'video-hnbGXq3nNPjBbc7CL': {'img_width': 1024, 'img_height': 1224, 'seq_length': 411}, # test
        'video-msNEBxJE5PPDqenBM': {'img_width': 1024, 'img_height': 1224, 'seq_length': 428}} # test
        """
        train_sequences = ['video-BzZspxAweF8AnKhWK', 
                        'video-FkqCGijjAKpABetZZ', 
                        'video-PGdt7pJChnKoJDt35',
                        'video-RMxN6a4CcCeLGu4tA']
        val_sequences = [
                        'video-YnfPeH8i2uBWmsSd2',
                        'video-dvZBYnphN2BwdMKBc',]
        test_sequences = ['video-hnbGXq3nNPjBbc7CL',
                        'video-msNEBxJE5PPDqenBM']
        generate_coco_from_custom(split_name='train_coco', seqs_names=train_sequences,
                                root_split='train', flir_adas_v2=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='val_coco', seqs_names=val_sequences,
                                root_split='train', flir_adas_v2=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='test_coco', seqs_names=test_sequences,
                                root_split='train', flir_adas_v2=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='all_coco', seqs_names=train_sequences+val_sequences+test_sequences,
                                root_split='train', flir_adas_v2=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )        
    elif FLIR_ADAS_V2_thermal:
        """
        {
        'video-4FRnNpmSmwktFJKjg': {'img_width': 512, 'img_height': 640, 'seq_length': 338}, # train
        'video-5RSrbWYu9eokv5bvR': {'img_width': 512, 'img_height': 640, 'seq_length': 411}, # test
        'video-6tLtjdkv5K5BuhB37': {'img_width': 512, 'img_height': 640, 'seq_length': 226}, # train
        'video-SCiKdG3MqZfiE292B': {'img_width': 512, 'img_height': 640, 'seq_length': 428}, # test
        'video-ZAtDSNuZZjkZFvMAo': {'img_width': 512, 'img_height': 640, 'seq_length': 1033}, # val
        'video-ePoikf5LyTTfqchga': {'img_width': 512, 'img_height': 640, 'seq_length': 540}, # val
        'video-t3f7QC8hZr6zYXpEZ': {'img_width': 512, 'img_height': 640, 'seq_length': 565}, # val
        'video-vbrSzr4vFTm5QwuGH': {'img_width': 512, 'img_height': 640, 'seq_length': 208} # train
        }

        """
        train_sequences = ['video-4FRnNpmSmwktFJKjg', 
                           'video-6tLtjdkv5K5BuhB37', 
                           'video-vbrSzr4vFTm5QwuGH',
                           'video-ZAtDSNuZZjkZFvMAo']
        val_sequences = [
                         'video-ePoikf5LyTTfqchga', 
                         'video-t3f7QC8hZr6zYXpEZ']
        test_sequences = ['video-5RSrbWYu9eokv5bvR', 
                          'video-SCiKdG3MqZfiE292B']
        generate_coco_from_custom(split_name='train_coco_t', seqs_names=train_sequences,
                                root_split='train_t', flir_adas_v2_thermal=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='val_coco_t', seqs_names=val_sequences,
                                root_split='train_t', flir_adas_v2_thermal=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='test_coco_t', seqs_names=test_sequences,
                                root_split='train_t', flir_adas_v2_thermal=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )
        generate_coco_from_custom(split_name='all_coco_t', seqs_names=train_sequences+val_sequences+test_sequences,
                                root_split='train_t', flir_adas_v2_thermal=True, 
                                frame_range=None, 
                                data_root=CUSTOM_ROOT,
                                )        