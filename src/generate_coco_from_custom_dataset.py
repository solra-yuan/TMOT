# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

CUSTOM_ROOT = '/app/object_detection/data/flir_adas_v2_orig/video_rgb_test'
TARGET_ROOT = '/app/object_detection/trackformer/data/flir_adas_v2_coco'
VIS_THRESHOLD = 0.25 ## 

FLIR_ADAS_V2_SEQS_INFO = {}



# dataset preview
if __name__ == '__main__':
    vidnames = sorted(os.listdir(os.path.join(CUSTOM_ROOT, 'data')))
    seqs = {v[:23]:None for v in vidnames}
    print("seqlen", len(seqs), seqs)

    for seq in seqs:
        # check sequence length
        seqlist = [name for name in vidnames if seq in name]
        print("seqname: ", seq, "length: ", len(seqlist))

        seq_first_im = plt.imread(os.path.join(CUSTOM_ROOT, 'data', seqlist[0]))
        seq_first_im_shape = seq_first_im.shape
        # data size check
        # for img_name in seqlist:
        #     im = plt.imread(os.path.join(CUSTOM_ROOT, 'data', img_name))
        #     if im.shape != seq_first_im_shape:
        #         print(im.shape)  ## raise assertionerror
        #     # check
        FLIR_ADAS_V2_SEQS_INFO[seq] = {'img_width': seq_first_im_shape[0],
                                       'img_height': seq_first_im_shape[1],
                                       'seq_length': len(seqlist)}
        
print(FLIR_ADAS_V2_SEQS_INFO)
# vidseq_to_names = {}
# video-5tYghHhFktjq4nQ5R-frame-001685-4kSNcWCSXX6FZyiZz
# video-5zpwfwcv9hXTFxw8m-frame-003207-EJa2An8JwK5o58BHh

def generate_coco_from_custom(split_name='train', seqs_names=None,
                              root_split='train', flir_adas_v2=False, 
                              frame_range=None, 
                              data_root='data/flir_adas_v2',
                              ):
    """
    Generates COCO data for Multi-object-tracking from custom dataset
    """

    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    # todo : check root setting ; target and original root
    if flir_adas_v2:
        data_root = TARGET_ROOT    

    # original customdata root
    root_split_custom_path = os.path.join(CUSTOM_ROOT, 'data')
    # target customdata root
    root_split_path = os.path.join(CUSTOM_ROOT, 'data') 
    coco_dir = os.path.join(data_root, split_name)    #target root?
    os.makedirs(coco_dir, exist_ok=True)


    #
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    
    ## todo : check labels
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

    # if split is {train,test} , CUSTOM annotation directory now resembles this.
    # |-data
    # | |-CUSTOMDATA
    # | | |-annotations
    # | | | |-train.json
    # | | | |-test.json
    
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')
    
    # IMAGE FILES
    img_id = 0

    # load img sequences(sorted) list
    full_seqs = sorted(os.listdir(root_split_custom_path))
    seqs = sorted(list({se[:23] for se in full_seqs}))
    if seqs_names is not None:
        seqs = [s for s in seqs if s in full_seqs]

    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, len(seqs))
    
    for seq in seqs:

        img_width = FLIR_ADAS_V2_SEQS_INFO[seq]['img_width']
        img_height = FLIR_ADAS_V2_SEQS_INFO[seq]['img_height']
        seq_length = FLIR_ADAS_V2_SEQS_INFO[seq]['seq_length']

        # custom datasets where
        # all image sequences is collected in one directories.

        this_seg_list = [f for f in full_seqs if seq in f]
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        this_seg_list = this_seg_list[start_frame: end_frame]
        print(f"{seq}: {len(this_seg_list)}/{seq_length}")
        seq_length = len(this_seg_list)

        # load file list in sorted order. [must be sorted?]
        for i, img in enumerate(sorted(this_seg_list)):
            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name":img,
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})
            if i == 0 :
                print(annotations['images'][-1])
            # "file_name": file name.
            # height : image height of any sequence image
            # width : image width of any sequence image
            # id : unique number inside total image sets
            # frame_id : frame order in sequence
            # seq_length : (sequence length) * (frame range ratio(0~1))
            # first_frame_image_id : image id of each sequence's first image
            img_id += 1
    
    # GT
    annotation_id = 0
    # img_file_name_to_id = {
    #      img_dict['file_name']: img_dict['id']
    #      for img_dict in annotations['images']}
    image_id_to_image_seq = {
        img_dict['id']: img_dict['file_name'][:23]
        for img_dict in annotations['images']
    }

    # for seq in seqs:   ## sequence 개수만큼 처리하고 있다. ㅇㅅㅇ.... 이 루프는 필요없다. 그래서 시퀀스 개수만큼 저장하는 건가?
    # GT FILE
    gt_file_path = os.path.join(Path(root_split_custom_path).parents[0], 'coco.json')
    if flir_adas_v2:
        gt_file_path = os.path.join(
            Path(root_split_custom_path).parents[0],'coco.json'
        )

    nan_track_id_count = 0
    # 이거 시퀀스별로 annotation 어떻게 나눠?
    # 설마 annotation에서 image id 보고
    # image id에 해당하는 image file 이름 읽어서
    # sequence 이름이랑 매칭한 후에
    # 시퀀스별로 저장이야?

    seq_annotations = []
    if flir_adas_v2:
        with open(gt_file_path, "r") as gt_file:
            annot_json_data = json.load(gt_file)

        # change annotation id to 0-based index.
        # don't ignore items
        # visibility set to 1
        # add sequence name field 
        flag1 = True
        for annot in annot_json_data['annotations']:
            if flag1:
                print("annot", annot)
                flag1 = False
            annotation = {
                "id": annotation_id,
                "bbox": annot['bbox'],
                "image_id": annot['image_id'],
                "segmentation": annot['segmentation'],
                "ignore": False if annot['category_id'] else False,
                "visibility": 1.0,
                "area": annot['area'],
                "iscrowd": annot['iscrowd'],
                "seq": image_id_to_image_seq[annot['image_id']],
                "category_id": coco_orig_category_id_to_sorted_order_dict[annot['category_id']],
                "track_id": annot['track_id'] if 'track_id' in annot else -1
                }
            if annotation['track_id'] == -1:
                nan_track_id_count += 1
                print("track id is nan!", nan_track_id_count)

            # {
            #   "annotations": [{"area": 4095, "bbox": [495, 441, 91, 45],
            #                    "category_id": 3,
            #                    "extra_info": {"human_annotated": "human"},
            #                    "id": 1, "image_id": 0, "iscrowd": false,
            #                    "segmentation": [[495,441,586,441, 495, 486,586,486]],
            #                    "track_id": 0},
            #                   {"area": 2106,"bbox": [

            seq_annotations.append(annotation)
            annotation_id += 1
        print('nan track id count', nan_track_id_count)
        
        annotations['annotations'].extend(seq_annotations)
        
        
    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1
    
    print("num_objs_per_image", num_objs_per_image)

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)
    
if __name__ == '__main__':
    generate_coco_from_custom(split_name='train', seqs_names=None,
                              root_split='train', flir_adas_v2=True, 
                              frame_range=None, 
                              data_root='data/flir_adas_v2',
                              )
