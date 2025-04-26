# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import copy
import csv
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

from solar_yuan_types import RandomState
from .coco import CocoDetection, make_coco_transforms


class FLIR_ADAS_V2(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, random_state_dict=None, flir_concat_size_tuple=None, frame_range=None, **kwargs):
        super(FLIR_ADAS_V2, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range
        self._frame_range = frame_range
        if random_state_dict is not None:
            self.random_state_dict = random_state_dict
        else:
            self.random_state_dict = None
        if flir_concat_size_tuple:
            self.flir_concat_size_tuple = flir_concat_size_tuple
        else:
            self.flir_concat_size_tuple = None

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        elif self._frame_range is not None:
            return self._frame_range
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        # if self.frame_range != None:
        #     start_frame = int(self.frame_range['start'] * self.coco.imgs[idx]['seq_length'])
        #     end_frame = int(self.frame_range['end'] * self.coco.imgs[idx]['seq_length'])
        #     return end_frame - start_frame  # Return the length of the selected range
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        if self.random_state_dict is not None:
            random_state = self.random_state_dict
        else:
            random_state: RandomState = {
               'random': random.getstate(),
               'torch': torch.random.get_rng_state()
            }

        img, target = self._getitem_from_id(
            idx, 
            random_state,
            random_jitter=False,
            concat_size_tuple=self.flir_concat_size_tuple
        )

        if self._prev_frame:
            frame_id = self.coco.imgs[idx]['frame_id']

            # PREV
            # first frame has no previous frame
            start = max(0, frame_id - self._prev_frame_range) 
            stop = min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1)
            if start > stop:
                raise ValueError(f"Invalid range for randint: start={start}, stop={stop}")
            prev_frame_id = random.randint(start, stop)
            prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + \
                prev_frame_id
            
            prev_img, prev_target = self._getitem_from_id(
                prev_image_id, 
                random_state,
                concat_size_tuple=self.flir_concat_size_tuple
            )

            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV frame equidistant as prev_frame
                prev_prev_frame_id = min(
                    max(0, prev_frame_id + prev_frame_id - frame_id), self.seq_length(idx) - 1)
                prev_prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + \
                    prev_prev_frame_id

                prev_prev_img, prev_prev_target = self._getitem_from_id(
                    prev_prev_image_id, 
                    random_state,
                    concat_size_tuple=self.flir_concat_size_tuple
                )
                
                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class FLIR_ADAS_V2_thermal(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, random_state_dict=None, flir_concat_size_tuple=None, frame_range=None, **kwargs):
        super(FLIR_ADAS_V2_thermal, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range
        self._frame_range = frame_range
        if random_state_dict is not None:
            self.random_state_dict = random_state_dict
        else:
            self.random_state_dict = None

        if flir_concat_size_tuple:
            self.flir_concat_size_tuple = flir_concat_size_tuple
        else:
            self.flir_concat_size_tuple = None

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        elif self._frame_range is not None:
            return self._frame_range
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        # if self.frame_range != None:
        #     start_frame = int(self.frame_range['start'] * self.coco.imgs[idx]['seq_length'])
        #     end_frame = int(self.frame_range['end'] * self.coco.imgs[idx]['seq_length'])
        #     return end_frame - start_frame  # Return the length of the selected range        
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        if self.random_state_dict is not None:
            random_state = self.random_state_dict
        else:
            random_state: RandomState = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}

        img, target = self._getitem_from_id(
            idx, random_state, 
            random_jitter=False,
            concat_size_tuple=self.flir_concat_size_tuple)

        if self._prev_frame:
            frame_id = self.coco.imgs[idx]['frame_id']

            # PREV
            # first frame has no previous frame
            prev_frame_id = random.randint(
                max(0, frame_id - self._prev_frame_range),
                min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1))
            prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + \
                prev_frame_id

            prev_img, prev_target = self._getitem_from_id(
                prev_image_id, 
                random_state,
                concat_size_tuple=self.flir_concat_size_tuple
            )
            
            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV frame equidistant as prev_frame
                prev_prev_frame_id = min(
                    max(0, prev_frame_id + prev_frame_id - frame_id), self.seq_length(idx) - 1)
                prev_prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + \
                    prev_prev_frame_id

                prev_prev_img, prev_prev_target = self._getitem_from_id(
                    prev_prev_image_id, 
                    random_state,
                    concat_size_tuple=self.flir_concat_size_tuple
                )

                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class FLIR_ADAS_V2_concat(CocoDetection):
    def __init__(self, 
                 *args, 
                 dataset_rgb: FLIR_ADAS_V2, 
                 dataset_thermal: FLIR_ADAS_V2_thermal, 
                 prev_frame_range=1, 
                 random_state_dict=None, 
                 flir_concat_size_tuple=None,
                 frame_range=None,
                 **kwargs
                ):
        super(FLIR_ADAS_V2_concat, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range
        self._frame_range = frame_range
        if random_state_dict is not None:
            self.random_state_dict = random_state_dict
        else:
            self.random_state_dict = None
        if flir_concat_size_tuple:
            self.flir_concat_size_tuple = flir_concat_size_tuple
        else:
            self.flir_concat_size_tuple = None

        self.dataset_rgb = dataset_rgb
        self.dataset_thermal = dataset_thermal

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        elif self._frame_range != None:
            return self._frame_range
        else:
            return {'start': 0, 'end': 1.0}
    
    def seq_length(self, idx):
        if self.frame_range != None:
            start_frame = int(self.frame_range['start'] * self.coco.imgs[idx]['seq_length'])
            end_frame = int(self.frame_range['end'] * self.coco.imgs[idx]['seq_length'])
            return end_frame - start_frame  # Return the length of the selected range
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __len__(self):
        l1 = len(self.dataset_rgb)
        l2 = len(self.dataset_thermal)
        assert l1 == l2,  'given concat element datasets length is different.'

        return l1
    # concat dataset의 index에 따라 이미지를 돌려줌
    # idx가 똑같다면 image도 똑같이 return받는다.

    def __getitem__(self, idx):
        # image를 받아서 concat해서 돌려줌 # target?
        img1, target1 = self.dataset_rgb[idx]
        img2, target2 = self.dataset_thermal[idx]
        
        # shared_transform = T.Resize([512, 640])

        # img1 = shared_transform(img1)
        # img2 = shared_transform(img2)
        concatenated_img = torch.cat([img1, img2[0:1,:,:]], dim=0)

        target = target1

        if target1.get('prev_image') is not None and target2.get('prev_image') is not None:
            concatenated_prev_img = torch.cat([
                target1['prev_image'], 
                target2['prev_image'][0:1,:,:]], 
                dim=0
            )
            target['prev_image'] = concatenated_prev_img


        if target1.get('prev_prev_image') is not None and target2.get('prev_prev_image') is not None:
            concatenated_prev_prev_img = torch.cat([
                target1['prev_prev_image'], 
                target2['prev_prev_image'][0:1,:,:]], 
                dim=0
            )
            target['prev_prev_image'] = concatenated_prev_prev_img

        
        # return concatenated_img, target1, target2
        return concatenated_img, target


class FLIR_ADAS_V2_concat_fix_random(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(FLIR_ADAS_V2_concat_fix_random, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        random_state: RandomState = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}

        img, target = self._getitem_from_id(
            idx, 
            random_state, 
            random_jitter=False
        )

        if self._prev_frame:
            frame_id = self.coco.imgs[idx]['frame_id']

            # PREV
            # first frame has no previous frame
            prev_frame_id = random.randint(
                max(0, frame_id - self._prev_frame_range),
                min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1))
            prev_image_id = \
                self.coco.imgs[idx]['first_frame_image_id'] + prev_frame_id

            prev_img, prev_target = self._getitem_from_id(
                prev_image_id,
                random_state
            )

            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target

            if self._prev_prev_frame:
                # PREV PREV frame equidistant as prev_frame
                prev_prev_frame_id = min(
                    max(0, prev_frame_id + prev_frame_id - frame_id), 
                    self.seq_length(idx) - 1
                )

                prev_prev_image_id = \
                    self.coco.imgs[idx]['first_frame_image_id'] + prev_prev_frame_id

                prev_prev_img, prev_prev_target = self._getitem_from_id(
                    prev_prev_image_id, 
                    random_state
                )

                target[f'prev_prev_image'] = prev_prev_img
                target[f'prev_prev_target'] = prev_prev_target

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])


def build_flir_adas_v2(image_set, args):
    # Set dataset training phase: {train, val}
    if image_set == 'train':
        root = Path(args.flir_adas_v2_path_train)
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range = args.track_prev_frame_range
    elif image_set == 'val':
        root = Path(args.flir_adas_v2_path_val)
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided flir_adas_v2 path {root} does not exist'
    # read split attribute
    split = getattr(args, f"{image_set}_split")
    # locate img folder and annotation file
    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"
    # transform
    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes)
    # return flir_adas_v2 dataset
    dataset = FLIR_ADAS_V2(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
    )

    return dataset


def build_flir_adas_v2_concat(image_set, args):
    # Set dataset training phase: {train, val}
    if image_set == 'train':
        root = Path(args.flir_adas_v2_path_train)
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range = args.track_prev_frame_range
        frame_range = {'start': 0.0, 'end': 0.8}
    elif image_set == 'val':
        root = Path(args.flir_adas_v2_path_val)
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
        frame_range = {'start': 0.8, 'end': 1.0}
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided flir_adas_v2 Det path {root} does not exist'
    

    # transform
    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes)  # transform before dataset
    flir_concat_random_state_dict = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
    flir_rgb_t_concat_size_tuple = (640, 512)


    # flir_adas_v2_thermal
    split = getattr(args, f"flir_adas_v2_thermal_{image_set}_split")  # 'train_coco_t'
    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"  # 'train_coco_t.json'
    # transform 어떻게 줘야 하지? 일단 img_transform max_size:1333 val_width 800이긴한데..

    dataset_thermal = FLIR_ADAS_V2_thermal(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        random_state_dict = flir_concat_random_state_dict,
        flir_concat_size_tuple = flir_rgb_t_concat_size_tuple,
        frame_range=frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
        thermal_to_rgb_H=args.thermal_to_rgb_H
    )

    # flir_adas_v2
    split = getattr(args, f"{image_set}_split")  # train_split: 'train_coco'
    img_folder = root / split  #
    ann_file = root / f"annotations/{split}.json"  # train_coco.json


    dataset_rgb = FLIR_ADAS_V2(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        random_state_dict = flir_concat_random_state_dict,
        flir_concat_size_tuple = flir_rgb_t_concat_size_tuple,
        frame_range=frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame        
    )

    # concat each dataset class
    dataset = FLIR_ADAS_V2_concat(
        img_folder, ann_file, transforms, norm_transforms,
        dataset_rgb=dataset_rgb, 
        dataset_thermal=dataset_thermal,
        prev_frame_range=prev_frame_range,
        random_state_dict = flir_concat_random_state_dict,
        flir_concat_size_tuple = flir_rgb_t_concat_size_tuple,
        frame_range=frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame)

    return dataset
