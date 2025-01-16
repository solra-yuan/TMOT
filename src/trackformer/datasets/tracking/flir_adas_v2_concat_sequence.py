"""
flir_adas_v2 sequence dataset.
"""
import configparser
import csv
import os
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from solar_yuan_types import RandomState

if __name__ != '__main__':
    from ..coco import make_coco_transforms
    from ..transforms import Compose

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


class FLIR_ADAS_V2_CONCATSequence(Dataset):
    """
    Reimplemented based on MOT17Sequence,
    this dataloader handles one FLIR_ADAS_V2_concat sequence.
    """
    data_folder = 'flir_adas_v2'

    def __init__(self, root_dir: str = 'data', seq_name: Optional[str] = None,
                 dets: str = '', vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): orginally intended to take in 
                                   Threshold of visibility of persons
                                   above which they are selected,
                                   but generalized to take in all categories
        """
        super().__init__()

        self._seq_name = seq_name.strip('_rgb_t')
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._data_dir = osp.join(root_dir, self.data_folder)

        # @TODO : separate train and test
        self._train_folders = os.listdir(os.path.join(self._data_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._data_dir, 'test'))

        self.transforms = Compose(make_coco_transforms(
            'val', img_transform, overflow_boxes=True))

        self.data = []
        self.no_gt = True
        if seq_name is not None:
            full_seq_name = seq_name.strip('_rgb_t')
            assert full_seq_name in self._train_folders or full_seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(full_seq_name)

            self.data = self._sequence()
            self.no_gt = not osp.exists(self.get_gt_file_path())
        self.no_gt = False #@TODO: 역시 죽을수도 있지만 강제입력한 것이므로 정리할것

    def __len__(self) -> int:
        return len(self.data)

    def set_random_state(self, random_seed):
        torch.manual_seed(random_seed)

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        new_random_seed = torch.randint(0, 10000, (1,)).item()
        self.set_random_state(new_random_seed)

        thermal_img = Image.open(data['thermal_im_path']).convert("RGB")
        thermal_width_orig, thermal_height_orig = thermal_img.size

        img, _ = self.transforms(img)
        thermal_img, _ = self.transforms(thermal_img)
        resize = transforms.Resize(img.size()[-2:])  
        resized_thermal_img = resize(thermal_img)
        width, height = img.size(2), img.size(1)

        concat_rgb_t_img = torch.cat([img, resized_thermal_img[0:1, :, :]], dim=0)

        sample = {}
        sample['img'] = concat_rgb_t_img
        sample['thermal_img_path'] = data['thermal_im_path']
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor(
            [int(height_orig), int(width_orig)])
        sample['thermal_orig_size'] = torch.as_tensor(
            [int(thermal_height_orig), int(thermal_width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample

    def load_total_seq_data_info(self,
                                 boxes,
                                 img_dir,
                                 thermal_img_dir,
                                 img_file_names,
                                 thermal_img_file_names,
                                 visibility,
                                 dets,
                                 seq_length):
        total_data_info = [
            {
                'gt': boxes[i],
                'im_path': osp.join(img_dir, img_file_names[i-1]),
                'vis': visibility[i],
                'dets': dets[i],
                'thermal_im_path': osp.join(thermal_img_dir, thermal_img_file_names[i-1])
            }
            for i in range(1, seq_length + 1)
        ]

        return total_data_info

    def get_img_dir(self, seq_path, imdir):
        img_dir = osp.join(seq_path, imdir)
        return img_dir

    def _sequence(self) -> List[dict]:

        dets = {i: [] for i in range(1, self.seq_length + 1)}
        # accumulate total

        img_dir = self.get_img_dir(
            self.get_seq_path(), self.config['Sequence']['imDir'])
        thermal_img_dir = self.get_img_dir(
            self.get_thermal_seq_path(), self.config['Sequence']['imDir'])
        boxes, visibility = self.get_track_boxes_and_visbility()

        img_file_names = sorted(os.listdir(img_dir))
        thermal_img_file_names = sorted(os.listdir(thermal_img_dir))

        total = self.load_total_seq_data_info(
            boxes,
            img_dir,
            thermal_img_dir,
            img_file_names,
            thermal_img_file_names,
            visibility,
            dets,
            self.seq_length
        )

        return total

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        """ Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}

        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self.get_gt_file_path()

        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1 # @TODO: row[7] is class, and now it only accepts 1. extend this to all class
                if int(float(row[6])) == 1 and int(float(row[7])) == 1 and float(row[8]) >= self._vis_threshold:

                    x1 = int(row[2])
                    y1 = int(row[3])
                    x2 = x1 + int(row[4])
                    y2 = y1 + int(row[5])
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    frame_id = int(row[0])
                    track_id = int(row[1])

                    boxes[frame_id][track_id] = bbox
                    visibility[frame_id][track_id] = float(row[8])

        return boxes, visibility

    def get_seq_path(self) -> str:
        """ Return directory path of sequence. """
        full_seq_name = self._seq_name

        if full_seq_name in self._train_folders:
            return osp.join(osp.join(self._data_dir, 'train'), full_seq_name)
        else:
            return osp.join(osp.join(self._data_dir, 'test'), full_seq_name)

    def get_thermal_seq_path(self) -> str:
        """ Return directory path of sequence. """
        full_seq_name = self._seq_name
        full_corresponding_thermal_seq_name = rgb_seq_to_thermal_seq[self._seq_name]

        if full_seq_name in self._train_folders:
            return osp.join(osp.join(self._data_dir, 'train_t'), full_corresponding_thermal_seq_name)
        else:
            return osp.join(osp.join(self._data_dir, 'test_t'), full_corresponding_thermal_seq_name)

    def get_config_file_path(self) -> str:
        """ Return config file of sequence. """
        return osp.join(self.get_seq_path(), 'seqinfo.ini')

    def get_gt_file_path(self) -> str:
        """ Return ground truth file of sequence. """
        return osp.join(self.get_seq_path(), 'gt', 'gt.txt')

    def get_det_file_path(self) -> str:
        """ Return public detections file of sequence. """
        if self._dets is None:
            return ""

        return osp.join(self.get_seq_path(), 'det', 'det.txt')

    @property
    def config(self) -> dict:
        """ Return config of sequence. """
        config_file = self.get_config_file_path()

        assert osp.exists(config_file), \
            f'Config file does not exist: {config_file}'

        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    @property
    def seq_length(self) -> int:
        """ Return sequence length, i.e, number of frames. """
        return int(self.config['Sequence']['seqLength'])

    def __str__(self) -> str:
        return f"{self._seq_name}"

    @property
    def results_file_name(self) -> str:
        """ Generate file name of results file. """
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        return f"{self}.txt"

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self.results_file_name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])

    def load_results(self, results_dir: str) -> dict:
        results = {}
        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if track_id not in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = {}
                results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
                results[track_id][frame_id]['score'] = 1.0

        return results


if __name__ == '__main__':
    if __package__ == None:
        import sys
        from os import path
        print(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(path.dirname(path.dirname(
            path.dirname(path.dirname(path.abspath(__file__))))))
        from trackformer.datasets.coco import make_coco_transforms
        from trackformer.datasets.transforms import Compose

        FLIR = FLIR_ADAS_V2_CONCATSequence(
            seq_name='video-BzZspxAweF8AnKhWK_rgb_t')
        print(FLIR._seq_name)
        print(FLIR.__len__())
        motseq = FLIR._sequence()
        from pprint import pprint
        pprint(motseq[0])
        print("----------------")
        pprint(motseq[1])
        print("FLIR seq length", FLIR.seq_length)
