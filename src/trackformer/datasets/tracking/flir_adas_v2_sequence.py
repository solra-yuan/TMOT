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

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if __name__ != '__main__':
    from ..coco import make_coco_transforms
    from ..transforms import Compose

 
class FLIR_ADAS_V2_Sequence(Dataset):
    """
    Reimplemented based on MOT17Sequence,
    this dataloader handles one FLIR_ADAS_V2 sequence.
    """
    data_folder = 'flir_adas_v2'

    def __init__(self, root_dir: str = 'data', seq_name: Optional[str] = None,
                 vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): orginally intended to take in 
                                   Threshold of visibility of persons
                                   above which they are selected,
                                   but generalized to take in all categories
        """
        super().__init__()

        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        
        self._data_dir = osp.join(root_dir, self.data_folder)

        # todo : separate train and test 
        self._train_seqs = [item[:23] for item in os.listdir(os.path.join(self._data_dir, 'video_rgb_test', 'data'))]
        self._test_seqs = [item[:23] for item in os.listdir(os.path.join(self._data_dir, 'video_rgb_test', 'data'))]

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self.data = []
        self.no_gt = True
        if seq_name is not None:
            full_seq_name = seq_name
            assert full_seq_name in self._train_seqs or full_seq_name in self._test_seqs, \
                'Image set does not exist: {}'.format(full_seq_name)

            self.data = self._sequence()
            self.no_gt = not osp.exists(self.get_gt_file_path())
            
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])
    
        return sample

    def _sequence(self) -> List[dict]:
        '''
    def _sequence(self) -> List[dict]:
        total = []
        for filename in sorted(os.listdir(self._data_dir)):
            extension = os.path.splitext(filename)[1]
            if extension in ['.png', '.jpg']:
                total.append({'im_path': osp.join(self._data_dir, filename)})

        return total        
        '''
        dets = {i: [] for i in range(1, self.seq_length + 1)}
        
        # accumulate total
        img_dir = osp.join(
            self.get_seq_path())

        boxes, visibility = self.get_track_boxes_and_visbility()

        total = [
            {'gt': boxes[i],
             'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
             'vis': visibility[i],
             'dets': dets[i]}
            for i in range(1, self.seq_length + 1)]

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
        

    def get_seq_path(self) -> str:
        """ Return directory path of sequence. """
        full_seq_name = self._seq_name

        if full_seq_name in self._train_seqs:
            return osp.join(osp.join(self._data_dir, 'video_rgb_test'), 'data')
        else:
            return osp.join(osp.join(self._data_dir, 'video_rgb_test'), 'data')

    def get_config_file_path(self) -> str:
        """ Return config file of sequence. """
        return osp.join(self.get_seq_path(), f'seqinfo_{self._seq_name}.ini')

    def get_gt_file_path(self) -> str:
        """ Return ground truth file of sequence. """
        return osp.join(osp.dirname(self.get_seq_path()), 'coco.json')


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
        
        seq_info_dict ={'video-BzZspxAweF8AnKhWK': {'img_width': 1024, 'img_height': 1224, 'seq_length': 338}, 
    'video-FkqCGijjAKpABetZZ': {'img_width': 1024, 'img_height': 1224, 'seq_length': 226}, 
    'video-PGdt7pJChnKoJDt35': {'img_width': 1024, 'img_height': 1224, 'seq_length': 208}, 
    'video-RMxN6a4CcCeLGu4tA': {'img_width': 768, 'img_height': 1024, 'seq_length': 1033}, 
    'video-YnfPeH8i2uBWmsSd2': {'img_width': 1024, 'img_height': 1224, 'seq_length': 540}, 
    'video-dvZBYnphN2BwdMKBc': {'img_width': 768, 'img_height': 1024, 'seq_length': 565}, 
    'video-hnbGXq3nNPjBbc7CL': {'img_width': 1024, 'img_height': 1224, 'seq_length': 411}, 
    'video-msNEBxJE5PPDqenBM': {'img_width': 1024, 'img_height': 1224, 'seq_length': 428}}


        return int(seq_info_dict[self._seq_name]['seq_length'] )

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
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ))))
        from trackformer.datasets.coco import make_coco_transforms   
        from trackformer.datasets.transforms import Compose

        FLIR = FLIR_ADAS_V2_Sequence(root_dir='/app/TMOT/data', seq_name='video-BzZspxAweF8AnKhWK')
        print(FLIR._seq_name)
        print(FLIR.__len__())
        motseq = FLIR._sequence()
        from pprint import pprint
        pprint(motseq[0])
        print("----------------")
        pprint(motseq[1])
        print("FLIR seq length", FLIR.seq_length)