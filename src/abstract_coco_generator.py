from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import json


def load_sequence_config(config_file_path):
    from configparser import ConfigParser

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(
            f"seqinfo.ini not found in '{config_file_path}'")

    config = ConfigParser()
    config.read(config_file_path)

    img_width = int(config['Sequence']['imWidth'])
    img_height = int(config['Sequence']['imHeight'])
    seq_length = int(config['Sequence']['seqLength'])

    return img_width, img_height, seq_length

def calculate_max_objects_per_image(annotations: List[Dict]) -> int:
    """
    Calculates the maximum number of objects (annotations) per image.

    Args:
        annotations (List[Dict]): A list of dictionaries, each representing an annotation.

    Returns:
        int: The maximum number of annotations found for a single image.
    """
    num_objs_per_image = {}
    for anno in annotations:
        image_id = anno["image_id"]
        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    max_objs = max(num_objs_per_image.values(), default=0)
    return max_objs

class AbstractCocoGenerator(ABC):
    def __init__(
        self,
        split_name: str,
        seqs_names: Optional[List[str]],
        root_split: str,
        frame_range: Dict[str, float],
        data_root: str,
        save_root: str,
        dataset_base_path: Optional[str],
        gt_file_name: str
    ):
        self.split_name = split_name
        self.seqs_names = seqs_names
        self.root_split = root_split
        self.frame_range = frame_range
        self.data_root = data_root
        self.save_root = save_root
        self.dataset_base_path = dataset_base_path or os.path.join(
            os.getcwd(), os.path.join(data_root, root_split))
        self.gt_file_name = gt_file_name
        self.root_split_path = os.path.join(self.data_root, self.root_split)

        self.coco_orig_category_id_to_sorted_order_dict = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            8: 5,
            10: 6,
            11: 7,
            12: 8,
            79: 9,
            17: 10
        }

        self.orig_image_name_to_parsed_image_id = {}
    
    def __create_coco_directories(self):
        import shutil
        coco_dir = os.path.join(self.save_root, self.split_name)
        if os.path.isdir(coco_dir):
            shutil.rmtree(coco_dir)
        os.mkdir(coco_dir)

        annotations_dir = os.path.join(self.save_root, 'annotations')
        if not os.path.isdir(annotations_dir):
            os.mkdir(annotations_dir)

        return coco_dir, annotations_dir
   
    def __generate_coco_structure(self, sequences):
        return {
            'type': 'instances',
            'images': [],
            'annotations': [],
            'sequences': sequences,
            'frame_range': self.frame_range,
            'categories': [
                {'id': 1, 'name': 'person', 'supercategory': 'unknown'},
                {'id': 2, 'name': 'bike', 'supercategory': 'unknown'},
                {'id': 3, 'name': 'car', 'supercategory': 'unknown'},
                {'id': 4, 'name': 'motor', 'supercategory': 'unknown'},
                {'id': 5, 'name': 'truck', 'supercategory': 'unknown'},
                {'id': 6, 'name': 'light', 'supercategory': 'unknown'},
                {'id': 7, 'name': 'hydrant', 'supercategory': 'unknown'},
                {'id': 8, 'name': 'sign', 'supercategory': 'unknown'},
                {'id': 9, 'name': 'other vehicle', 'supercategory': 'unknown'},
                {'id': 10, 'name': 'dog', 'supercategory': 'unknown'}]
        }
        
    def __load_annotation_data(self):
        from pathlib import Path

        # Construct the full path to the JSON file
        gt_file_path = Path(self.root_split_path).parents[0].joinpath(
            self.gt_file_name)

        # Ensure the file exists
        if not gt_file_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: '{gt_file_path}'")

        # Load and return the JSON data
        with open(gt_file_path, "r") as gt_file:
            return json.load(gt_file)

    def __process_images_for_coco(self, coco_dir, seqs):

        img_id = 0  # 모든 시퀀스 대해 통합된 image id임

        images = []

        for seq in seqs:
            config_file_path = os.path.join(
                self.root_split_path, seq, 'seqinfo.ini')
            if not os.path.isfile(config_file_path):
                raise FileNotFoundError(
                    f"seqinfo.ini not found in '{config_file_path}'")

            img_width, img_height, seq_length = \
                load_sequence_config(config_file_path)

            seg_list_dir = os.listdir(
                os.path.join(self.root_split_path, seq, 'img1'))
            start_frame = int(self.frame_range['start'] * seq_length)
            end_frame = int(self.frame_range['end'] * seq_length)
            seg_list_dir = seg_list_dir[start_frame:end_frame]

            print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
            seq_length = len(seg_list_dir)

            for i, img in enumerate(sorted(seg_list_dir)):
                if i == 0:
                    first_frame_image_id = img_id

                images.append({
                    "file_name": f"{seq}_{img}",
                    "height": img_height,
                    "width": img_width,
                    "id": img_id,
                    "frame_id": i,
                    "seq_length": seq_length,
                    "first_frame_image_id": first_frame_image_id
                })

                self.orig_image_name_to_parsed_image_id[img] = img_id
                img_id += 1

                originPath = os.path.join(self.dataset_base_path, seq, 'img1', img)
                symlinkPath = os.path.join(coco_dir, f"{seq}_{img}")
                os.symlink(originPath, symlinkPath)

        return images
    
    def __process_annotations(self, annot_json_data, seqs):
        """
        Processes annotations for given sequences. Filters and transforms annotation data based on the sequences.

        Args:
            seqs: List of sequences to process.
            annot_json_data: Original annotation JSON data.

        Returns:
            List of processed annotations for the given sequences.
        """
        processed_annotations = []
        # Start with 0 or the next available ID if appending to existing annotations.
        annotation_id = 0
        nan_track_id_count = 0

        orig_image_id_to_image_name = {
            img_dict['id']: img_dict['file_name']
            for img_dict
            in annot_json_data['images']
        }

        for seq in seqs:
            for annot in annot_json_data['annotations']:
                image_name = \
                    orig_image_id_to_image_name.get(annot['image_id'], None)

                if image_name:
                    image_seq = image_name[5:28]

                if seq != image_seq:
                    continue

                image_id = self.orig_image_name_to_parsed_image_id.get(
                    image_name[5:],
                   None
                )

                if image_id is not None:
                    if 'track_id' in annot:
                        annotation = {
                            "id": annotation_id,
                            "bbox": annot['bbox'],
                            "image_id": image_id,
                            "segmentation": annot['segmentation'],
                            "ignore": 0 if annot['category_id'] else 1,
                            "visibility": 1.0,
                            "area": annot['area'],
                            "iscrowd": 1 if annot['iscrowd'] else 0,
                            "seq": image_seq,
                            "category_id": self.coco_orig_category_id_to_sorted_order_dict[annot['category_id']],
                            "track_id": annot['track_id']+1
                        }

                        processed_annotations.append(annotation)
                        annotation_id += 1
                    else:
                        nan_track_id_count += 1
                        print("track id is nan!", nan_track_id_count)
                        print("track id == nan annotation", annot)

        return processed_annotations
    
    @abstractmethod
    def generate_sequences(self):
        pass

    def generate(self):
        coco_dir, annotations_dir = self.__create_coco_directories()
        annotation_file = os.path.join(annotations_dir, f'{self.split_name}.json')
        seqs = self.generate_sequences()
        annot_json_data = self.__load_annotation_data()

        print(self.split_name, seqs)

        coco_structure = self.__generate_coco_structure(seqs)
        coco_structure['images'] = self.__process_images_for_coco(coco_dir, seqs)
        coco_structure['annotations'] = self.__process_annotations(
           annot_json_data,
           seqs,
        )

        max_objs_per_image = calculate_max_objects_per_image(
            coco_structure['annotations'])
        print(f'max objs per image: {max_objs_per_image}')

        with open(annotation_file, 'w') as anno_file:
            json.dump(coco_structure, anno_file, indent=4)