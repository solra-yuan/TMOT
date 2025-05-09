from typing import Optional
from abstract_coco_generator import AbstractCocoGenerator


class FlirThermalCocoGenerator(AbstractCocoGenerator):
    def __init__(
        self,
        split_name: str,
        seqs_names: Optional[list[str]],
        root_split: str,
        frame_range: dict[str, float],
        data_root: str,
        save_root: str,
        rgb_seq_to_thermal_seq: dict[str, str],
        gt_file_name: str = 'coco_gt_t/coco.json',
        dataset_base_path: Optional[str] = None,
    ):
        super().__init__(
            split_name=split_name,
            seqs_names=seqs_names,
            root_split=root_split,
            frame_range=frame_range,
            data_root=data_root,
            save_root=save_root,
            dataset_base_path=dataset_base_path,
            gt_file_name=gt_file_name
        )
        self.rgb_seq_to_thermal_seq = rgb_seq_to_thermal_seq

    def generate_sequences(self):
        import os

        # File name list at root_split_path
        seq_root_split_path = os.path.join(
            self.data_root,
            self.root_split.split('_')[0]
        )

        print("seq_root_split_path", seq_root_split_path)
        seqs = sorted(os.listdir(seq_root_split_path))  # Sort file name list 

        # Parse thermal-rgb based on order of sorted RGB seq list
        seqs = [self.rgb_seq_to_thermal_seq[se] for se in seqs]

        if self.seqs_names is not None:
            seqs = [s for s in seqs if s in self.seqs_names]

        return seqs
