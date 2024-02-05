from abstract_coco_generator import AbstractCocoGenerator


class FlirCocoGenerator(AbstractCocoGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gt_file_name='coco_gt/coco.json')

    def generate_sequences(self):
        import os

        # sorted flie name list
        seqs = sorted(os.listdir(self.root_split_path))

        if self.seqs_names is not None:
            seqs = [s for s in seqs if s in self.seqs_names]

        return seqs
