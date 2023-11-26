from argparse import ArgumentParser
from configs.paths_config import model_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, default='output_path/to/fmri2face/sub01/',
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=model_paths['fmri2face_sub01'], type=str,
                                 help='Path to model checkpoint')
        self.parser.add_argument('--stage', default=2, type=int, help='Results of stage i')
        self.parser.add_argument('--is_training', default=False, type=bool, help='Training or testing')
        self.parser.add_argument('--couple_outputs', action='store_true',
                                 help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', default=True, action='store_true',
                                 help='Whether to resize outputs4')
        self.parser.add_argument('--save_inverted_codes', action='store_true',
                                 help='Whether to save the inverted latent codes')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=0, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--dataset_type', default='celeba_fmri', type=str, help='Type of dataset/experiment to run')
    def parse(self):
        opts = self.parser.parse_args()
        return opts