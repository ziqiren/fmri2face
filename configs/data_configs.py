from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'celeba_img': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celeba_fmri': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['face_train'],
		'train_target_root': dataset_paths['face_train'],
		'train_vec_root': dataset_paths['face_train'],
		'test_source_root': dataset_paths['face_test'],
		'test_target_root': dataset_paths['face_test'],
		'test_vec_root': dataset_paths['face_test']
	},
}
