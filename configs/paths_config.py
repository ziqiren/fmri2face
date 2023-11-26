import os

dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'face_train': '',
    'face_test': './testdata/sub01',
}

model_paths = {
	'stylegan_ffhq': '../pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '../pretrained_models/model_ir_se50.pth',
	'parsing_net': '../pretrained_models/parsing.pth',
    'alex':'../pretrained_models/alex.pth',
	'circular_face': '../pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '../pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': '../pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': '../pretrained_models/mtcnn/onet.npy',
	'shape_predictor': '../pretrained_models/shape_predictor_68_face_landmarks.dat',
	'inversion': '../pretrained_models/inversion.pt',
    'fmri2face_sub01': './scripts/fmri2face/checkpoint/sub01_fmri2face.pt' ##
}
