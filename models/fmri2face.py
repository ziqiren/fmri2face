import torch
from torch import nn
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from models.fmriNet.fmrinet import fmri2map, map2style, adapter
from models.encoders import backbone_encoders

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt

def getkey_map2style(ckpt, net):
	map2style_dict = {}
	for k, v in net.items():
		if 'weight' in k:
			new_k = "encoder_refinestage_list.0." + k.replace('.weight', '.3.weight')
		elif 'bias' in k:
			new_k = "encoder_refinestage_list.0." + k.replace('.bias', '.3.bias')
		new_v = ckpt['state_dict'][new_k]
		map2style_dict[k] = new_v
	return map2style_dict

class Fmri2Face(nn.Module):
	def __init__(self, opts):
		super(Fmri2Face, self).__init__()
		self.set_opts(opts)
		self.fmri2map = fmri2map(vec_dim=1024) # sub01:74183; sub02: 75373; sub03:114506; sub04: 162388
		self.fmri2map.initialize_weights()
		self.map2style = map2style()
		self.adapter_high = adapter()
		self.stage = self.opts.training_stage if self.opts.is_training is True else self.opts.stage
		self.encoder_firststage = backbone_encoders.BackboneEncoderFirstStage(50, 'ir_se', self.opts)
		if self.stage > 1:
			self.encoder_refinestage_list = nn.ModuleList([backbone_encoders.BackboneEncoderRefineStage(50, 'ir_se', self.opts) for i in range(self.stage-1)])
		self.decoder = Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.load_weights()

	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):  # for test/eval
			if self.stage > self.opts.training_stage:
				raise ValueError(f'The stage must be no greater than {self.opts.training_stage} when testing!')
			print('Loading checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 1:
				for i in range(self.stage - 1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'),
																	 strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.fmri2map.load_state_dict(get_keys(ckpt, 'fmri2map'), strict=True)
			self.adapter_high.load_state_dict(get_keys(ckpt, 'adapter_high'), strict=False)
			self.map2style.load_state_dict(getkey_map2style(ckpt, self.map2style.state_dict()))
			self.__load_latent_avg(ckpt)

		elif (self.opts.checkpoint_path is not None) and self.opts.train_imgencoder: # for training stage 1
			print(f'Train: The {self.stage}-th image encoder is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
			
		elif (self.opts.checkpoint_path is not None) and (not self.opts.train_imgencoder) and self.opts.train_fmrinet: # for training stage 2
			print(f'Train: The fmri2map is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.fmri2map.load_state_dict(get_keys(ckpt, 'fmri2map'), strict=True)
			self.map2style.load_state_dict(getkey_map2style(ckpt, self.map2style.state_dict()))
			self.__load_latent_avg(ckpt)

		elif (self.opts.checkpoint_path is not None) and (not self.opts.train_fmrinet) and self.opts.is_training: # for training stage 3
			print(f'Train: The high-adapter is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.fmri2map.load_state_dict(get_keys(ckpt, 'fmri2map'), strict=True)
			self.adapter_high.load_state_dict(get_keys(ckpt, 'adapter_high'), strict=False)
			self.map2style.load_state_dict(getkey_map2style(ckpt, self.map2style.state_dict()))
			self.__load_latent_avg(ckpt)	

	def forward(self, x, v, resize=True, input_code=False, randomize_noise=True, return_latents=True, feature_level = 'high'):
		stage_output_list = []
		if input_code:
			codes = x
		else:
			codes = self.encoder_firststage(x)
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		input_is_latent = not input_code
		first_stage_output, _ = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
		stage_output_list.append(first_stage_output)
		if self.stage > 1:
			for i in range(self.stage-1):
				codes = codes + self.encoder_refinestage_list[i](x, self.face_pool(stage_output_list[i]))
				refine_stage_output, _ = self.decoder([codes],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
				stage_output_list.append(refine_stage_output)
		vec_map = self.fmri2map(v)
		codes_vec, codes_h, codes_m, codes_l = self.map2style(vec_map)
		if feature_level == 'high':
			codes_h_w = self.adapter_high(codes_h,vec_map,type='high')
			codes_vec_w = torch.cat((codes_h_w, codes_m, codes_l), dim=1)
		fmri_output, _ = self.decoder([codes_vec_w],input_is_latent=input_is_latent,randomize_noise=randomize_noise,return_latents=return_latents)
		if resize:
			images_vec = self.face_pool(fmri_output)
		if return_latents:
			return images_vec, codes, codes_vec
		else:
			return images_vec

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None): 
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
