import os, sys, time
sys.path.append(".")
sys.path.append("..")
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from configs import data_configs
from datasets.images_dataset import ImagesDataset_vec as ImagesDataset
from utils.common import tensor2im
from options.test_options import TestOptions
from models.fmri2face import Fmri2Face

def run():
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)
    # update test options with options used during training for model
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = Namespace(**opts)
    net = Fmri2Face(opts)
    net.eval()
    net.cuda()
    print('Loading dataset for {}'.format(test_opts.dataset_type))
    dataset_args = data_configs.DATASETS[test_opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                        target_root=dataset_args['test_target_root'],
                                        vec_root=dataset_args['test_vec_root'],
                                        source_transform=transforms_dict['transform_inference'],
                                        target_transform=transforms_dict['transform_inference'],
                                        opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)
    if opts.n_images is None:
        opts.n_images = len(dataset)
    global_i = 0
    global_time = []
    device = 'cuda'
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            x, y, vec = input_batch
            vec = torch.squeeze(vec)
            x, y, vec = x.to(device).float(), y.to(device).float(), vec.to(device).float()
            tic = time.time()
            result_batch, _, _ = net.forward(x, v=vec, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
            toc = time.time()
            global_time.append(toc - tic)
        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.source_paths[global_i]
            basename_re = os.path.basename(im_path).split(".")[0] + '_recon.jpg'
            basename_so = os.path.basename(im_path).split(".")[0] + '.jpg'
            source = Image.open(im_path)
            source.save(os.path.join(out_path_results, basename_so))
            im_save_path = os.path.join(out_path_results, basename_re)
            result.save(im_save_path)
            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)
    f.close()

if __name__ == '__main__':
    run()
