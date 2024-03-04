import cv2
import einops
import numpy as np
import torch
from PIL import Image
import os, argparse

from ldm.models.diffusion.plms import PLMSSampler
from torch.utils.data import DataLoader
from dataset import ADE20KDataset, COCODataset
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from pytorch_lightning import seed_everything


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="output/ade20k"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/xxx.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K"],
        default="COCO"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ade20k.ckpt",
        help="path to checkpoint of model",
    )  

    opt = parser.parse_args()

    seed_everything(opt.seed)

    def get_state_dict(d):
        return d.get('state_dict', d)

    targetpth = opt.outdir
    if not os.path.exists(targetpth):
        os.mkdir(targetpth)

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model).cpu()

    state_dict = get_state_dict(torch.load(opt.ckpt, map_location=torch.device('cuda')))
    state_dict = get_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.cuda()

    sampler = PLMSSampler(model)

    if opt.dataset == 'ADE20K':
        dataset = ADE20KDataset(opt.data_root)
    elif opt.dataset == 'COCO':
        dataset = COCODataset(opt.data_root)

    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    for batch in dataloader:
        names = batch['sourcepath'][0].split('/')[-1]
        print('processing:',targetpth+'/'+names)
        N = 1
        z, c = model.get_input(batch, model.first_stage_key, bs=1)
        c_tkscls = c['tkscls'][0][:N]
        c_cat, c, view_ctrol = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c['viewcontrol'][0][:N]

        uc_cross = torch.zeros((c.shape[0],77),dtype=torch.int64) + 49407
        uc_cross[:,0] = 49406
        uc_cross = model.cond_stage_model.encode(uc_cross.to(model.device))
        if isinstance(uc_cross, DiagonalGaussianDistribution):
            uc_cross = uc_cross.mode()
            
        uc_cat = c_cat
        uc_tkscls = torch.zeros_like(c_tkscls) + 49407
        uc_tkscls[:,0] = 49406
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "tkscls": [uc_tkscls]}

        cond = {"c_concat": [c_cat], "c_crossattn": [c], "tkscls": [c_tkscls]}
        un_cond ={"c_concat": [uc_cat], "c_crossattn": [uc_cross], "tkscls": [uc_tkscls]}

        H,W=512,512
        shape = (4, H // 8, W // 8)
        samples_ddim, _ = sampler.sample(50,
                                        conditioning=cond,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=2.0,
                                        unconditional_conditioning=un_cond,
                                        eta=0.0,
                                        x_T=None)
        x_samples = model.decode_first_stage(samples_ddim)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(1)]
        Image.fromarray(results[0]).save(os.path.join(targetpth,names.replace('png','jpg')))

if __name__ == "__main__":
    main()
