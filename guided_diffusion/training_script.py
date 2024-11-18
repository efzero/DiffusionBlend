import os
import json
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from pathlib import Path

from guided_diffusion.script_util import create_model, create_gaussian_diffusion
from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path
from physics.ct import CT
from physics.mri import SinglecoilMRI_comp, MulticoilMRI
from utils import CG, clear, get_mask, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from functools import partial

torch.set_printoptions(sci_mode=False)
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--type", type=str, required=True, help="Either [2d, 3d]"
    )
    parser.add_argument(
        "--CG_iter", type=int, default=5, help="Inner number of iterations for CG"
    )
    parser.add_argument(
        "--Nview", type=int, default=16, help="number of projections for CT"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="./exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--ckpt_load_name", type=str, default="AAPM256_1M.pt", help="Load pre-trained ckpt"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--rho", type=float, default=10.0, help="rho"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.04, help="lambda for TV"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="regularizer for noisy recon"
    )
    parser.add_argument(
        "--T_sampling", type=int, default=50, help="Total number of sampling steps"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="./results",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/media/harry/tomo/AAPM_data_vol/256_sorted/L067", help="The folder of the dataset"
    )
    
    # MRI-exp arguments
    parser.add_argument(
        "--mask_type", type=str, default="uniform1d", help="Undersampling type"
    )
    parser.add_argument(
        "--acc_factor", type=int, default=4, help="acceleration factor"
    )
    parser.add_argument(
        "--nspokes", type=int, default=30, help="Number of sampled lines in radial trajectory"
    )
    parser.add_argument(
        "--center_fraction", type=float, default=0.08, help="ACS region"
    )
    

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs/vp", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if "CT" in args.deg:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"view{args.Nview}"
    elif "MRI" in args.deg:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"{args.mask_type}_acc{args.acc_factor}"
                            
    args.image_folder.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        for t in ["input", "recon", "label"]:
            if t == "recon":
                (self.args.image_folder / t / "progress").mkdir(exist_ok=True, parents=True)
            else:
                (self.args.image_folder / t).mkdir(exist_ok=True, parents=True)
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = os.path.join(self.args.exp, "vp", self.args.ckpt_load_name)
        
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"Model ckpt loaded from {ckpt}")
        model.to("cuda")
        model.train()
        
        
#         model.eval()

#         print('Run DDS.',
#             f'{self.args.T_sampling} sampling steps.',
#             f'Task: {self.args.deg}.'
#             )
#         self.dds(model)



def main():
    args, config = parse_args_and_config()
    diffusion_model = Diffusion(args, config)
    diffusion_model.train()
    
    
if __name__ == "__main__":
    print("running training")
    main()