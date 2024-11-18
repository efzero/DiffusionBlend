import os
import json
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np
from datetime import datetime

from pathlib import Path
from guided_diffusion.CTDataset import *
from guided_diffusion.train_util import *
from guided_diffusion.script_util import create_model, create_gaussian_diffusion
from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path
from physics.ct import CT
from physics.mri import SinglecoilMRI_comp, MulticoilMRI
from utils import CG, clear, get_mask, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from functools import partial

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
        "--resume_checkpoint", type=bool, default=False, help="resume training from a previous checkpoint"
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
        config_dict["class_cond"] = True
        config_dict["use_spacecode"] = False
        print(config_dict)
        model = create_model(**config_dict)
        print(model.use_spacecode, "using spacecode")
        ckpt = os.path.join(self.args.exp, "vp", self.args.ckpt_load_name)
        
        pretrainsteps = 0
        if self.args.resume_checkpoint is True:
            print("resuming training")
            ckpt = "/nfs/turbo/coe-liyues/bowenbw/3DCT/checkpoints/triplane3D_finetune_452024_iter50099_cond.ckpt"
            pretrainsteps = 50000
        else:
            ckpt = "/nfs/turbo/coe-liyues/bowenbw/3DCT/checkpoints/256x256_diffusion_uncond.pt"
        
        loaded = torch.load(ckpt, map_location=self.device)
        for key in loaded['state_dict']:
            print(key, "printed")
        model.load_state_dict(torch.load(ckpt, map_location=self.device)['state_dict'])
        
        print(f"Model ckpt loaded from {ckpt}")
        model.to("cuda")
        model.train()
        
        diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
        print(diffusion.training_losses, "training loss")
        
        lr = 1e-5
        params = list(model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)

        #############################testing feed numpy matrix into the training script########################
        
        #########################use the given trainer in improved_diffusion#########################
        
        """use the given trainer in improved_diffusion
        ds = CTDataset()
        params = {'batch_size': 4}
        training_generator = torch.utils.data.DataLoader(ds, **params)
        def load_data(loader):
            while True:
                yield from loader
        
        data = load_data(training_generator)
        
        TrainLoop2(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=4,
            microbatch=-1,
            lr=3e-4,
            ema_rate="0.9999",
            log_interval=10,
            save_interval=2500,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler="uniform",
            weight_decay=0.0,
            lr_anneal_steps=0,
        ).run_loop()
        """
    ################################train manually####################################
#         files = glob('/nfs/turbo/coe-liyues/bowenbw/3DCT/AAPM_fusion_training/*')
        files = glob("/nfs/turbo/coe-liyues/bowenbw/3DCT/AAPM_fusion_training_cond/*")
        files2 = glob("/nfs/turbo/coe-liyues/bowenbw/3DCT/slice_fusion_training/*")
        
            
        for m in range(25100):

            x_train = np.zeros((4, 3, 256, 256))
            y = torch.randint(0, 3, (4,))
            for l in range(4):
                luck = np.random.randint(0, 2)
                if luck == 0:
                    filename = np.random.choice(files)
                    x_raw = np.transpose(np.load(filename), (2,0,1))[0:3]
                    x_raw = np.clip(x_raw*2-1, -1, 1)
                    x_train[l] = x_raw.copy()
                    y_val = int((filename.split(".")[0]).split("_")[-1])
                    y[l] = y_val
                    print(y_val)
                else:
                    filename = np.random.choice(files2)
                    x_raw = np.transpose(np.load(filename), (2,0,1))[0:3]
                    x_raw = np.clip(x_raw*2-1, -1, 1)
                    x_train[l] = x_raw.copy()
                    y[l] = 3
            x_orig = torch.from_numpy(x_train).to("cuda").to(torch.float)
            i = torch.randint(0, 1000, (4,))
            t = i.to("cuda").long()
            y = y.to("cuda").long()
            
            
            model_kwargs = {}
            model_kwargs["y"] = y
            
#             if m % 1000 == 0:
#                 x_sample = diffusion.ddim_sample_loop_progressive(model, (4,3,256,256), task = "None", progress= True, model_kwargs = model_kwargs)
#                 np.save("/nfs/turbo/coe-liyues/bowenbw/3DCT/x_sample_ddim_iter" + str(m + pretrainsteps) + "_finetune_452024_cond.npy", x_sample.detach().cpu().numpy())
            
            loss = diffusion.training_losses(model, x_orig, t, model_kwargs=model_kwargs)["loss"]
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
#             print(x_orig.dtype)
#             loss = diffusion.training_losses(model, x_orig, t)["loss"]
#             loss= loss.mean()
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
            t0 = datetime.now()
            if m % 20 == 0:
                print(loss.item(), "loss", " at ", m, "th iteration")
                t1 = datetime.now()
                print(t1 - t0, "time elapsed")
                t0 = t1
#         ####################################################################################################################
#             if m % 5000 == 99:
#                 torch.save({'iterations':m,'state_dict': model.state_dict()}, "/nfs/turbo/coe-liyues/bowenbw/3DCT/checkpoints/triplane3D_finetune_452024_iter" + str(m + pretrainsteps) + "_cond.ckpt")
                   
        ####################################################################################                             
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