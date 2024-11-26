import os
import logging
import time
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict, create_gaussian_diffusion
from guided_diffusion.utils import get_alpha_schedule
import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from pathlib import Path

from physics.ct import CT
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, get_beta_schedule


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


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
        print(self.config)
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

    def sample(self):
        config_dict = vars(self.config.model)
        config_dict['use_spacecode'] = False
        config_dict["class_cond"] = True
        model = create_model(**config_dict)
        ckpt = "/nfs/turbo/coe-liyues/bowenbw/3DCT/checkpoints/LIDC_triplane3D_finetunelarge_4232024_iter62099_cond.ckpt"
        
        model.load_state_dict(torch.load(ckpt, map_location=self.device)["state_dict"])
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)

        print('Run 3D DDS + DiffusionMBIR.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: {self.args.deg}.'
            )
        self.dds3d(model)
        
        
    def blendscore(self, xt, model,t, start_ind = None, start_head = 0, num_batches = 180):
        model_kwargs = {}
        y = torch.ones(1) * 1
        y=y.to(xt.device).to(torch.long)
        model_kwargs["y"] = y
        
        et = torch.zeros((1, num_batches * 3, 256, 256)).to(xt.device).to(torch.float32)
        xt = torch.reshape(xt, (1, num_batches * 3, 256, 256))
        et[:,:3,:,:] = model(xt[:,:3,:,:], t, **model_kwargs)[:,:3]
        et[:,xt.shape[1]-3:, :,:] = model(xt[:,xt.shape[1]-3:,:,:], t, **model_kwargs)[:,:3]
        
        if start_ind is None:
            start_ind = np.random.randint(start_head,3)
        for j in range(start_ind, xt.shape[1]-2, 3):
            #####randomly select instead of summing
            et_sing = model(xt[:,j:(j+3),:,:], t, **model_kwargs)[:,:3] #####1 x 3 x 256 x 256
            et[:,j:(j+3), :,:] = et_sing
        return et
    
    def vps_blend(self, xt, model,t, start_ind = None, start_head = 0, num_batches = 180):
        model_kwargs = {}
        y = torch.ones(1) * 3
        y=y.to(xt.device).to(torch.long)
        model_kwargs["y"] = y
        
        et = torch.zeros((1, num_batches * 3, 256, 256)).to(xt.device).to(torch.float32)
        
        xt = torch.reshape(xt, (1, num_batches * 3, 256, 256))
#         147258369
        for i in range(0,xt.shape[1], 9):
            for m in range(3):
                et[:,[i+m,i+m+3, i+m+6],:,:] = model(xt[:,[i+m,i+m+3, i+m+6],:,:], t, **model_kwargs)[:,:3]
                
        return et
                  
            
    def dds3d(self, model):
        args, config = self.args, self.config
        print(f"Dataset path: {self.args.dataset_path}")
        root = Path(self.args.dataset_path)
        
        # parameters to be moved to args
        Nview = self.args.Nview
        rho = 0.001 ###9:23pm 3/3
        lamb = 0.05/1000
        n_ADMM = 1
        n_CG = self.args.CG_iter
        print(n_CG)

        blend= True ####test again 4/7

        time_travel = False ### 4:20 3/10
        
        vps = False ###debugging

        ddimsteps= 200  ####11:17pm 3/20
        
        ####1:21 try hard consistency for only one time
        
#         print(rho, lamb, "admm params")
#         print("blending", blend)
        
        # Specify save directory for saving generated samples
        save_root = Path(self.args.image_folder)
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['vol', 'input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        fname_list = os.listdir("/nfs/turbo/coe-liyues/bowenbw/3DCT/benchmark/validation_LIDC")
        root = "/nfs/turbo/coe-liyues/bowenbw/3DCT/benchmark/validation_LIDC"
        
        fname_list.sort()
        num_batches = 87 ######4/22 2:16pm 
        fname_list = fname_list[:(3 * num_batches)] #####4/22 2:22pm

        ######################################################################################################
    
        print(fname_list)
        all_img = []
        batch_size = 3
        print("Loading all data")
        if time_travel:
            tot_iters = 2
        else:
            tot_iters = 1
        for fname in fname_list:
            just_name = fname.split('.')[0]
            img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
            h, w = img.shape
            img = img.view(1, 1, h, w)
            all_img.append(img)
        all_img = torch.cat(all_img, dim=0) 
        x_orig = all_img
        print(f"Data loaded shape : {all_img.shape}")
        x_orig = x_orig.to(torch.float32)
        print("Data type is :", x_orig.dtype)
        img_shape = (x_orig.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)
        A = lambda z: A_funcs.A(z)
        Ap = lambda z: A_funcs.A_dagger(z)
        def Acg_TV(x):
            return A_funcs.AT(A_funcs.A(x)) + rho * _DzT(_Dz(x))
        def ADMM(x, ATy, n_ADMM=n_ADMM):
            nonlocal del_z, udel_z
            for _ in range(n_ADMM):
                bcg_TV = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
                x = CG(Acg_TV, bcg_TV, x, n_inner=n_CG)
                del_z = shrink(_Dz(x) + udel_z, lamb / rho)
                udel_z = _Dz(x) - del_z + udel_z
            return x
        del_z = torch.zeros(img_shape, device=self.device)
        udel_z = torch.zeros(img_shape, device=self.device)
        x_orig = x_orig.to(self.device) ######n x 1 x 256 x 256
        print(x_orig.min(), x_orig.max(), "xorig")
        y = A(x_orig)
        print(y.shape, "projection shape")
        Apy = Ap(y)
        print(Apy.shape, "Apy backprojection shape")
        ATy = A_funcs.AT(y)
        ##########################original####################################################
        x = torch.randn(num_batches, 3, 256, 256, device = self.device) ####initial noise
        ##################################################################################################

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
        xt = None
        with torch.no_grad():
            skip = config.diffusion.num_diffusion_timesteps//ddimsteps
            n = x.size(0)
            x0_preds = []
            xt = x ###20 x 3 x 256 x 256
            
            # generate time schedule
            times = range(0, 1000, skip)  #########0, 1, 2, ....
            times_next = [-1] + list(times[:-1])
            times_pair = zip(reversed(times), reversed(times_next))
            
            if blend:
                n = 1
            else:
                n = 1
                
            ct = 0
            ###################################start reverse sampling############################################
            for i, j in tqdm.tqdm(times_pair, total=len(times)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                ########if time travel do two passes, otherwise do one pass#########
                travels = tot_iters
                ct += 1
                
                for zhoumu in range(travels):
                    print("zhoumu: ", zhoumu)
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    bt = torch.index_select(self.betas,0,t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    at = at[0,0,0,0]
                    at_next = at_next[0,0,0,0]
                #################################reverse with consistency########################################
                    et_agg = list() ###initialize a list of scores
                    ###########################################ADJ slices#############################################
                    if ct % 2 == 0:
                        if vps:
                            for M in range(1): ####number of VPS iterations
                                noise = torch.randn_like(xt)
                                ####################added by bowen 3/24/2024####################
                                et = self.vps_blend(xt, model, t, num_batches = num_batches)
                                ####################################################################
                                et = torch.reshape(et, (num_batches, 3, 256, 256))
                                lam_ = vps_scale
                                xt = xt - lam_ * (1 - at).sqrt() * et
                                xt = xt + ((lam_ * (2-lam_))*(1-at)).sqrt() * noise * 1
                        if blend:
                            et = self.blendscore(xt, model,t,num_batches = num_batches)
                        else:
                            y = torch.ones(1) * 1
                            y=y.to(xt.device).to(torch.long)
                            model_kwargs = {}
                            model_kwargs["y"] = y
                            for j in range(xt.shape[0]//1):
                                et_sing = model(xt[j*1:(j+1)*1], t, **model_kwargs) ####4 x 6 x 256 x 256
                                et_agg.append(et_sing)
                            et = torch.cat(et_agg, dim=0) ####20 x 6 x 256 x 256
                            et = et[:, :3] ####20 x 3 x 256 x 256
                        ###reshape xt and et
                        et_ = torch.reshape(et, ((num_batches * 3), 1, 256, 256))
                    #######################################SLICE JUMP#############################################
                    if ct % 2 == 1: ###147258369   4710 
                        print(ct, "changing et to slice jumping")
                        et_ = self.vps_blend(xt, model, t, num_batches = num_batches)
                        et_ = torch.reshape(et_, ((num_batches * 3), 1, 256, 256))
                    xt_ = torch.reshape(xt, ((num_batches * 3), 1, 256, 256))
                    x0_t = (xt_ - et_ * (1 - at).sqrt()) / at.sqrt()  ###60 x 1 x 256 x 256 scale [-1, 1]
#                     x0_t = torch.clip(x0_t, -1, 1) ####clip to [-1, 1]
                    x0_t = (x0_t +1)/2 ###rescale to [0, 1]
                    x0_t_hat = None
                    eta = self.args.eta
                    if zhoumu == 0:
                        x0_t_hat = ADMM(x0_t, ATy, n_ADMM=n_ADMM) ######[0,1]
#                         x0_t_hat = torch.clip(x0_t_hat, 0, 1)
                        x0_t_hat = x0_t_hat * 2 - 1 #######rescale back to [-1, 1]
                    else:
                        x0_t_hat = x0_t * 2 - 1 #######rescale back to [-1, 1]

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                    if j != 0:
                        xt_ = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et_
                    else:
                        xt_ = x0_t_hat
                    xt = torch.reshape(xt_, (num_batches, 3, 256, 256)) ####reshape back
                    
            if self.args.deg == "SV-CT":
                np.save("x_sample_ddim" + str(ddimsteps) + "_iter62000large_reconstructionL67_blend3_rho" + str(rho) + "ttnew" + str(tot_iters) + "_full_view4_424_jump_LIDC_ftldct_half_retest.npy", xt.detach().cpu().numpy()) #############imnet pretrain 40000 iters
                
                
            if self.args.deg == "LA-CT":
                np.save("x_sample_ddim" + str(ddimsteps) + "_iter62000large_lactlidc_blend3_full_view90.npy", xt.detach().cpu().numpy())
                
