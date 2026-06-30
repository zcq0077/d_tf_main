# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration flags to run the main script.
"""

import os
import pickle
import torch


class Config():
    retrain = True
    tb_log = False
    device = torch.device("cuda:0")
#     device = torch.device("cpu")
    
    max_epochs = 20
    batch_size = 32
    n_samples = 16
    
    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36
    
    dataset_name = "ct_dma"

    if dataset_name == "ct_dma": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon    
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    
    #===========================================================================
    # Model and sampling flags
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1", 
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity", "pos_resample", "pos_score" or "velo"
    top_k = 10 # int or None 
    r_vicinity = 40 # int
    plot_test_trajectories = True
    n_test_plots = 8

    # Confidence- and Constraint-Aware Scheduled Sampling flags
    #===================================================
    # This auxiliary training loss gradually feeds back model predictions when
    # they are confident and navigation-consistent, reducing exposure bias.
    use_ccass = False
    ccass_loss_w = 0.20
    ccass_start_epoch = 2
    ccass_ramp_epochs = 8
    ccass_max_pred_prob = 0.50
    ccass_max_steps = 24
    ccass_temperature = 1.0
    ccass_sample = True
    ccass_use_vicinity = True
    ccass_top_k = top_k
    ccass_conf_w = 4.0
    ccass_penalty_w = 2.0
    ccass_conf_center = 0.50
    ccass_penalty_center = 1.20

    # Point-wise candidate scoring flags
    #===================================================
    # During autoregressive sampling, resample a next point only when it has
    # poor motion-consistency according to the scoring terms below.
    score_n_candidates = 64
    score_prob_w = 1.0
    score_dist_w = 0.20
    score_sog_w = 0.60
    score_cog_w = 0.60
    score_heading_w = 0.60
    score_turn_w = 0.30
    score_dt_hours = 1.0 / 6.0
    score_dist_scale_km = 2.0
    score_resample_attempts = 8
    score_resample_max_penalty = 1.2
    
    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # Data flags
    #===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
    
    
    # model parameters
    #===================================================
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 4 # for DataLoader
    
    score_tag = f"-R{score_resample_attempts}" if sample_mode in ("pos_resample", "pos_vicinity_resample") else ""
    score_tag = f"-C{score_n_candidates}" if sample_mode in ("pos_score", "pos_vicinity_score") else score_tag
    ccass_tag = "-ccass" if use_ccass else ""
    filename = f"{dataset_name}-{mode}-{sample_mode}{score_tag}{ccass_tag}"\
        + f"-bs{batch_size}-lr{learning_rate}"\
        + f"-seq{init_seqlen}-{max_seqlen}"
    savedir = "./results/"+filename+"/"
    
    ckpt_path = os.path.join(savedir,"model.pt")   
