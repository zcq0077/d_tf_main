#!/usr/bin/env python
# coding: utf-8
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

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4

    model.eval()
    l_best_errors, l_best_ades, l_best_fdes, l_traj_masks, l_masks = [], [], [], [], []
    plot_records = []
    n_eval_plots = getattr(cf, "n_test_plots", 8) if getattr(cf, "plot_test_trajectories", True) else 0
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            pred_len = max_seqlen - cf.init_seqlen
            error_ens = torch.zeros((batchsize, pred_len, cf.n_samples)).to(cf.device)
            pred_ens = torch.zeros((batchsize, max_seqlen, 4, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.generate(model,
                                          seqs_init,
                                          max_seqlen - init_seqlen,
                                          temperature=1.0,
                                          sample=True,
                                          sample_mode=cf.sample_mode,
                                          r_vicinity=cf.r_vicinity,
                                          top_k=cf.top_k,
                                          score_n_candidates=getattr(cf, "score_n_candidates", 64),
                                          score_prob_w=getattr(cf, "score_prob_w", 1.0),
                                          score_dist_w=getattr(cf, "score_dist_w", 0.25),
                                          score_sog_w=getattr(cf, "score_sog_w", 1.0),
                                          score_cog_w=getattr(cf, "score_cog_w", 1.0),
                                          score_heading_w=getattr(cf, "score_heading_w", 1.0),
                                          score_turn_w=getattr(cf, "score_turn_w", 0.5),
                                          score_dt_hours=getattr(cf, "score_dt_hours", 1.0 / 6.0),
                                          score_dist_scale_km=getattr(cf, "score_dist_scale_km", 2.0),
                                          score_resample_attempts=getattr(cf, "score_resample_attempts", 8),
                                          score_resample_max_penalty=getattr(cf, "score_resample_max_penalty", 1.2))
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
                pred_ens[:, :, :, i_sample] = preds[:, :max_seqlen, :]

            future_masks = masks[:, cf.init_seqlen:]
            valid_counts = future_masks.sum(dim=1).clamp_min(1)

            # Select one complete candidate trajectory by its whole-trajectory ADE.
            ade_ens = (error_ens * future_masks.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
            best_idx = ade_ens.argmin(dim=-1)
            gather_idx = best_idx.view(batchsize, 1, 1).expand(-1, pred_len, 1)
            best_errors = error_ens.gather(dim=-1, index=gather_idx).squeeze(-1)
            best_ade = ade_ens.gather(dim=-1, index=best_idx.unsqueeze(-1)).squeeze(-1)
            pred_gather_idx = best_idx.view(batchsize, 1, 1, 1).expand(-1, max_seqlen, 4, 1)
            best_preds = pred_ens.gather(dim=-1, index=pred_gather_idx).squeeze(-1)

            last_valid_idx = (future_masks.sum(dim=1).long() - 1).clamp_min(0)
            best_fde = best_errors.gather(dim=1, index=last_valid_idx.unsqueeze(-1)).squeeze(-1)
            valid_trajs = future_masks.sum(dim=1) > 0

            if len(plot_records) < n_eval_plots:
                remaining = n_eval_plots - len(plot_records)
                for i_plot in range(min(remaining, batchsize)):
                    true_len = int(masks[i_plot].sum().detach().cpu().item())
                    true_len = max(init_seqlen, min(true_len, max_seqlen))
                    plot_records.append({
                        "true": seqs[i_plot, :max_seqlen, :].detach().cpu().numpy(),
                        "pred": best_preds[i_plot, :max_seqlen, :].detach().cpu().numpy(),
                        "true_len": true_len,
                        "ade": best_ade[i_plot].detach().cpu().item(),
                        "fde": best_fde[i_plot].detach().cpu().item(),
                    })

            # Accumulation through batches
            l_best_errors.append(best_errors)
            l_best_ades.append(best_ade)
            l_best_fdes.append(best_fde)
            l_traj_masks.append(valid_trajs)
            l_masks.append(future_masks)

    m_masks = torch.cat(l_masks, dim=0)
    best_errors = torch.cat(l_best_errors, dim=0) * m_masks
    pred_errors = best_errors.sum(dim=0) / m_masks.sum(dim=0).clamp_min(1)
    pred_errors = pred_errors.detach().cpu().numpy()

    traj_masks = torch.cat(l_traj_masks, dim=0)
    best_ades = torch.cat(l_best_ades, dim=0)
    best_fdes = torch.cat(l_best_fdes, dim=0)
    valid_denom = traj_masks.float().sum().clamp_min(1)
    ade_km = (best_ades * traj_masks.float()).sum() / valid_denom
    fde_km = (best_fdes * traj_masks.float()).sum() / valid_denom
    ade_km = ade_km.detach().cpu().item()
    fde_km = fde_km.detach().cpu().item()

    metrics_path = os.path.join(cf.savedir, "best_trajectory_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"selection=lowest_full_trajectory_ADE_among_{cf.n_samples}_samples\n")
        f.write(f"ADE_km={ade_km:.6f}\n")
        f.write(f"FDE_km={fde_km:.6f}\n")
        for hour in (1, 2, 3):
            timestep = hour * 6
            if timestep < len(pred_errors):
                f.write(f"error_{hour}h_km={pred_errors[timestep]:.6f}\n")
    np.savetxt(os.path.join(cf.savedir, "best_trajectory_error_curve.csv"),
               np.column_stack((np.arange(len(pred_errors)) / 6, pred_errors)),
               delimiter=",",
               header="time_hours,error_km",
               comments="")
    logging.info(f"Best trajectory ADE/FDE: ADE {ade_km:.4f} km, FDE {fde_km:.4f} km.")
    print(f"Best trajectory ADE/FDE: ADE {ade_km:.4f} km, FDE {fde_km:.4f} km")

    if plot_records:
        plot_dir = os.path.join(cf.savedir, "test_trajectory_plots")
        os.makedirs(plot_dir, exist_ok=True)

        def to_lat_lon(x):
            lat = x[:, 0] * model.lat_range + model.lat_min
            lon = x[:, 1] * model.lon_range + model.lon_min
            return lat, lon

        def draw_trajectory(ax, record, title):
            true = record["true"]
            pred = record["pred"]
            true_len = record["true_len"]
            future_start = max(init_seqlen - 1, 0)
            true_lat, true_lon = to_lat_lon(true)
            pred_lat, pred_lon = to_lat_lon(pred)

            ax.plot(true_lon[:init_seqlen], true_lat[:init_seqlen],
                    color="#1f77b4", marker="o", markersize=3,
                    linewidth=2.0, label="History")
            ax.plot(true_lon[future_start:true_len], true_lat[future_start:true_len],
                    color="#2ca02c", marker="o", markersize=3,
                    linewidth=2.0, label="True future")
            ax.plot(pred_lon[future_start:true_len], pred_lat[future_start:true_len],
                    color="#d62728", linestyle="--", marker="x", markersize=4,
                    linewidth=2.0, label="Predicted future")
            ax.scatter(true_lon[0], true_lat[0], color="#1f77b4", s=35, zorder=5)
            ax.scatter(true_lon[init_seqlen - 1], true_lat[init_seqlen - 1],
                       color="#111111", s=35, zorder=5)
            ax.scatter(true_lon[true_len - 1], true_lat[true_len - 1],
                       color="#2ca02c", s=35, zorder=5)
            ax.scatter(pred_lon[true_len - 1], pred_lat[true_len - 1],
                       color="#d62728", s=35, zorder=5)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.legend(loc="best", fontsize=8)

        n_cols = 2
        n_rows = int(math.ceil(len(plot_records) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.8 * n_rows), dpi=150, squeeze=False)
        axes = axes.reshape(-1)
        for i_record, record in enumerate(plot_records):
            title = f"Test sample {i_record + 1} | ADE {record['ade']:.3f} km | FDE {record['fde']:.3f} km"
            draw_trajectory(axes[i_record], record, title)

            fig_single, ax_single = plt.subplots(figsize=(7, 5.5), dpi=180)
            draw_trajectory(ax_single, record, title)
            fig_single.tight_layout()
            fig_single.savefig(os.path.join(plot_dir, f"test_trajectory_{i_record + 1:02d}.png"),
                               bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes[len(plot_records):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(cf.savedir, "test_trajectory_examples.png"), bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved test trajectory plots to {plot_dir}.")
        print(f"Saved test trajectory plots to {plot_dir}")

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")

    # Yeah, done!!!
