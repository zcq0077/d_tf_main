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

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

from trAISformer import TB_LOG

logger = logging.getLogger(__name__)


def _safe_probs_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    uniform = torch.ones_like(probs) / probs.size(-1)
    return torch.where(denom > 0, probs / denom.clamp_min(1e-12), uniform)


def _candidate_indexes(probs, n_candidates, do_sample):
    argmax_ix = torch.argmax(probs, dim=-1, keepdim=True)
    if do_sample:
        cand_ix = torch.multinomial(probs, num_samples=n_candidates, replacement=True)
    else:
        k = min(n_candidates, probs.size(-1))
        cand_ix = torch.topk(probs, k=k, dim=-1).indices
        if k < n_candidates:
            pad = cand_ix[:, -1:].repeat(1, n_candidates - k)
            cand_ix = torch.cat((cand_ix, pad), dim=-1)
    return torch.cat((argmax_ix, cand_ix), dim=-1)


def _circular_diff(a, b):
    diff = torch.abs(a - b)
    return torch.minimum(diff, 1.0 - diff)


def _norm_to_degrees(model, x):
    lat_range = getattr(model, "lat_range", 1.0)
    lon_range = getattr(model, "lon_range", 1.0)
    lat_min = getattr(model, "lat_min", 0.0)
    lon_min = getattr(model, "lon_min", 0.0)
    lat = x[..., 0] * lat_range + lat_min
    lon = x[..., 1] * lon_range + lon_min
    return lat, lon


def _haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1 = lat1_deg * math.pi / 180.0
    lon1 = lon1_deg * math.pi / 180.0
    lat2 = lat2_deg * math.pi / 180.0
    lon2 = lon2_deg * math.pi / 180.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a.clamp_min(0.0)), torch.sqrt((1 - a).clamp_min(0.0)))
    return 6371.0 * c


def _bearing_norm(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1 = lat1_deg * math.pi / 180.0
    lon1 = lon1_deg * math.pi / 180.0
    lat2 = lat2_deg * math.pi / 180.0
    lon2 = lon2_deg * math.pi / 180.0
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    bearing = torch.atan2(y, x) / (2 * math.pi)
    return torch.remainder(bearing + 1.0, 1.0)


def _score_next_point_candidates(model,
                                 seqs_cond,
                                 lat_logits,
                                 lon_logits,
                                 sog_logits,
                                 cog_logits,
                                 sample,
                                 score_n_candidates=64,
                                 score_prob_w=1.0,
                                 score_dist_w=0.25,
                                 score_sog_w=1.0,
                                 score_cog_w=1.0,
                                 score_heading_w=1.0,
                                 score_turn_w=0.5,
                                 score_dt_hours=1.0 / 6.0,
                                 score_dist_scale_km=2.0):
    lat_probs = _safe_probs_from_logits(lat_logits)
    lon_probs = _safe_probs_from_logits(lon_logits)
    sog_probs = _safe_probs_from_logits(sog_logits)
    cog_probs = _safe_probs_from_logits(cog_logits)

    lat_log_probs = torch.log(lat_probs.clamp_min(1e-12))
    lon_log_probs = torch.log(lon_probs.clamp_min(1e-12))
    sog_log_probs = torch.log(sog_probs.clamp_min(1e-12))
    cog_log_probs = torch.log(cog_probs.clamp_min(1e-12))

    lat_ix = _candidate_indexes(lat_probs, score_n_candidates, sample)
    lon_ix = _candidate_indexes(lon_probs, score_n_candidates, sample)
    sog_ix = _candidate_indexes(sog_probs, score_n_candidates, sample)
    cog_ix = _candidate_indexes(cog_probs, score_n_candidates, sample)
    cand_ix = torch.stack((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)

    logp = torch.gather(lat_log_probs, 1, lat_ix) \
        + torch.gather(lon_log_probs, 1, lon_ix) \
        + torch.gather(sog_log_probs, 1, sog_ix) \
        + torch.gather(cog_log_probs, 1, cog_ix)

    cand_x = (cand_ix.float() + 0.5) / model.att_sizes
    prev_x = seqs_cond[:, -1:, :]

    prev_lat, prev_lon = _norm_to_degrees(model, prev_x)
    cand_lat, cand_lon = _norm_to_degrees(model, cand_x)
    dist_km = _haversine_km(prev_lat, prev_lon, cand_lat, cand_lon)

    sog_range = getattr(model, "sog_range", 30.0)
    expected_km = ((prev_x[..., 2] + cand_x[..., 2]) * 0.5 * sog_range) * 1.852 * score_dt_hours
    dist_penalty = torch.abs(dist_km - expected_km) / score_dist_scale_km
    sog_penalty = torch.abs(cand_x[..., 2] - prev_x[..., 2])
    cog_penalty = _circular_diff(cand_x[..., 3], prev_x[..., 3])

    bearing = _bearing_norm(prev_lat, prev_lon, cand_lat, cand_lon)
    heading_penalty = _circular_diff(bearing, cand_x[..., 3])
    heading_penalty = torch.where(dist_km > 0.05, heading_penalty, torch.zeros_like(heading_penalty))

    if seqs_cond.size(1) > 1:
        prev2_x = seqs_cond[:, -2:-1, :]
        prev2_lat, prev2_lon = _norm_to_degrees(model, prev2_x)
        prev_dist_km = _haversine_km(prev2_lat, prev2_lon, prev_lat, prev_lon)
        prev_bearing = _bearing_norm(prev2_lat, prev2_lon, prev_lat, prev_lon)
        turn_penalty = _circular_diff(prev_bearing, bearing)
        moving = (prev_dist_km > 0.05) & (dist_km > 0.05)
        turn_penalty = torch.where(moving, turn_penalty, torch.zeros_like(turn_penalty))
    else:
        turn_penalty = torch.zeros_like(dist_penalty)

    score = score_prob_w * logp \
        - score_dist_w * dist_penalty \
        - score_sog_w * sog_penalty \
        - score_cog_w * cog_penalty \
        - score_heading_w * heading_penalty \
        - score_turn_w * turn_penalty

    best = torch.argmax(score, dim=1)
    return cand_ix[torch.arange(cand_ix.size(0), device=cand_ix.device), best, :]


def _score_fixed_next_point_candidates(model,
                                       seqs_cond,
                                       cand_ix,
                                       logp,
                                       score_prob_w=1.0,
                                       score_dist_w=0.25,
                                       score_sog_w=1.0,
                                       score_cog_w=1.0,
                                       score_heading_w=1.0,
                                       score_turn_w=0.5,
                                       score_dt_hours=1.0 / 6.0,
                                       score_dist_scale_km=2.0):
    cand_x = (cand_ix.float() + 0.5) / model.att_sizes
    prev_x = seqs_cond[:, -1:, :]

    prev_lat, prev_lon = _norm_to_degrees(model, prev_x)
    cand_lat, cand_lon = _norm_to_degrees(model, cand_x)
    dist_km = _haversine_km(prev_lat, prev_lon, cand_lat, cand_lon)

    sog_range = getattr(model, "sog_range", 30.0)
    expected_km = ((prev_x[..., 2] + cand_x[..., 2]) * 0.5 * sog_range) * 1.852 * score_dt_hours
    dist_penalty = torch.abs(dist_km - expected_km) / score_dist_scale_km
    sog_penalty = torch.abs(cand_x[..., 2] - prev_x[..., 2])
    cog_penalty = _circular_diff(cand_x[..., 3], prev_x[..., 3])

    bearing = _bearing_norm(prev_lat, prev_lon, cand_lat, cand_lon)
    heading_penalty = _circular_diff(bearing, cand_x[..., 3])
    heading_penalty = torch.where(dist_km > 0.05, heading_penalty, torch.zeros_like(heading_penalty))

    if seqs_cond.size(1) > 1:
        prev2_x = seqs_cond[:, -2:-1, :]
        prev2_lat, prev2_lon = _norm_to_degrees(model, prev2_x)
        prev_dist_km = _haversine_km(prev2_lat, prev2_lon, prev_lat, prev_lon)
        prev_bearing = _bearing_norm(prev2_lat, prev2_lon, prev_lat, prev_lon)
        turn_penalty = _circular_diff(prev_bearing, bearing)
        moving = (prev_dist_km > 0.05) & (dist_km > 0.05)
        turn_penalty = torch.where(moving, turn_penalty, torch.zeros_like(turn_penalty))
    else:
        turn_penalty = torch.zeros_like(dist_penalty)

    motion_penalty = score_dist_w * dist_penalty \
        + score_sog_w * sog_penalty \
        + score_cog_w * cog_penalty \
        + score_heading_w * heading_penalty \
        + score_turn_w * turn_penalty
    score = score_prob_w * logp - motion_penalty
    return score, motion_penalty


def _four_hot_entropy_confidence(lat_probs, lon_probs, sog_probs, cog_probs):
    def attr_conf(probs):
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        max_entropy = math.log(probs.size(-1))
        return 1.0 - entropy / max_entropy

    return (
        attr_conf(lat_probs)
        + attr_conf(lon_probs)
        + attr_conf(sog_probs)
        + attr_conf(cog_probs)
    ) / 4.0


def _four_hot_ce_loss_from_logits(model, logits, targets):
    lat_logits, lon_logits, sog_logits, cog_logits = model._split_logits(logits)
    batchsize, seqlen, _ = targets.size()

    lat_loss = F.cross_entropy(
        lat_logits.reshape(-1, model.lat_size),
        targets[:, :, 0].reshape(-1),
        reduction="none",
    ).view(batchsize, seqlen)
    lon_loss = F.cross_entropy(
        lon_logits.reshape(-1, model.lon_size),
        targets[:, :, 1].reshape(-1),
        reduction="none",
    ).view(batchsize, seqlen)
    sog_loss = F.cross_entropy(
        sog_logits.reshape(-1, model.sog_size),
        targets[:, :, 2].reshape(-1),
        reduction="none",
    ).view(batchsize, seqlen)
    cog_loss = F.cross_entropy(
        cog_logits.reshape(-1, model.cog_size),
        targets[:, :, 3].reshape(-1),
        reduction="none",
    ).view(batchsize, seqlen)
    return lat_loss + lon_loss + sog_loss + cog_loss


def _ccass_epoch_scale(config, epoch):
    start_epoch = int(getattr(config, "ccass_start_epoch", 0))
    if epoch < start_epoch:
        return 0.0

    ramp_epochs = max(1, int(getattr(config, "ccass_ramp_epochs", 1)))
    return min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))


def _confidence_constraint_scheduled_sampling_loss(model,
                                                   seqs,
                                                   masks,
                                                   config,
                                                   epoch):
    if not getattr(config, "use_ccass", False):
        return None, None

    epoch_scale = _ccass_epoch_scale(config, epoch)
    if epoch_scale <= 0:
        return None, None

    init_seqlen = int(getattr(config, "init_seqlen", 1))
    max_steps = int(getattr(config, "ccass_max_steps", 0))
    if max_steps <= 0 or seqs.size(1) <= init_seqlen:
        return None, None

    max_seqlen = model.get_max_seqlen()
    rollout_end = min(seqs.size(1) - 1, init_seqlen + max_steps, max_seqlen)
    if rollout_end <= init_seqlen:
        return None, None

    with torch.no_grad():
        true_idxs, _ = model.to_indexes(seqs, mode=model.partition_mode)
        mixed = seqs.detach().clone()
        prefix = mixed[:, :init_seqlen, :].clone()

        pred_prob_sum = 0.0
        use_pred_sum = 0.0
        valid_sum = 0.0
        conf_sum = 0.0
        penalty_sum = 0.0

        for t in range(init_seqlen, rollout_end):
            seqs_cond = prefix if prefix.size(1) <= max_seqlen else prefix[:, -max_seqlen:]
            logits, _ = model(seqs_cond)
            logits = logits[:, -1, :] / getattr(config, "ccass_temperature", 1.0)
            lat_logits, lon_logits, sog_logits, cog_logits = model._split_logits(logits)

            if getattr(config, "ccass_use_vicinity", True):
                _, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :], mode=model.partition_mode)
                lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
                lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, getattr(config, "r_vicinity", 20))
                lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, getattr(config, "r_vicinity", 20))

            top_k = getattr(config, "ccass_top_k", None)
            if top_k is not None:
                lat_logits = utils.top_k_logits(lat_logits, top_k)
                lon_logits = utils.top_k_logits(lon_logits, top_k)
                sog_logits = utils.top_k_logits(sog_logits, top_k)
                cog_logits = utils.top_k_logits(cog_logits, top_k)

            lat_probs = _safe_probs_from_logits(lat_logits)
            lon_probs = _safe_probs_from_logits(lon_logits)
            sog_probs = _safe_probs_from_logits(sog_logits)
            cog_probs = _safe_probs_from_logits(cog_logits)

            do_sample = getattr(config, "ccass_sample", True)
            if do_sample:
                lat_ix = torch.multinomial(lat_probs, num_samples=1)
                lon_ix = torch.multinomial(lon_probs, num_samples=1)
                sog_ix = torch.multinomial(sog_probs, num_samples=1)
                cog_ix = torch.multinomial(cog_probs, num_samples=1)
            else:
                lat_ix = torch.argmax(lat_probs, dim=-1, keepdim=True)
                lon_ix = torch.argmax(lon_probs, dim=-1, keepdim=True)
                sog_ix = torch.argmax(sog_probs, dim=-1, keepdim=True)
                cog_ix = torch.argmax(cog_probs, dim=-1, keepdim=True)

            cand_ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
            logp = torch.log(torch.gather(lat_probs, 1, lat_ix).clamp_min(1e-12)) \
                + torch.log(torch.gather(lon_probs, 1, lon_ix).clamp_min(1e-12)) \
                + torch.log(torch.gather(sog_probs, 1, sog_ix).clamp_min(1e-12)) \
                + torch.log(torch.gather(cog_probs, 1, cog_ix).clamp_min(1e-12))

            _, motion_penalty = _score_fixed_next_point_candidates(
                model,
                seqs_cond,
                cand_ix.unsqueeze(1),
                logp,
                score_prob_w=getattr(config, "score_prob_w", 1.0),
                score_dist_w=getattr(config, "score_dist_w", 0.25),
                score_sog_w=getattr(config, "score_sog_w", 1.0),
                score_cog_w=getattr(config, "score_cog_w", 1.0),
                score_heading_w=getattr(config, "score_heading_w", 1.0),
                score_turn_w=getattr(config, "score_turn_w", 0.5),
                score_dt_hours=getattr(config, "score_dt_hours", 1.0 / 6.0),
                score_dist_scale_km=getattr(config, "score_dist_scale_km", 2.0),
            )
            motion_penalty = motion_penalty.squeeze(1)
            confidence = _four_hot_entropy_confidence(lat_probs, lon_probs, sog_probs, cog_probs)

            gate_logits = getattr(config, "ccass_conf_w", 4.0) \
                * (confidence - getattr(config, "ccass_conf_center", 0.5)) \
                - getattr(config, "ccass_penalty_w", 2.0) \
                * (motion_penalty - getattr(config, "ccass_penalty_center", 1.2))
            pred_prob = epoch_scale * getattr(config, "ccass_max_pred_prob", 0.5) * torch.sigmoid(gate_logits)
            valid = masks[:, t] > 0
            use_pred = (torch.rand_like(pred_prob) < pred_prob) & valid

            pred_x = ((cand_ix.float() + 0.5) / model.att_sizes).detach()
            next_x = torch.where(use_pred.unsqueeze(-1), pred_x, seqs[:, t, :])
            mixed[:, t, :] = next_x
            prefix = torch.cat((prefix, next_x.unsqueeze(1)), dim=1)

            valid_f = valid.float()
            valid_sum += valid_f.sum().item()
            pred_prob_sum += (pred_prob * valid_f).sum().item()
            use_pred_sum += (use_pred.float() * valid_f).sum().item()
            conf_sum += (confidence * valid_f).sum().item()
            penalty_sum += (motion_penalty * valid_f).sum().item()

    inputs = mixed[:, :-1, :]
    targets = true_idxs[:, 1:, :].contiguous()
    input_idxs, _ = model.to_indexes(inputs, mode=model.partition_mode)
    logits = model.head(model._encode_indexes(input_idxs))
    token_loss = _four_hot_ce_loss_from_logits(model, logits, targets)

    loss_mask = masks[:, 1:].float()
    scheduled_mask = torch.zeros_like(loss_mask)
    scheduled_mask[:, init_seqlen - 1:rollout_end] = 1.0
    loss_mask = loss_mask * scheduled_mask
    denom = loss_mask.sum()
    if denom.item() <= 0:
        return None, None

    loss = (token_loss * loss_mask).sum() / denom
    stats = {
        "pred_prob": pred_prob_sum / max(valid_sum, 1.0),
        "use_pred_rate": use_pred_sum / max(valid_sum, 1.0),
        "confidence": conf_sum / max(valid_sum, 1.0),
        "motion_penalty": penalty_sum / max(valid_sum, 1.0),
    }
    return loss, stats


def _sample_next_point_with_resampling(model,
                                       seqs_cond,
                                       lat_logits,
                                       lon_logits,
                                       sog_logits,
                                       cog_logits,
                                       sample,
                                       score_resample_attempts=8,
                                       score_resample_max_penalty=1.2,
                                       score_prob_w=1.0,
                                       score_dist_w=0.25,
                                       score_sog_w=1.0,
                                       score_cog_w=1.0,
                                       score_heading_w=1.0,
                                       score_turn_w=0.5,
                                       score_dt_hours=1.0 / 6.0,
                                       score_dist_scale_km=2.0):
    lat_probs = _safe_probs_from_logits(lat_logits)
    lon_probs = _safe_probs_from_logits(lon_logits)
    sog_probs = _safe_probs_from_logits(sog_logits)
    cog_probs = _safe_probs_from_logits(cog_logits)

    lat_log_probs = torch.log(lat_probs.clamp_min(1e-12))
    lon_log_probs = torch.log(lon_probs.clamp_min(1e-12))
    sog_log_probs = torch.log(sog_probs.clamp_min(1e-12))
    cog_log_probs = torch.log(cog_probs.clamp_min(1e-12))

    batchsize = lat_probs.size(0)
    device = lat_probs.device
    attempts = max(1, int(score_resample_attempts))

    accepted = torch.zeros(batchsize, dtype=torch.bool, device=device)
    selected_ix = torch.zeros(batchsize, 4, dtype=torch.long, device=device)
    best_ix = torch.zeros(batchsize, 4, dtype=torch.long, device=device)
    best_score = torch.full((batchsize,), -float("Inf"), device=device)

    for _ in range(attempts):
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            lat_ix = torch.argmax(lat_probs, dim=-1, keepdim=True)
            lon_ix = torch.argmax(lon_probs, dim=-1, keepdim=True)
            sog_ix = torch.argmax(sog_probs, dim=-1, keepdim=True)
            cog_ix = torch.argmax(cog_probs, dim=-1, keepdim=True)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        logp = torch.gather(lat_log_probs, 1, lat_ix) \
            + torch.gather(lon_log_probs, 1, lon_ix) \
            + torch.gather(sog_log_probs, 1, sog_ix) \
            + torch.gather(cog_log_probs, 1, cog_ix)
        score, motion_penalty = _score_fixed_next_point_candidates(
            model,
            seqs_cond,
            ix.unsqueeze(1),
            logp,
            score_prob_w=score_prob_w,
            score_dist_w=score_dist_w,
            score_sog_w=score_sog_w,
            score_cog_w=score_cog_w,
            score_heading_w=score_heading_w,
            score_turn_w=score_turn_w,
            score_dt_hours=score_dt_hours,
            score_dist_scale_km=score_dist_scale_km,
        )
        score = score.squeeze(1)
        motion_penalty = motion_penalty.squeeze(1)

        better = score > best_score
        best_score = torch.where(better, score, best_score)
        best_ix = torch.where(better.unsqueeze(-1), ix, best_ix)

        good = motion_penalty <= score_resample_max_penalty
        newly_accepted = good & (~accepted)
        selected_ix = torch.where(newly_accepted.unsqueeze(-1), ix, selected_ix)
        accepted = accepted | newly_accepted
        if bool(accepted.all()):
            break

    return torch.where(accepted.unsqueeze(-1), selected_ix, best_ix)


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None,
           score_n_candidates=64,
           score_prob_w=1.0,
           score_dist_w=0.25,
           score_sog_w=1.0,
           score_cog_w=1.0,
           score_heading_w=1.0,
           score_turn_w=0.5,
           score_dt_hours=1.0 / 6.0,
           score_dist_scale_km=2.0,
           score_resample_attempts=8,
           score_resample_max_penalty=1.2):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity", "pos_score", "pos_vicinity_score", "pos_resample", "pos_vicinity_resample"):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        if sample_mode in ("pos_score", "pos_vicinity_score"):
            ix = _score_next_point_candidates(
                model,
                seqs_cond,
                lat_logits,
                lon_logits,
                sog_logits,
                cog_logits,
                sample,
                score_n_candidates=score_n_candidates,
                score_prob_w=score_prob_w,
                score_dist_w=score_dist_w,
                score_sog_w=score_sog_w,
                score_cog_w=score_cog_w,
                score_heading_w=score_heading_w,
                score_turn_w=score_turn_w,
                score_dt_hours=score_dt_hours,
                score_dist_scale_km=score_dist_scale_km,
            )
        elif sample_mode in ("pos_resample", "pos_vicinity_resample"):
            ix = _sample_next_point_with_resampling(
                model,
                seqs_cond,
                lat_logits,
                lon_logits,
                sog_logits,
                cog_logits,
                sample,
                score_resample_attempts=score_resample_attempts,
                score_resample_max_penalty=score_resample_max_penalty,
                score_prob_w=score_prob_w,
                score_dist_w=score_dist_w,
                score_sog_w=score_sog_w,
                score_cog_w=score_cog_w,
                score_heading_w=score_heading_w,
                score_turn_w=score_turn_w,
                score_dt_hours=score_dt_hours,
                score_dist_scale_km=score_dist_scale_km,
            )
        else:
            # apply softmax to convert to probabilities
            lat_probs = F.softmax(lat_logits, dim=-1)
            lon_probs = F.softmax(lon_logits, dim=-1)
            sog_probs = F.softmax(sog_logits, dim=-1)
            cog_probs = F.softmax(cog_logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
                lon_ix = torch.multinomial(lon_probs, num_samples=1)
                sog_ix = torch.multinomial(sog_probs, num_samples=1)
                cog_ix = torch.multinomial(cog_probs, num_samples=1)
            else:
                _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
                _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
                _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
                _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

            ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs


@torch.no_grad()
def generate(model,
             seqs,
             steps,
             temperature=1.0,
             sample=True,
             sample_mode="pos_vicinity",
             r_vicinity=20,
             top_k=None,
             score_n_candidates=64,
             score_prob_w=1.0,
             score_dist_w=0.25,
             score_sog_w=1.0,
             score_cog_w=1.0,
             score_heading_w=1.0,
             score_turn_w=0.5,
             score_dt_hours=1.0 / 6.0,
             score_dist_scale_km=2.0,
             score_resample_attempts=8,
             score_resample_max_penalty=1.2):
    """Shared generation entry point for autoregressive sampling."""
    return globals()["sample"](
        model,
        seqs,
        steps,
        temperature=temperature,
        sample=sample,
        sample_mode=sample_mode,
        r_vicinity=r_vicinity,
        top_k=top_k,
        score_n_candidates=score_n_candidates,
        score_prob_w=score_prob_w,
        score_dist_w=score_dist_w,
        score_sog_w=score_sog_w,
        score_cog_w=score_cog_w,
        score_heading_w=score_heading_w,
        score_turn_w=score_turn_w,
        score_dt_hours=score_dt_hours,
        score_dist_scale_km=score_dist_scale_km,
        score_resample_attempts=score_resample_attempts,
        score_resample_max_penalty=score_resample_max_penalty,
    )


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_ccass_loss, d_ccass_rate, d_ccass_n, d_n = 0, 0, 0, 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks.to(self.device)
                loss_masks = masks[:, :-1]

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs,
                                                         masks=loss_masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                        logits, loss = model(seqs, masks=loss_masks, with_targets=True)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    base_loss = loss
                    ccass_loss = None
                    ccass_stats = None

                    if is_train and (not return_loss_tuple) and getattr(config, "use_ccass", False):
                        ccass_loss, ccass_stats = _confidence_constraint_scheduled_sampling_loss(
                            raw_model,
                            seqs,
                            masks,
                            config,
                            epoch,
                        )
                        if ccass_loss is not None:
                            loss = loss + getattr(config, "ccass_loss_w", 0.2) * ccass_loss

                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if ccass_loss is not None:
                    d_ccass_loss += ccass_loss.item() * seqs.shape[0]
                    d_ccass_rate += ccass_stats["use_pred_rate"] * seqs.shape[0]
                    d_ccass_n += seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            progress = min(1.0, progress)
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if ccass_loss is not None:
                        pbar.set_description(
                            f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}, "
                            f"base {base_loss.item():.5f}, ccass {ccass_loss.item():.5f}, "
                            f"ss {ccass_stats['use_pred_rate']:.3f}. lr {lr:e}"
                        )
                    else:
                        pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # tb logging
                    if TB_LOG:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch * n_batches + it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}.")
                elif d_ccass_n > 0:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, "
                        f"ccass {d_ccass_loss / d_ccass_n:.5f}, "
                        f"ss_rate {d_ccass_rate / d_ccass_n:.4f}, lr {lr:e}."
                    )
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = generate(raw_model,
                             seqs_init,
                             96 - init_seqlen,
                             temperature=1.0,
                             sample=True,
                             sample_mode=self.config.sample_mode,
                             r_vicinity=self.config.r_vicinity,
                             top_k=self.config.top_k,
                             score_n_candidates=getattr(self.config, "score_n_candidates", 64),
                             score_prob_w=getattr(self.config, "score_prob_w", 1.0),
                             score_dist_w=getattr(self.config, "score_dist_w", 0.25),
                             score_sog_w=getattr(self.config, "score_sog_w", 1.0),
                             score_cog_w=getattr(self.config, "score_cog_w", 1.0),
                             score_heading_w=getattr(self.config, "score_heading_w", 1.0),
                             score_turn_w=getattr(self.config, "score_turn_w", 0.5),
                             score_dt_hours=getattr(self.config, "score_dt_hours", 1.0 / 6.0),
                             score_dist_scale_km=getattr(self.config, "score_dist_scale_km", 2.0),
                             score_resample_attempts=getattr(self.config, "score_resample_attempts", 8),
                             score_resample_max_penalty=getattr(self.config, "score_resample_max_penalty", 1.2))

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)
